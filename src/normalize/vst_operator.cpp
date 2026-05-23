#include "vst_operator.h"
#include "normalize_internal.h"
#include "hdf5/hdf5_file.h"
#include "hdf5/hdf5_csc_matrix.h"
#include "utils/progress.h"
#include "utils/resource_monitor.h"
#include <cmath>
#include <algorithm>
#include <hdf5.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sclean {

VSTOperator::VSTOperator(int n_top_features, double loess_span, int n_bins)
    : n_top_features_(n_top_features), loess_span_(loess_span), n_bins_(n_bins) {}

// --- Dense path: read dense column chunks (fast, memory-heavy) ---

void VSTOperator::compute_mean_variance(
    HDF5File* file, const std::string& input_group,
    int64 n_genes, int64 n_cells,
    std::vector<double>& means, std::vector<double>& variances,
    ChunkScheduler& scheduler, int n_threads) {

    std::vector<GeneWelford> global_stats(n_genes);

    auto cfg = scheduler.schedule(n_genes, n_cells, OperationType::VST, n_threads);

    bool oom_occurred = false;
    #pragma omp parallel if(n_threads > 1) num_threads(n_threads) shared(oom_occurred)
    {
        // Serialize HDF5CSCMatrix construction to avoid H5Dopen2 race
        std::unique_ptr<HDF5CSCMatrix> t_mat;
        #pragma omp critical
        {
            hid_t t_fid = (n_threads > 1) ? file->open_thread_handle(FileMode::ReadOnly) : -1;
            t_mat = std::unique_ptr<HDF5CSCMatrix>(
                new HDF5CSCMatrix(file, input_group, t_fid));
        }

        // Per-thread local accumulators
        std::vector<GeneWelford> local_stats(n_genes);

        std::vector<double> buf;
        #pragma omp for schedule(dynamic)
        for (int64 c = 0; c < n_cells; c += cfg.chunk_size) {
            if (oom_occurred) continue;
            int64 cc = std::min(cfg.chunk_size, n_cells - c);
            try {
                buf.resize(n_genes * cc);
                t_mat->read_cols(buf.data(), 0, n_genes, c, cc);
            } catch (const std::bad_alloc&) {
                #pragma omp critical
                { oom_occurred = true; }
                continue;
            }

            // Pure computation: accumulate stats from dense chunk
            accumulate_dense_chunk(buf.data(), n_genes, cc, local_stats);
        }

        // Merge thread-local stats into global
        #pragma omp critical
        {
            merge_stats_arrays(local_stats, global_stats, n_genes);
        }
    }

    if (oom_occurred) {
        throw std::bad_alloc();
    }

    // Pure computation: extract means and variances
    finalize_stats_arrays(global_stats, means, variances);
}

// --- Sparse path: read CSC raw data, no dense buffer ---

void VSTOperator::compute_mean_variance_sparse(
    HDF5File* file, const std::string& input_group,
    int64 n_genes, int64 n_cells,
    std::vector<double>& means, std::vector<double>& variances,
    ChunkScheduler& scheduler, int n_threads) {

    std::vector<GeneWelford> global_stats(n_genes);

    auto cfg = scheduler.schedule(n_genes, n_cells, OperationType::VST, n_threads);

    std::string data_path = input_group + "/data";
    std::string idx_path = input_group + "/indices";
    std::string indptr_path = input_group + "/indptr";

    #pragma omp parallel if(n_threads > 1) num_threads(n_threads)
    {
        hid_t t_fid = (n_threads > 1) ? file->open_thread_handle(FileMode::ReadOnly) : file->file_id();

        // Per-thread local accumulators
        std::vector<GeneWelford> local_stats(n_genes);

        #pragma omp for schedule(dynamic)
        for (int64 c = 0; c < n_cells; c += cfg.chunk_size) {
            int64 cc = std::min(cfg.chunk_size, n_cells - c);

            // Read indptr for this chunk
            std::vector<int64> ip(cc + 1);
            {
                hid_t d = H5Dopen2(t_fid, indptr_path.c_str(), H5P_DEFAULT);
                hsize_t s[1] = {static_cast<hsize_t>(c)};
                hsize_t cnt[1] = {static_cast<hsize_t>(cc + 1)};
                hid_t ms = H5Screate_simple(1, cnt, nullptr);
                hid_t fs = H5Dget_space(d);
                H5Sselect_hyperslab(fs, H5S_SELECT_SET, s, nullptr, cnt, nullptr);
                H5Dread(d, H5T_NATIVE_INT64, ms, fs, H5P_DEFAULT, ip.data());
                H5Sclose(fs); H5Sclose(ms); H5Dclose(d);
            }

            int64 chunk_nnz = ip[cc] - ip[0];
            if (chunk_nnz == 0) continue;

            // Read data and indices for this chunk
            std::vector<double> vals(chunk_nnz);
            std::vector<int32> idxs(chunk_nnz);
            {
                hsize_t s[1] = {static_cast<hsize_t>(ip[0])};
                hsize_t cnt[1] = {static_cast<hsize_t>(chunk_nnz)};
                hid_t ms = H5Screate_simple(1, cnt, nullptr);

                hid_t d1 = H5Dopen2(t_fid, data_path.c_str(), H5P_DEFAULT);
                hid_t fs1 = H5Dget_space(d1);
                H5Sselect_hyperslab(fs1, H5S_SELECT_SET, s, nullptr, cnt, nullptr);
                H5Dread(d1, H5T_NATIVE_DOUBLE, ms, fs1, H5P_DEFAULT, vals.data());
                H5Sclose(fs1); H5Dclose(d1);

                hid_t d2 = H5Dopen2(t_fid, idx_path.c_str(), H5P_DEFAULT);
                hid_t fs2 = H5Dget_space(d2);
                H5Sselect_hyperslab(fs2, H5S_SELECT_SET, s, nullptr, cnt, nullptr);
                H5Dread(d2, H5T_NATIVE_INT32, ms, fs2, H5P_DEFAULT, idxs.data());
                H5Sclose(fs2); H5Dclose(d2);

                H5Sclose(ms);
            }

            // Pure computation: accumulate stats from sparse chunk
            accumulate_sparse_chunk(vals.data(), idxs.data(), ip.data(),
                                    cc, local_stats);
        }

        // Merge thread-local stats into global
        #pragma omp critical
        {
            merge_stats_arrays(local_stats, global_stats, n_genes);
        }
    }

    // Pure computation: extract means and variances
    finalize_stats_arrays(global_stats, means, variances);
}

VSTResult VSTOperator::run(
    HDF5File* file, const std::string& input_group,
    int64 n_genes, int64 n_cells,
    ChunkScheduler& scheduler, int n_threads) {

    VSTResult result;

    // Refresh resource state
    scheduler.refresh_available_ram();
    auto pre_cfg = scheduler.schedule(n_genes, n_cells, OperationType::VST, n_threads);
    bool use_sparse = (pre_cfg.bottleneck == Bottleneck::MemoryBound ||
                       pre_cfg.bottleneck == Bottleneck::BothBound);

    ProgressReporter progress("FindVariableFeatures", 3,
                              ProgressReporter::is_verbose());

    // Pass 1: Compute per-gene mean and variance (with OOM fallback)
    progress.message("Computing gene mean and variance...");
    bool pass1_done = false;
    for (int attempt = 0; attempt < 3 && !pass1_done; ++attempt) {
        try {
            if (use_sparse) {
                compute_mean_variance_sparse(file, input_group, n_genes, n_cells,
                                             result.gene_means, result.gene_variances,
                                             scheduler, n_threads);
            } else {
                compute_mean_variance(file, input_group, n_genes, n_cells,
                                      result.gene_means, result.gene_variances,
                                      scheduler, n_threads);
            }
            pass1_done = true;
        } catch (const std::bad_alloc&) {
            if (!use_sparse) {
                use_sparse = true;
                REprintf("[scLean] FindVariableFeatures: OOM, switching to sparse path\n");
            } else {
                auto snap = ResourceMonitor().snapshot();
                if (!scheduler.shrink_and_retry(
                        n_genes, n_cells, OperationType::VST,
                        n_threads, pre_cfg)) {
                    REprintf("[scLean] FindVariableFeatures: FATAL after %d retries "
                             "(free: %lld MB, RSS: %lld MB)\n",
                             attempt + 1, (long long)(snap.free_ram >> 20),
                             (long long)(snap.current_rss >> 20));
                    throw;
                }
                REprintf("[scLean] FindVariableFeatures: shrinking chunk to %lld, retry %d/3\n",
                         (long long)pre_cfg.chunk_size, attempt + 1);
            }
        }
    }
    progress.step();

    // Log-transform for mean-variance relationship
    std::vector<double> log_means(n_genes), log_variances(n_genes);
    for (int64 i = 0; i < n_genes; ++i) {
        log_means[i] = (result.gene_means[i] > 0) ?
            std::log10(result.gene_means[i]) : -INFINITY;
        log_variances[i] = (result.gene_variances[i] > 0) ?
            std::log10(result.gene_variances[i]) : -INFINITY;
    }

    // Fit LOESS (binned approximation) — pure computation
    progress.message("Fitting mean-variance trend...");
    std::vector<double> fitted;
    fit_loess_binned(log_means, log_variances, fitted, n_bins_, loess_span_);
    progress.step();

    // Compute standardized variance
    result.vst_variances.resize(n_genes);
    for (int64 i = 0; i < n_genes; ++i) {
        if (fitted[i] > 0) {
            result.vst_variances[i] =
                (result.gene_variances[i] - std::pow(10.0, fitted[i])) /
                std::pow(10.0, fitted[i] * 0.5);
        } else {
            result.vst_variances[i] = 0.0;
        }
    }

    // Select top variable features — pure computation
    progress.message("Selecting variable features...");
    select_top_features(result.vst_variances,
                         result.variable_features, n_top_features_);

    result.n_variable = 0;
    for (auto v : result.variable_features) {
        if (v) result.n_variable++;
    }

    progress.done();
    return result;
}

} // namespace sclean
