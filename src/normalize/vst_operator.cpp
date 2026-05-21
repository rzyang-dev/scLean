#include "vst_operator.h"
#include "hdf5/hdf5_file.h"
#include "hdf5/hdf5_csc_matrix.h"
#include "utils/progress.h"
#include "utils/resource_monitor.h"
#include <cmath>
#include <algorithm>
#include <numeric>
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

    means.assign(n_genes, 0.0);
    variances.assign(n_genes, 0.0);
    std::vector<double> M2(n_genes, 0.0);
    std::vector<int64> counts(n_genes, 0);

    auto cfg = scheduler.schedule(n_genes, n_cells, OperationType::VST, n_threads);

    bool oom_occurred = false;
    #pragma omp parallel num_threads(n_threads) shared(oom_occurred)
    {
        hid_t t_fid = (n_threads > 1) ? file->open_thread_handle(FileMode::ReadOnly) : -1;
        auto t_mat = std::unique_ptr<HDF5CSCMatrix>(
            new HDF5CSCMatrix(file, input_group, t_fid));

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

            for (int64 i = 0; i < n_genes; ++i) {
                double mean = 0.0;
                double m2 = 0.0;
                int64 cnt = 0;

                for (int64 j = 0; j < cc; ++j) {
                    double x = buf[i * cc + j];
                    cnt++;
                    double delta = x - mean;
                    mean += delta / cnt;
                    double delta2 = x - mean;
                    m2 += delta * delta2;
                }

                #pragma omp critical
                {
                    double old_mean = means[i];
                    double old_m2 = M2[i];
                    int64 old_n = counts[i];

                    int64 new_n = old_n + cnt;
                    double delta = mean - old_mean;
                    means[i] = old_mean + delta * cnt / new_n;
                    M2[i] = old_m2 + m2 + delta * delta * old_n * cnt / new_n;
                    counts[i] = new_n;
                }
            }
        }
    }

    if (oom_occurred) {
        throw std::bad_alloc();
    }

    for (int64 i = 0; i < n_genes; ++i) {
        variances[i] = (counts[i] > 1) ? M2[i] / (counts[i] - 1) : 0.0;
    }
}

// --- Sparse path: read CSC raw data, no dense buffer ---

void VSTOperator::compute_mean_variance_sparse(
    HDF5File* file, const std::string& input_group,
    int64 n_genes, int64 n_cells,
    std::vector<double>& means, std::vector<double>& variances,
    ChunkScheduler& scheduler, int n_threads) {

    means.assign(n_genes, 0.0);
    variances.assign(n_genes, 0.0);
    std::vector<double> M2(n_genes, 0.0);
    std::vector<int64> counts(n_genes, 0);

    auto cfg = scheduler.schedule(n_genes, n_cells, OperationType::VST, n_threads);

    std::string data_path = input_group + "/data";
    std::string idx_path = input_group + "/indices";
    std::string indptr_path = input_group + "/indptr";

    #pragma omp parallel if(n_threads > 1) num_threads(n_threads)
    {
        hid_t t_fid = (n_threads > 1) ? file->open_thread_handle(FileMode::ReadOnly) : file->file_id();

        // Per-thread local accumulators
        std::vector<double> local_means(n_genes, 0.0);
        std::vector<double> local_M2(n_genes, 0.0);
        std::vector<int64> local_counts(n_genes, 0);

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

            // Welford per column, processing only non-zero entries
            for (int64 j = 0; j < cc; ++j) {
                int64 col_start = ip[j] - ip[0];
                int64 col_end = ip[j + 1] - ip[0];

                for (int64 k = col_start; k < col_end; ++k) {
                    int64 gene = idxs[k];
                    double x = vals[k];

                    int64& cnt = local_counts[gene];
                    double& mean = local_means[gene];
                    double& m2 = local_M2[gene];

                    cnt++;
                    double delta = x - mean;
                    mean += delta / cnt;
                    double delta2 = x - mean;
                    m2 += delta * delta2;
                }
            }
        }

        // Merge per-thread accumulators into global
        #pragma omp critical
        {
            for (int64 i = 0; i < n_genes; ++i) {
                if (local_counts[i] == 0) continue;
                double old_mean = means[i];
                double old_m2 = M2[i];
                int64 old_n = counts[i];

                int64 new_n = old_n + local_counts[i];
                double delta = local_means[i] - old_mean;
                means[i] = old_mean + delta * local_counts[i] / new_n;
                M2[i] = old_m2 + local_M2[i] + delta * delta * old_n * local_counts[i] / new_n;
                counts[i] = new_n;
            }
        }
    }

    for (int64 i = 0; i < n_genes; ++i) {
        variances[i] = (counts[i] > 1) ? M2[i] / (counts[i] - 1) : 0.0;
    }
}

void VSTOperator::fit_loess_binned(
    const std::vector<double>& log_means,
    const std::vector<double>& log_variances,
    std::vector<double>& fitted,
    int n_bins, double span) {

    int64 n = static_cast<int64>(log_means.size());
    fitted.resize(n);

    // Filter valid entries
    std::vector<int64> valid_idx;
    for (int64 i = 0; i < n; ++i) {
        if (std::isfinite(log_means[i]) && std::isfinite(log_variances[i])) {
            valid_idx.push_back(i);
        }
    }

    if (valid_idx.empty()) {
        std::fill(fitted.begin(), fitted.end(), 0.0);
        return;
    }

    // Sort valid entries by log_mean
    std::sort(valid_idx.begin(), valid_idx.end(),
              [&](int64 a, int64 b) { return log_means[a] < log_means[b]; });

    // Bin and compute median variance per bin
    int64 n_valid = static_cast<int64>(valid_idx.size());
    int64 per_bin = std::max(static_cast<int64>(1), n_valid / n_bins);

    std::vector<double> bin_centers, bin_variances;

    for (int64 b = 0; b < n_bins; ++b) {
        int64 start = b * per_bin;
        int64 end = std::min(start + per_bin, n_valid);
        if (start >= n_valid) break;

        double sum_mean = 0.0;
        std::vector<double> bin_vars;
        for (int64 k = start; k < end; ++k) {
            sum_mean += log_means[valid_idx[k]];
            bin_vars.push_back(log_variances[valid_idx[k]]);
        }

        // Median variance per bin
        std::sort(bin_vars.begin(), bin_vars.end());
        double med_var = bin_vars[bin_vars.size() / 2];

        bin_centers.push_back(sum_mean / (end - start));
        bin_variances.push_back(med_var);
    }

    // Linear interpolation across bins
    for (int64 i = 0; i < n; ++i) {
        if (!std::isfinite(log_means[i])) {
            fitted[i] = 0.0;
            continue;
        }

        double lm = log_means[i];

        // Find surrounding bins
        if (lm <= bin_centers.front()) {
            fitted[i] = bin_variances.front();
        } else if (lm >= bin_centers.back()) {
            fitted[i] = bin_variances.back();
        } else {
            // Linear interpolation between bins
            for (size_t b = 0; b < bin_centers.size() - 1; ++b) {
                if (lm >= bin_centers[b] && lm <= bin_centers[b + 1]) {
                    double t = (lm - bin_centers[b]) /
                               (bin_centers[b + 1] - bin_centers[b]);
                    fitted[i] = bin_variances[b] * (1.0 - t) +
                                bin_variances[b + 1] * t;
                    break;
                }
            }
        }
    }
}

void VSTOperator::select_top_features(
    const std::vector<double>& vst_variances,
    std::vector<int8_t>& variable,
    int n_select) {

    int64 n = static_cast<int64>(vst_variances.size());
    variable.assign(n, 0);

    if (n_select <= 0) return;

    if (n_select >= n) {
        std::fill(variable.begin(), variable.end(), 1);
        return;
    }

    // Partial sort to find top N
    std::vector<std::pair<double, int64>> ranked;
    ranked.reserve(n);
    for (int64 i = 0; i < n; ++i) {
        if (std::isfinite(vst_variances[i])) {
            ranked.emplace_back(vst_variances[i], i);
        }
    }

    if (ranked.empty()) return;

    int64 actual_select = std::min(static_cast<int64>(n_select),
                                    static_cast<int64>(ranked.size()));

    std::nth_element(ranked.begin(), ranked.begin() + actual_select,
                     ranked.end(), std::greater<>{});

    for (int64 k = 0; k < actual_select; ++k) {
        variable[ranked[k].second] = 1;
    }
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

    // Fit LOESS (binned approximation)
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

    // Select top variable features
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
