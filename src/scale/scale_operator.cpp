#include "scale_operator.h"
#include "hdf5/hdf5_file.h"
#include "hdf5/hdf5_csc_matrix.h"
#include "hdf5/hdf5_sparse_writer.h"
#include "utils/progress.h"
#include "utils/resource_monitor.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <hdf5.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sclean {

ScaleOperator::ScaleOperator(bool do_scale, bool do_center,
                               const std::vector<std::vector<double>>* vars_to_regress)
    : do_scale_(do_scale), do_center_(do_center),
      vars_to_regress_(vars_to_regress) {}

void ScaleOperator::compute_moments_row_chunked(
    HDF5File* file, const std::string& input_group,
    int64 n_genes, int64 n_cells,
    std::vector<double>& means, std::vector<double>& sds,
    ChunkScheduler& scheduler, int n_threads) {

    means.assign(n_genes, 0.0);
    sds.assign(n_genes, 0.0);
    std::vector<double> M2(n_genes, 0.0);  // sum of squared differences

    auto cfg = scheduler.schedule(n_genes, n_cells, OperationType::ScaleSimple, n_threads);

    // Row-chunked Welford's online algorithm
    bool oom_occurred = false;
    #pragma omp parallel if(n_threads > 1) num_threads(n_threads) shared(oom_occurred)
    {
        std::vector<double> buf;
        hid_t t_fid = (n_threads > 1) ? file->open_thread_handle(FileMode::ReadOnly) : file->file_id();

        #pragma omp for schedule(dynamic)
        for (int64 r = 0; r < n_genes; r += cfg.chunk_size) {
            if (oom_occurred) continue;
            int64 rc = std::min(cfg.chunk_size, n_genes - r);
            try {
                buf.resize(rc * n_cells);
                auto mat = file->open_csc_matrix(input_group, t_fid);
                mat->read_rows(buf.data(), r, rc, 0, n_cells);
            } catch (const std::bad_alloc&) {
                #pragma omp critical
                { oom_occurred = true; }
                continue;
            }

            // Welford for each row in chunk
            for (int64 i = 0; i < rc; ++i) {
                int64 gene = r + i;
                double mean = 0.0;
                double m2 = 0.0;
                int64 count = 0;

                for (int64 j = 0; j < n_cells; ++j) {
                    double x = buf[i * n_cells + j];
                    count++;
                    double delta = x - mean;
                    mean += delta / count;
                    double delta2 = x - mean;
                    m2 += delta * delta2;
                }

                means[gene] = mean;
                M2[gene] = m2;
                double var = (count > 1) ? std::max(0.0, m2 / (count - 1)) : 0.0;
                sds[gene] = std::sqrt(var);
            }
        }
    }

    if (oom_occurred) {
        throw std::bad_alloc();
    }
}

ScaleResult ScaleOperator::compute_stats(
    HDF5File* file, const std::string& input_group,
    int64 n_genes, int64 n_cells,
    ChunkScheduler& scheduler, int n_threads) {

    ScaleResult result;
    result.n_genes = n_genes;
    result.n_cells = n_cells;

    compute_moments_row_chunked(file, input_group, n_genes, n_cells,
                                 result.gene_means, result.gene_sds,
                                 scheduler, n_threads);

    return result;
}

ScaleResult ScaleOperator::run(
    HDF5File* file,
    const std::string& input_group,
    const std::string& output_group,
    ChunkScheduler& scheduler,
    int n_threads) {

    // Refresh resource state
    scheduler.refresh_available_ram();

    // Get dimensions
    auto input = file->open_csc_matrix(input_group);
    int64 n_genes = input->n_rows();
    int64 n_cells = input->n_cols();
    int64 nnz = input->nnz();

    // Step 1: Compute gene means and SDs
    ProgressReporter progress("ScaleData", 2,
                              ProgressReporter::is_verbose());
    progress.message("Computing gene moments...");
    std::vector<double> means, sds;
    bool moments_ok = false;
    auto oom_cfg = scheduler.schedule(n_genes, n_cells, OperationType::ScaleSimple, n_threads);
    for (int attempt = 0; attempt < 3 && !moments_ok; ++attempt) {
        try {
            compute_moments_row_chunked(file, input_group, n_genes, n_cells,
                                        means, sds, scheduler, n_threads);
            moments_ok = true;
            scheduler.clear_override();
        } catch (const std::bad_alloc&) {
            auto snap = ResourceMonitor().snapshot();
            if (!scheduler.shrink_and_retry(n_genes, n_cells,
                    OperationType::ScaleSimple, n_threads, oom_cfg)) {
                scheduler.clear_override();
                REprintf("[scLean] ScaleData: FATAL after %d retries "
                         "(free: %lld MB, RSS: %lld MB)\n",
                         attempt + 1, (long long)(snap.free_ram >> 20),
                         (long long)(snap.current_rss >> 20));
                throw;
            }
            REprintf("[scLean] ScaleData: OOM, shrinking chunk to %lld, retry %d/3\n",
                     (long long)oom_cfg.chunk_size, attempt + 1);
        }
    }
    progress.step();

    // Step 2: Write scaled residuals column by column
    // After centering, all entries become non-zero (zeros → -mean/sd),
    // so the output can be fully dense. Use n_genes*n_cells as upper bound.
    progress.message("Scaling and writing residuals...");
    int64 max_possible_nnz = n_genes * n_cells;
    HDF5SparseWriter writer(file, output_group, n_genes, n_cells,
                             max_possible_nnz, 10 * 1024 * 1024, 1);

    auto cfg = scheduler.schedule(n_genes, n_cells, OperationType::ScaleSimple, n_threads);

    // Read sparse column chunks, scale, write sparse
    std::string data_path = input_group + "/data";
    std::string idx_path = input_group + "/indices";
    std::string indptr_path = input_group + "/indptr";
    hid_t fid = file->file_id();

    #pragma omp parallel if(n_threads > 1) num_threads(n_threads)
    {
        hid_t t_fid = (n_threads > 1) ? file->open_thread_handle(FileMode::ReadOnly) : file->file_id();

        #pragma omp for schedule(dynamic)
        for (int64 c = 0; c < n_cells; c += cfg.chunk_size) {
            int64 cc = std::min(cfg.chunk_size, n_cells - c);

            // Read indptr slice
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

            if (chunk_nnz > 0) {
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

                // Scale each column and write
                #pragma omp critical
                {
                    for (int64 j = 0; j < cc; ++j) {
                        int64 col_start = ip[j] - ip[0];
                        int64 col_end = ip[j + 1] - ip[0];
                        int64 col_nnz = col_end - col_start;

                        std::vector<double> out_vals;
                        std::vector<int32> out_idx;

                        for (int64 k = col_start; k < col_end; ++k) {
                            int64 gene = idxs[k];
                            double v = vals[k];
                            double mean = means[gene];
                            double sd = sds[gene];

                            double scaled = v;
                            if (do_center_) scaled -= mean;
                            if (do_scale_ && sd > 0) scaled /= sd;

                            if (scaled != 0.0) {
                                out_vals.push_back(scaled);
                                out_idx.push_back(static_cast<int32>(gene));
                            }
                        }

                        writer.write_column(out_vals.data(), out_idx.data(),
                                            static_cast<int64>(out_vals.size()));
                    }
                }
            } else {
                #pragma omp critical
                {
                    for (int64 j = 0; j < cc; ++j) {
                        writer.write_column(nullptr, nullptr, 0);
                    }
                }
            }
        }
    }

    writer.finalize();

    progress.done();

    ScaleResult result;
    result.gene_means = means;
    result.gene_sds = sds;
    result.n_genes = n_genes;
    result.n_cells = n_cells;
    return result;
}

} // namespace sclean
