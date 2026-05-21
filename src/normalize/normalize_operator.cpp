#include "normalize_operator.h"
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

NormalizeOperator::NormalizeOperator(NormalizeMethod method,
                                       double scale_factor, bool do_pseudocount)
    : method_(method), scale_factor_(scale_factor), do_pseudocount_(do_pseudocount) {}

// ============================================================
// Size factor computation (first pass)
// ============================================================

std::vector<double> NormalizeOperator::compute_libsize_chunked(
    HDF5File* file, const std::string& input_group,
    int64 n_cells, int64 n_rows,
    ChunkScheduler& scheduler, int n_threads) {

    std::vector<double> libsize(n_cells, 0.0);
    auto cfg = scheduler.schedule(n_rows, n_cells, OperationType::Normalize, n_threads);

    std::string data_path = input_group + "/data";
    std::string idx_path = input_group + "/indices";
    std::string indptr_path = input_group + "/indptr";

    #pragma omp parallel if(n_threads > 1) num_threads(n_threads)
    {
        hid_t t_fid = (n_threads > 1) ? file->open_thread_handle(FileMode::ReadOnly) : file->file_id();
        #pragma omp for schedule(dynamic)
        for (int64 c = 0; c < n_cells; c += cfg.chunk_size) {
            int64 cc = std::min(cfg.chunk_size, n_cells - c);

            // Read indptr for this chunk
            std::vector<int64> indptr_chunk(cc + 1);
            {
                hid_t d = H5Dopen2(t_fid, indptr_path.c_str(), H5P_DEFAULT);
                hsize_t start[1] = {static_cast<hsize_t>(c)};
                hsize_t count[1] = {static_cast<hsize_t>(cc + 1)};
                hid_t ms = H5Screate_simple(1, count, nullptr);
                hid_t fs = H5Dget_space(d);
                H5Sselect_hyperslab(fs, H5S_SELECT_SET, start, nullptr, count, nullptr);
                H5Dread(d, H5T_NATIVE_INT64, ms, fs, H5P_DEFAULT, indptr_chunk.data());
                H5Sclose(fs); H5Sclose(ms); H5Dclose(d);
            }

            int64 chunk_nnz = indptr_chunk[cc] - indptr_chunk[0];
            if (chunk_nnz == 0) continue;

            // Read data values
            std::vector<double> vals(chunk_nnz);
            {
                hid_t d = H5Dopen2(t_fid, data_path.c_str(), H5P_DEFAULT);
                hsize_t start[1] = {static_cast<hsize_t>(indptr_chunk[0])};
                hsize_t count[1] = {static_cast<hsize_t>(chunk_nnz)};
                hid_t ms = H5Screate_simple(1, count, nullptr);
                hid_t fs = H5Dget_space(d);
                H5Sselect_hyperslab(fs, H5S_SELECT_SET, start, nullptr, count, nullptr);
                H5Dread(d, H5T_NATIVE_DOUBLE, ms, fs, H5P_DEFAULT, vals.data());
                H5Sclose(fs); H5Sclose(ms); H5Dclose(d);
            }

            // For each column, sum values
            for (int64 j = 0; j < cc; ++j) {
                int64 start = indptr_chunk[j] - indptr_chunk[0];
                int64 end = indptr_chunk[j + 1] - indptr_chunk[0];
                double sum = 0.0;
                for (int64 k = start; k < end; ++k) {
                    sum += vals[k];
                }
                libsize[c + j] = sum;
            }
        }
    }

    return libsize;
}

std::vector<double> NormalizeOperator::compute_geometric_means_chunked(
    HDF5File* file, const std::string& input_group,
    int64 n_cells, int64 n_rows,
    ChunkScheduler& scheduler, int n_threads) {

    std::vector<double> log_sums(n_cells, 0.0);
    std::vector<int> nz_counts(n_cells, 0);
    auto cfg = scheduler.schedule(n_rows, n_cells, OperationType::Normalize, n_threads);

    std::string data_path = input_group + "/data";
    std::string indptr_path = input_group + "/indptr";

    #pragma omp parallel if(n_threads > 1) num_threads(n_threads)
    {
        hid_t t_fid = (n_threads > 1) ? file->open_thread_handle(FileMode::ReadOnly) : file->file_id();
        #pragma omp for schedule(dynamic)
        for (int64 c = 0; c < n_cells; c += cfg.chunk_size) {
            int64 cc = std::min(cfg.chunk_size, n_cells - c);

            std::vector<int64> indptr_chunk(cc + 1);
            {
                hid_t d = H5Dopen2(t_fid, indptr_path.c_str(), H5P_DEFAULT);
                hsize_t start[1] = {static_cast<hsize_t>(c)};
                hsize_t count[1] = {static_cast<hsize_t>(cc + 1)};
                hid_t ms = H5Screate_simple(1, count, nullptr);
                hid_t fs = H5Dget_space(d);
                H5Sselect_hyperslab(fs, H5S_SELECT_SET, start, nullptr, count, nullptr);
                H5Dread(d, H5T_NATIVE_INT64, ms, fs, H5P_DEFAULT, indptr_chunk.data());
                H5Sclose(fs); H5Sclose(ms); H5Dclose(d);
            }

            int64 chunk_nnz = indptr_chunk[cc] - indptr_chunk[0];
            if (chunk_nnz == 0) continue;

            std::vector<double> vals(chunk_nnz);
            {
                hid_t d = H5Dopen2(t_fid, data_path.c_str(), H5P_DEFAULT);
                hsize_t start[1] = {static_cast<hsize_t>(indptr_chunk[0])};
                hsize_t count[1] = {static_cast<hsize_t>(chunk_nnz)};
                hid_t ms = H5Screate_simple(1, count, nullptr);
                hid_t fs = H5Dget_space(d);
                H5Sselect_hyperslab(fs, H5S_SELECT_SET, start, nullptr, count, nullptr);
                H5Dread(d, H5T_NATIVE_DOUBLE, ms, fs, H5P_DEFAULT, vals.data());
                H5Sclose(fs); H5Sclose(ms); H5Dclose(d);
            }

            for (int64 j = 0; j < cc; ++j) {
                int64 start = indptr_chunk[j] - indptr_chunk[0];
                int64 end = indptr_chunk[j + 1] - indptr_chunk[0];
                double log_sum = 0.0;
                int nz = 0;
                for (int64 k = start; k < end; ++k) {
                    double v = vals[k];
                    if (v > 0) {
                        log_sum += std::log(v);
                        nz++;
                    }
                }
                log_sums[c + j] = log_sum;
                nz_counts[c + j] = nz;
            }
        }
    }

    std::vector<double> geo_means(n_cells);
    for (int64 j = 0; j < n_cells; ++j) {
        geo_means[j] = (nz_counts[j] > 0) ?
            std::exp(log_sums[j] / nz_counts[j]) : 1.0;
    }
    return geo_means;
}

std::vector<double> NormalizeOperator::compute_size_factors(
    HDF5File* file, const std::string& input_group,
    int64 n_cells, int64 n_rows, int64,
    ChunkScheduler& scheduler, int n_threads) {

    switch (method_) {
        case NormalizeMethod::LogNormalize:
        case NormalizeMethod::RelativeCounts:
            return compute_libsize_chunked(file, input_group, n_cells, n_rows,
                                            scheduler, n_threads);
        case NormalizeMethod::CLR:
            return compute_geometric_means_chunked(file, input_group, n_cells, n_rows,
                                                    scheduler, n_threads);
    }
    return {};
}

// ============================================================
// Normalization transforms
// ============================================================

void NormalizeOperator::normalize_sparse_chunk(
    const double* in_data, const int32* in_indices,
    const int64* in_indptr, int64 col_count,
    const double* size_factors_segment,
    HDF5SparseWriter* writer) {

    switch (method_) {
        case NormalizeMethod::LogNormalize:
            for (int64 j = 0; j < col_count; ++j) {
                double sf = size_factors_segment[j];
                double norm_factor = (sf > 0) ? scale_factor_ / sf : 1.0;
                int64 start = in_indptr[j] - in_indptr[0];
                int64 end = in_indptr[j + 1] - in_indptr[0];
                int64 nnz = end - start;

                std::vector<double> out_vals(nnz);
                for (int64 k = 0; k < nnz; ++k) {
                    out_vals[k] = std::log1p(in_data[start + k] * norm_factor);
                }
                writer->write_column(out_vals.data(), in_indices + start, nnz);
            }
            break;

        case NormalizeMethod::RelativeCounts:
            for (int64 j = 0; j < col_count; ++j) {
                double sf = size_factors_segment[j];
                double norm_factor = (sf > 0) ? scale_factor_ / sf : 1.0;
                int64 start = in_indptr[j] - in_indptr[0];
                int64 end = in_indptr[j + 1] - in_indptr[0];
                int64 nnz = end - start;

                std::vector<double> out_vals(nnz);
                for (int64 k = 0; k < nnz; ++k) {
                    out_vals[k] = in_data[start + k] * norm_factor;
                }
                writer->write_column(out_vals.data(), in_indices + start, nnz);
            }
            break;

        case NormalizeMethod::CLR:
            // CLR does not preserve zeros — we need dense per column
            // Write zeros as non-zero entries
            for (int64 j = 0; j < col_count; ++j) {
                double gm = size_factors_segment[j];
                int64 start = in_indptr[j] - in_indptr[0];
                int64 end = in_indptr[j + 1] - in_indptr[0];
                int64 nnz = end - start;

                // Build full column (dense) from sparse input
                // For efficiency, only write entries where result != 0
                std::vector<double> col_vals;
                std::vector<int32> col_idx;
                for (int64 k = 0; k < nnz; ++k) {
                    double v = in_data[start + k];
                    double pseudo = do_pseudocount_ ? 1.0 : 0.0;
                    double r = std::log((v + pseudo) / gm);
                    if (r != 0.0) {
                        col_vals.push_back(r);
                        col_idx.push_back(in_indices[start + k]);
                    }
                }
                writer->write_column(col_vals.data(), col_idx.data(),
                                     static_cast<int64>(col_vals.size()));
            }
            break;
    }
}

void NormalizeOperator::normalize_dense_chunk(
    const double* in_buf, int64 n_rows, int64 n_cols,
    const double* size_factors_segment,
    HDF5SparseWriter* writer) {

    for (int64 j = 0; j < n_cols; ++j) {
        double sf = size_factors_segment[j];
        double norm_factor = (sf > 0) ? scale_factor_ / sf : 1.0;

        std::vector<double> col_vals;
        std::vector<int32> col_idx;

        for (int64 i = 0; i < n_rows; ++i) {
            double v = in_buf[i * n_cols + j];
            double r = 0.0;

            switch (method_) {
                case NormalizeMethod::LogNormalize:
                    r = std::log1p(v * norm_factor);
                    break;
                case NormalizeMethod::CLR: {
                    double pseudo = do_pseudocount_ ? 1.0 : 0.0;
                    r = std::log((v + pseudo) / sf);
                    break;
                }
                case NormalizeMethod::RelativeCounts:
                    r = v * norm_factor;
                    break;
            }

            if (r != 0.0) {
                col_vals.push_back(r);
                col_idx.push_back(static_cast<int32>(i));
            }
        }

        writer->write_column(col_vals.data(), col_idx.data(),
                             static_cast<int64>(col_vals.size()));
    }
}

// ============================================================
// Main run method
// ============================================================

NormalizeResult NormalizeOperator::run(
    HDF5File* file,
    const std::string& input_group,
    const std::string& output_group,
    ChunkScheduler& scheduler,
    int n_threads) {

    // Refresh resource state
    scheduler.refresh_available_ram();

    // Open input matrix to get dimensions
    auto input = file->open_csc_matrix(input_group);
    int64 n_rows = input->n_rows();
    int64 n_cells = input->n_cols();
    int64 nnz = input->nnz();

    bool verbose = ProgressReporter::is_verbose();
    ProgressReporter progress("NormalizeData", 2, verbose);

    // Step 1: Compute size factors
    if (verbose) progress.message("Computing size factors...");
    auto size_factors = compute_size_factors(file, input_group,
        n_cells, n_rows, nnz, scheduler, n_threads);
    if (verbose) progress.step();

    // Step 2: Create output writer (same nnz for zero-preserving methods,
    //         estimate nnz for CLR)
    int64 est_output_nnz = nnz;
    if (method_ == NormalizeMethod::CLR) {
        // CLR produces non-zero for all entries with pseudocount
        est_output_nnz = n_rows * n_cells;
    }

    HDF5SparseWriter writer(file, output_group, n_rows, n_cells,
                             est_output_nnz, 10 * 1024 * 1024, 3);

    // Step 3: Read input sparse columns, normalize, write output
    progress.message("Normalizing...");
    auto cfg = scheduler.schedule(n_rows, n_cells, OperationType::Normalize, n_threads);

    if (method_ == NormalizeMethod::CLR) {
        // CLR: read dense chunks (output may have different sparsity pattern)
        // MemoryBound → process single column at a time to avoid dense buffer
        if (cfg.bottleneck == Bottleneck::MemoryBound ||
            cfg.bottleneck == Bottleneck::BothBound) {
            cfg.chunk_size = 1;
        }
        bool oom_occurred = false;
        #pragma omp parallel if(n_threads > 1) num_threads(n_threads) shared(oom_occurred)
        {
            hid_t t_fid = (n_threads > 1) ? file->open_thread_handle(FileMode::ReadOnly) : -1;
            auto t_input = (n_threads > 1)
                ? file->open_csc_matrix(input_group, t_fid)
                : std::unique_ptr<HDF5CSCMatrix>(nullptr);
            HDF5CSCMatrix* reader = (n_threads > 1) ? t_input.get() : input.get();

            std::vector<double> in_buf;
            #pragma omp for schedule(dynamic)
            for (int64 c = 0; c < n_cells; c += cfg.chunk_size) {
                if (oom_occurred) continue;
                int64 cc = std::min(cfg.chunk_size, n_cells - c);
                try {
                    in_buf.resize(n_rows * cc);
                    reader->read_cols(in_buf.data(), 0, n_rows, c, cc);
                } catch (const std::bad_alloc&) {
                    #pragma omp critical
                    { oom_occurred = true; }
                    continue;
                }
                #pragma omp critical
                normalize_dense_chunk(in_buf.data(), n_rows, cc,
                                       size_factors.data() + c, &writer);
            }
        }
        if (oom_occurred) {
            REprintf("[scLean] NormalizeData CLR: OOM, consider smaller max_dense_chunk_mb\n");
            throw std::bad_alloc();
        }
    } else {
        // LogNormalize/RC: preserve sparsity, read and write sparse
        std::string data_path = input_group + "/data";
        std::string idx_path = input_group + "/indices";
        std::string indptr_path = input_group + "/indptr";

        #pragma omp parallel if(n_threads > 1) num_threads(n_threads)
        {
            hid_t t_fid = (n_threads > 1) ? file->open_thread_handle(FileMode::ReadOnly) : file->file_id();
            #pragma omp for schedule(dynamic)
            for (int64 c = 0; c < n_cells; c += cfg.chunk_size) {
                int64 cc = std::min(cfg.chunk_size, n_cells - c);

                // Read indptr
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

                    #pragma omp critical
                    normalize_sparse_chunk(vals.data(), idxs.data(), ip.data(),
                                            cc, size_factors.data() + c, &writer);
                } else {
                    // All-zero columns — write empty columns
                    #pragma omp critical
                    {
                        for (int64 j = 0; j < cc; ++j) {
                            writer.write_column(nullptr, nullptr, 0);
                        }
                    }
                }
            }
        }
    }

    writer.finalize();

    progress.done();

    NormalizeResult result;
    result.size_factors = size_factors;
    result.n_cells = n_cells;
    result.n_genes = n_rows;
    return result;
}

} // namespace sclean
