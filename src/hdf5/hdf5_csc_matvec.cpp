#include "hdf5_csc_matrix.h"
#include "hdf5_utils.h"
#include <algorithm>
#include <vector>

namespace sclean {

// --- matvec: y = A * x ---

void HDF5CSCMatrix::matvec(const double* x, double* y) {
    // y = A * x  where A is (n_rows x n_cols), x is (n_cols,)
    // CSC: for each column j, for each entry (i,j), y[i] += A[i,j] * x[j]
    std::fill(y, y + n_rows_, 0.0);

    hid_t fid = thread_file_ >= 0 ? thread_file_ : file_->file_id();

    // Adaptive chunk size: limit per-chunk nnz to ~5M entries (~60 MB)
    // to avoid OOM on dense-ish matrices (e.g. scaled data)
    constexpr int64 kMaxNnzPerChunk = 5 * 1024 * 1024;
    double avg_nnz_per_col = (n_cols_ > 0)
        ? static_cast<double>(nnz_) / n_cols_ : 0.0;
    int64 chunk_size = (avg_nnz_per_col > 0)
        ? std::max(static_cast<int64>(1),
                   std::min(static_cast<int64>(4096),
                            kMaxNnzPerChunk / std::max(static_cast<int64>(1),
                                static_cast<int64>(avg_nnz_per_col))))
        : 4096;
    for (int64 col_start = 0; col_start < n_cols_; col_start += chunk_size) {
        int64 col_count = std::min(chunk_size, n_cols_ - col_start);
        int64 col_end = col_start + col_count;

        // Read indptr slice
        std::vector<int64> indptr_slice(col_count + 1);
        {
            hsize_t start[1] = {static_cast<hsize_t>(col_start)};
            hsize_t count[1] = {static_cast<hsize_t>(col_count + 1)};
            hid_t ms = H5Screate_simple(1, count, nullptr);
            hid_t fs = H5Dget_space(dset_indptr_);
            H5Sselect_hyperslab(fs, H5S_SELECT_SET, start, nullptr, count, nullptr);
            H5Dread(dset_indptr_, H5T_NATIVE_INT64, ms, fs, H5P_DEFAULT,
                    indptr_slice.data());
            H5Sclose(fs); H5Sclose(ms);
        }

        int64 slice_nnz = indptr_slice[col_count] - indptr_slice[0];
        if (slice_nnz == 0) continue;

        std::vector<double> vals(slice_nnz);
        std::vector<int32> rows(slice_nnz);
        {
            hsize_t start[1] = {static_cast<hsize_t>(indptr_slice[0])};
            hsize_t count[1] = {static_cast<hsize_t>(slice_nnz)};
            hid_t ms = H5Screate_simple(1, count, nullptr);

            hid_t fs_d = H5Dget_space(dset_data_);
            H5Sselect_hyperslab(fs_d, H5S_SELECT_SET, start, nullptr, count, nullptr);
            H5Dread(dset_data_, H5T_NATIVE_DOUBLE, ms, fs_d, H5P_DEFAULT, vals.data());
            H5Sclose(fs_d);

            hid_t fs_i = H5Dget_space(dset_indices_);
            H5Sselect_hyperslab(fs_i, H5S_SELECT_SET, start, nullptr, count, nullptr);
            H5Dread(dset_indices_, H5T_NATIVE_INT32, ms, fs_i, H5P_DEFAULT, rows.data());
            H5Sclose(fs_i);

            H5Sclose(ms);
        }

        // For each column, multiply
        for (int64 j = 0; j < col_count; ++j) {
            int64 col = col_start + j;
            double xj = x[col];
            if (xj == 0.0) continue;
            int64 nnz_start = indptr_slice[j] - indptr_slice[0];
            int64 nnz_end = indptr_slice[j + 1] - indptr_slice[0];
            for (int64 k = nnz_start; k < nnz_end; ++k) {
                y[rows[k]] += vals[k] * xj;
            }
        }
    }
}

// --- rmatvec: y = A^T * x ---

void HDF5CSCMatrix::rmatvec(const double* x, double* y) {
    // y = A^T * x  where x is (n_rows,)
    // CSC: for each column j, for each entry (i,j), y[j] += A[i,j] * x[i]
    // Data is stored in column-major order, so iterate by columns.
    hid_t fid = thread_file_ >= 0 ? thread_file_ : file_->file_id();

    constexpr int64 kMaxNnzPerChunk = 5 * 1024 * 1024;
    double avg_nnz_per_col = (n_cols_ > 0)
        ? static_cast<double>(nnz_) / n_cols_ : 0.0;
    int64 chunk_size = (avg_nnz_per_col > 0)
        ? std::max(static_cast<int64>(1),
                   std::min(static_cast<int64>(4096),
                            kMaxNnzPerChunk / std::max(static_cast<int64>(1),
                                static_cast<int64>(avg_nnz_per_col))))
        : 4096;

    for (int64 col_start = 0; col_start < n_cols_; col_start += chunk_size) {
        int64 col_count = std::min(chunk_size, n_cols_ - col_start);

        // Read indptr slice
        std::vector<int64> indptr_slice(col_count + 1);
        {
            hsize_t start[1] = {static_cast<hsize_t>(col_start)};
            hsize_t count[1] = {static_cast<hsize_t>(col_count + 1)};
            hid_t ms = H5Screate_simple(1, count, nullptr);
            hid_t fs = H5Dget_space(dset_indptr_);
            H5Sselect_hyperslab(fs, H5S_SELECT_SET, start, nullptr, count, nullptr);
            H5Dread(dset_indptr_, H5T_NATIVE_INT64, ms, fs, H5P_DEFAULT,
                    indptr_slice.data());
            H5Sclose(fs); H5Sclose(ms);
        }

        int64 slice_nnz = indptr_slice[col_count] - indptr_slice[0];
        if (slice_nnz == 0) {
            // Zero-fill these columns
            std::fill(y + col_start, y + col_start + col_count, 0.0);
            continue;
        }

        std::vector<double> vals(slice_nnz);
        std::vector<int32> rows(slice_nnz);
        {
            hsize_t start[1] = {static_cast<hsize_t>(indptr_slice[0])};
            hsize_t count[1] = {static_cast<hsize_t>(slice_nnz)};
            hid_t ms = H5Screate_simple(1, count, nullptr);

            hid_t fs_d = H5Dget_space(dset_data_);
            H5Sselect_hyperslab(fs_d, H5S_SELECT_SET, start, nullptr, count, nullptr);
            H5Dread(dset_data_, H5T_NATIVE_DOUBLE, ms, fs_d, H5P_DEFAULT, vals.data());
            H5Sclose(fs_d);

            hid_t fs_i = H5Dget_space(dset_indices_);
            H5Sselect_hyperslab(fs_i, H5S_SELECT_SET, start, nullptr, count, nullptr);
            H5Dread(dset_indices_, H5T_NATIVE_INT32, ms, fs_i, H5P_DEFAULT, rows.data());
            H5Sclose(fs_i);

            H5Sclose(ms);
        }

        // For each column, compute dot product: y[col] = sum_i A[i,col] * x[i]
        for (int64 j = 0; j < col_count; ++j) {
            int64 col = col_start + j;
            int64 nnz_start = indptr_slice[j] - indptr_slice[0];
            int64 nnz_end = indptr_slice[j + 1] - indptr_slice[0];
            double yj = 0.0;
            for (int64 k = nnz_start; k < nnz_end; ++k) {
                yj += vals[k] * x[rows[k]];
            }
            y[col] = yj;
        }
    }
}

} // namespace sclean
