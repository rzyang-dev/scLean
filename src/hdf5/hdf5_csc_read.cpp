#include "hdf5_csc_matrix.h"
#include "hdf5_utils.h"
#include <algorithm>
#include <vector>
#include <utility>

namespace sclean {

// --- Column chunk read ---

void HDF5CSCMatrix::read_cols(double* buffer,
                               int64 row_start, int64 row_count,
                               int64 col_start, int64 col_count) {
    hid_t fid = thread_file_ >= 0 ? thread_file_ : file_->file_id();

    std::vector<int64> indptr_chunk(col_count + 1);
    {
        hsize_t start[1] = {static_cast<hsize_t>(col_start)};
        hsize_t count[1] = {static_cast<hsize_t>(col_count + 1)};
        hid_t memspace = H5Screate_simple(1, count, nullptr);
        hid_t filespace = H5Dget_space(dset_indptr_);
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, count, nullptr);
        H5Dread(dset_indptr_, H5T_NATIVE_INT64, memspace, filespace, H5P_DEFAULT,
                indptr_chunk.data());
        H5Sclose(filespace);
        H5Sclose(memspace);
    }

    int64 start_nnz = indptr_chunk[0];
    int64 end_nnz = indptr_chunk[col_count];
    int64 chunk_nnz = end_nnz - start_nnz;

    // Zero the output buffer
    int64 total_cells = row_count * col_count;
    std::fill(buffer, buffer + total_cells, 0.0);

    if (chunk_nnz == 0) return;

    std::vector<double> data_chunk(chunk_nnz);
    std::vector<int32> idx_chunk(chunk_nnz);

    {
        hsize_t ds[1] = {static_cast<hsize_t>(start_nnz)};
        hsize_t dc[1] = {static_cast<hsize_t>(chunk_nnz)};
        hid_t ms = H5Screate_simple(1, dc, nullptr);
        hid_t fs = H5Dget_space(dset_data_);
        H5Sselect_hyperslab(fs, H5S_SELECT_SET, ds, nullptr, dc, nullptr);
        H5Dread(dset_data_, H5T_NATIVE_DOUBLE, ms, fs, H5P_DEFAULT, data_chunk.data());
        H5Sclose(fs); H5Sclose(ms);
    }
    {
        hsize_t ds[1] = {static_cast<hsize_t>(start_nnz)};
        hsize_t dc[1] = {static_cast<hsize_t>(chunk_nnz)};
        hid_t ms = H5Screate_simple(1, dc, nullptr);
        hid_t fs = H5Dget_space(dset_indices_);
        H5Sselect_hyperslab(fs, H5S_SELECT_SET, ds, nullptr, dc, nullptr);
        H5Dread(dset_indices_, H5T_NATIVE_INT32, ms, fs, H5P_DEFAULT, idx_chunk.data());
        H5Sclose(fs); H5Sclose(ms);
    }

    // Scatter into dense buffer
    for (int64 i = 0; i < chunk_nnz; ++i) {
        int64 row = idx_chunk[i];
        if (row < row_start || row >= row_start + row_count) continue;
        // Find which column this entry belongs to
        // Binary search in indptr_chunk
        int64 local_col = 0;
        {
            int64 lo = 0, hi = col_count;
            int64 target_nnz = start_nnz + i;
            while (lo < hi) {
                int64 mid = lo + (hi - lo) / 2;
                if (indptr_chunk[mid] <= target_nnz) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            local_col = lo - 1;
            if (local_col < 0) local_col = 0;
        }
        int64 buf_idx = (row - row_start) * col_count + local_col;
        buffer[buf_idx] = data_chunk[i];
    }
}

// --- Row chunk read (using row_ptr) ---

void HDF5CSCMatrix::read_rows(double* buffer,
                               int64 row_start, int64 row_count,
                               int64 col_start, int64 col_count) {
    // For CSC storage, the natural read direction is column-wise.
    // Scan all entries in the requested column range and scatter
    // into the dense output buffer (row-major layout).
    int64 total_cells = row_count * col_count;
    std::fill(buffer, buffer + total_cells, 0.0);

    hid_t fid = thread_file_ >= 0 ? thread_file_ : file_->file_id();

    // Read indptr slice for the column range
    std::vector<int64> indptr_chunk(col_count + 1);
    {
        hsize_t start[1] = {static_cast<hsize_t>(col_start)};
        hsize_t count[1] = {static_cast<hsize_t>(col_count + 1)};
        hid_t memspace = H5Screate_simple(1, count, nullptr);
        hid_t filespace = H5Dget_space(dset_indptr_);
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, count, nullptr);
        H5Dread(dset_indptr_, H5T_NATIVE_INT64, memspace, filespace, H5P_DEFAULT,
                indptr_chunk.data());
        H5Sclose(filespace);
        H5Sclose(memspace);
    }

    int64 start_nnz = indptr_chunk[0];
    int64 end_nnz = indptr_chunk[col_count];
    int64 chunk_nnz = end_nnz - start_nnz;

    if (chunk_nnz == 0) return;

    std::vector<double> data_chunk(chunk_nnz);
    std::vector<int32> idx_chunk(chunk_nnz);

    // Read data
    {
        hsize_t ds[1] = {static_cast<hsize_t>(start_nnz)};
        hsize_t dc[1] = {static_cast<hsize_t>(chunk_nnz)};
        hid_t ms = H5Screate_simple(1, dc, nullptr);
        hid_t fs = H5Dget_space(dset_data_);
        H5Sselect_hyperslab(fs, H5S_SELECT_SET, ds, nullptr, dc, nullptr);
        H5Dread(dset_data_, H5T_NATIVE_DOUBLE, ms, fs, H5P_DEFAULT, data_chunk.data());
        H5Sclose(fs); H5Sclose(ms);
    }

    // Read indices
    {
        hsize_t ds[1] = {static_cast<hsize_t>(start_nnz)};
        hsize_t dc[1] = {static_cast<hsize_t>(chunk_nnz)};
        hid_t ms = H5Screate_simple(1, dc, nullptr);
        hid_t fs = H5Dget_space(dset_indices_);
        H5Sselect_hyperslab(fs, H5S_SELECT_SET, ds, nullptr, dc, nullptr);
        H5Dread(dset_indices_, H5T_NATIVE_INT32, ms, fs, H5P_DEFAULT, idx_chunk.data());
        H5Sclose(fs); H5Sclose(ms);
    }

    // Scatter into dense buffer (row-major)
    for (int64 i = 0; i < chunk_nnz; ++i) {
        int64 row = idx_chunk[i];
        if (row < row_start || row >= row_start + row_count) continue;

        // Binary search in indptr_chunk to find which column this entry belongs to
        int64 local_col = 0;
        {
            int64 lo = 0, hi = col_count;
            int64 target_nnz = start_nnz + i;
            while (lo < hi) {
                int64 mid = lo + (hi - lo) / 2;
                if (indptr_chunk[mid] <= target_nnz) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            local_col = lo - 1;
            if (local_col < 0) local_col = 0;
        }
        buffer[(row - row_start) * col_count + local_col] = data_chunk[i];
    }
}

// --- Sparse-specific row access ---

void HDF5CSCMatrix::read_sparse_rows(
    std::vector<double>& values,
    std::vector<int32>& indices,
    std::vector<int64>& row_starts,
    int64 row_start, int64 row_count) {

    // Collect entries for the requested rows by scanning columns in CSC order.
    // Each entry is (value, col_index); grouped by local row.
    int64 row_end = row_start + row_count;
    std::vector<std::vector<std::pair<int32, double>>> row_entries(row_count);

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

        // Scatter entries into per-row buffers
        for (int64 j = 0; j < col_count; ++j) {
            int32 col = static_cast<int32>(col_start + j);
            int64 nnz_start = indptr_slice[j] - indptr_slice[0];
            int64 nnz_end = indptr_slice[j + 1] - indptr_slice[0];
            for (int64 k = nnz_start; k < nnz_end; ++k) {
                int64 row = rows[k];
                if (row >= row_start && row < row_end) {
                    row_entries[row - row_start].emplace_back(col, vals[k]);
                }
            }
        }
    }

    // Flatten into output format
    row_starts.resize(row_count + 1);
    row_starts[0] = 0;
    for (int64 i = 0; i < row_count; ++i) {
        row_starts[i + 1] = row_starts[i] + static_cast<int64>(row_entries[i].size());
    }

    int64 total_nnz = row_starts[row_count];
    values.resize(total_nnz);
    indices.resize(total_nnz);

    for (int64 i = 0; i < row_count; ++i) {
        int64 offset = row_starts[i];
        for (size_t j = 0; j < row_entries[i].size(); ++j) {
            values[offset + j] = row_entries[i][j].second;
            indices[offset + j] = row_entries[i][j].first;
        }
    }
}

} // namespace sclean
