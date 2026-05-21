#include "hdf5_csc_matrix.h"
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <hdf5_hl.h>

namespace sclean {

// --- Utilities ---

std::string HDF5CSCMatrix::full_path(const std::string& name) const {
    return group_path_ + "/" + name;
}

static void check_write(herr_t status, const std::string& path) {
    if (status < 0) {
        char buf[512];
        snprintf(buf, sizeof(buf), "HDF5 write failed: %s", path.c_str());
        throw std::runtime_error(buf);
    }
}

static hid_t open_dset(hid_t file_id, const std::string& path) {
    hid_t d = H5Dopen2(file_id, path.c_str(), H5P_DEFAULT);
    if (d < 0) throw std::runtime_error("Cannot open HDF5 dataset: " + path);
    return d;
}

static hsize_t dset_len(hid_t dset) {
    hid_t space = H5Dget_space(dset);
    hsize_t len;
    H5Sget_simple_extent_dims(space, &len, nullptr);
    H5Sclose(space);
    return len;
}

static hid_t create_1d(hid_t file_id, const std::string& path, hid_t dtype,
                        hsize_t len, int compression) {
    // Create intermediate groups
    std::string group_path;
    size_t slash = 0;
    while ((slash = path.find('/', slash + 1)) != std::string::npos) {
        std::string g = path.substr(0, slash);
        if (H5Lexists(file_id, g.c_str(), H5P_DEFAULT) <= 0) {
            hid_t grp = H5Gcreate2(file_id, g.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Gclose(grp);
        }
    }

    if (H5Lexists(file_id, path.c_str(), H5P_DEFAULT) > 0) {
        H5Ldelete(file_id, path.c_str(), H5P_DEFAULT);
    }

    hsize_t chunk_dims[1] = {std::min(len, static_cast<hsize_t>(HDF5_CHUNK_SIZE_1D))};
    hid_t plist = H5P_DEFAULT;
    if (compression > 0 && len > chunk_dims[0]) {
        plist = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(plist, 1, chunk_dims);
        H5Pset_deflate(plist, compression);
    }

    hid_t space = H5Screate_simple(1, &len, nullptr);
    hid_t dset = H5Dcreate2(file_id, path.c_str(), dtype, space,
                             H5P_DEFAULT, plist, H5P_DEFAULT);
    H5Sclose(space);
    if (compression > 0 && len > chunk_dims[0]) H5Pclose(plist);
    return dset;
}

// --- Build CSR row_ptr from CSC ---

void HDF5CSCMatrix::build_row_ptr(int64 n_rows, int64 n_cols,
                                   const int32* indices, const int64* indptr,
                                   int64* row_ptr_out) {
    int64 nnz = indptr[n_cols];
    std::fill(row_ptr_out, row_ptr_out + n_rows + 1, 0);

    // Count non-zeros per row
    for (int64 i = 0; i < nnz; ++i) {
        row_ptr_out[indices[i] + 1]++;
    }

    // Cumulative sum
    for (int64 i = 1; i <= n_rows; ++i) {
        row_ptr_out[i] += row_ptr_out[i - 1];
    }
}

// --- Constructor: open existing ---

HDF5CSCMatrix::HDF5CSCMatrix(HDF5File* file, const std::string& group_path,
                               hid_t thread_file)
    : file_(file), group_path_(group_path),
      dset_data_(-1), dset_indices_(-1), dset_indptr_(-1), dset_row_ptr_(-1),
      thread_file_(thread_file) {

    hid_t fid = thread_file >= 0 ? thread_file : file_->file_id();

    // Read shape attribute
    std::string shape_path = group_path + "/shape";
    if (H5Lexists(fid, shape_path.c_str(), H5P_DEFAULT) > 0) {
        hid_t d = H5Dopen2(fid, shape_path.c_str(), H5P_DEFAULT);
        int64 shape[2];
        H5Dread(d, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, shape);
        n_rows_ = shape[0];
        n_cols_ = shape[1];
        H5Dclose(d);
    }

    // Read or infer dimensions
    std::string indptr_path = full_path("indptr");
    hid_t d_indptr = H5Dopen2(fid, indptr_path.c_str(), H5P_DEFAULT);
    hsize_t indptr_len = dset_len(d_indptr);
    n_cols_ = static_cast<int64>(indptr_len) - 1;
    H5Dclose(d_indptr);

    std::string data_path = full_path("data");
    hid_t d_data = H5Dopen2(fid, data_path.c_str(), H5P_DEFAULT);
    nnz_ = static_cast<int64>(dset_len(d_data));
    H5Dclose(d_data);

    // Infer n_rows from max index + 1
    std::string indices_path = full_path("indices");
    if (H5Lexists(fid, indices_path.c_str(), H5P_DEFAULT) > 0) {
        // Read last element's index to estimate; better to have shape attr
        if (n_rows_ == 0) {
            // Try reading the row_ptr to infer
            std::string rp_path = full_path("row_ptr");
            if (H5Lexists(fid, rp_path.c_str(), H5P_DEFAULT) > 0) {
                hid_t drp = H5Dopen2(fid, rp_path.c_str(), H5P_DEFAULT);
                hsize_t rp_len = dset_len(drp);
                n_rows_ = static_cast<int64>(rp_len) - 1;
                H5Dclose(drp);
            }
        }
    }

    open_datasets(fid);
}

// --- Open all datasets ---

void HDF5CSCMatrix::open_datasets(hid_t file_id) {
    dset_data_ = open_dset(file_id, full_path("data"));
    dset_indices_ = open_dset(file_id, full_path("indices"));
    dset_indptr_ = open_dset(file_id, full_path("indptr"));

    std::string rp_path = full_path("row_ptr");
    if (H5Lexists(file_id, rp_path.c_str(), H5P_DEFAULT) > 0) {
        dset_row_ptr_ = open_dset(file_id, rp_path);
    }
}

void HDF5CSCMatrix::close_datasets() {
    if (dset_data_ >= 0)    { H5Dclose(dset_data_);    dset_data_ = -1; }
    if (dset_indices_ >= 0) { H5Dclose(dset_indices_); dset_indices_ = -1; }
    if (dset_indptr_ >= 0)  { H5Dclose(dset_indptr_);  dset_indptr_ = -1; }
    if (dset_row_ptr_ >= 0) { H5Dclose(dset_row_ptr_); dset_row_ptr_ = -1; }
    row_ptr_cache_.clear();
}

// --- Constructor: create new ---

HDF5CSCMatrix::HDF5CSCMatrix(HDF5File* file, const std::string& group_path,
                               const std::vector<double>& data,
                               const std::vector<int32>& indices,
                               const std::vector<int64>& indptr,
                               int64 n_rows, int64 n_cols, int compression)
    : file_(file), group_path_(group_path),
      dset_data_(-1), dset_indices_(-1), dset_indptr_(-1), dset_row_ptr_(-1),
      thread_file_(-1), n_rows_(n_rows), n_cols_(n_cols), nnz_(static_cast<int64>(data.size())) {

    hid_t fid = file_->file_id();

    // Ensure group exists
    if (H5Lexists(fid, group_path.c_str(), H5P_DEFAULT) <= 0) {
        std::string prev;
        size_t pos = 0;
        while ((pos = group_path.find('/', pos + 1)) != std::string::npos) {
            std::string g = group_path.substr(0, pos);
            if (!g.empty() && H5Lexists(fid, g.c_str(), H5P_DEFAULT) <= 0) {
                hid_t grp = H5Gcreate2(fid, g.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                H5Gclose(grp);
            }
        }
        hid_t grp = H5Gcreate2(fid, group_path.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Gclose(grp);
    }

    // Write data, indices, indptr
    dset_data_ = create_1d(fid, full_path("data"), H5T_NATIVE_DOUBLE, data.size(), compression);
    check_write(H5Dwrite(dset_data_, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()),
                full_path("data"));

    dset_indices_ = create_1d(fid, full_path("indices"), H5T_NATIVE_INT32, indices.size(), compression);
    check_write(H5Dwrite(dset_indices_, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, indices.data()),
                full_path("indices"));

    dset_indptr_ = create_1d(fid, full_path("indptr"), H5T_NATIVE_INT64, indptr.size(), 0);
    check_write(H5Dwrite(dset_indptr_, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, indptr.data()),
                full_path("indptr"));

    // Build and write row_ptr (CSR pointer)
    std::vector<int64> row_ptr(n_rows + 1);
    build_row_ptr(n_rows, n_cols, indices.data(), indptr.data(), row_ptr.data());
    dset_row_ptr_ = create_1d(fid, full_path("row_ptr"), H5T_NATIVE_INT64, row_ptr.size(), 0);
    check_write(H5Dwrite(dset_row_ptr_, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, row_ptr.data()),
                full_path("row_ptr"));

    // Write shape attribute
    int64 shape[2] = {n_rows, n_cols};
    hid_t space = H5Screate_simple(1, (hsize_t[]){2}, nullptr);
    hid_t d_shape = H5Dcreate2(fid, full_path("shape").c_str(), H5T_NATIVE_INT64,
                                space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    check_write(H5Dwrite(d_shape, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, shape),
                full_path("shape"));
    H5Dclose(d_shape);
    H5Sclose(space);
}

HDF5CSCMatrix::~HDF5CSCMatrix() {
    close_datasets();
}

// --- ensure_row_ptr ---

void HDF5CSCMatrix::ensure_row_ptr() {
    // Already have in-memory cache or HDF5 dataset open
    if (!row_ptr_cache_.empty() || dset_row_ptr_ >= 0) return;

    hid_t fid = thread_file_ >= 0 ? thread_file_ : file_->file_id();
    std::string rp_path = full_path("row_ptr");

    if (H5Lexists(fid, rp_path.c_str(), H5P_DEFAULT) > 0) {
        dset_row_ptr_ = open_dset(fid, rp_path);
        return;
    }

    // Build row_ptr and cache in memory — do NOT write to HDF5,
    // as writing during read operations corrupts file state.
    row_ptr_cache_.resize(n_rows_ + 1, 0);

    constexpr int64 kRowPtrChunk = 10 * 1024 * 1024;  // 10M indices per chunk
    for (int64 start = 0; start < nnz_; start += kRowPtrChunk) {
        int64 count = std::min(kRowPtrChunk, nnz_ - start);
        std::vector<int32> indices_chunk(count);
        hsize_t ds[1] = {static_cast<hsize_t>(start)};
        hsize_t dc[1] = {static_cast<hsize_t>(count)};
        hid_t ms = H5Screate_simple(1, dc, nullptr);
        hid_t fs = H5Dget_space(dset_indices_);
        H5Sselect_hyperslab(fs, H5S_SELECT_SET, ds, nullptr, dc, nullptr);
        H5Dread(dset_indices_, H5T_NATIVE_INT32, ms, fs, H5P_DEFAULT,
                indices_chunk.data());
        H5Sclose(fs); H5Sclose(ms);
        for (int64 k = 0; k < count; ++k) {
            row_ptr_cache_[indices_chunk[k] + 1]++;
        }
    }

    for (int64 i = 1; i <= n_rows_; ++i) {
        row_ptr_cache_[i] += row_ptr_cache_[i - 1];
    }
}

// --- Column chunk I/O ---

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

void HDF5CSCMatrix::write_cols(const double* buffer,
                                int64 row_start, int64 row_count,
                                int64 col_start, int64 col_count) {
    // For writing, we need to update the sparse representation.
    // Read existing data for these columns, merge with new data,
    // and write back. This is expensive for CSC — use batch rewrite.
    //
    // For now: read existing NNZ structure, then rebuild these columns.
    // In production, prefer writing as dense chunks when feasible.

    std::lock_guard<std::mutex> lock(file_->write_mutex_);

    // Read current indptr range
    std::vector<int64> old_indptr_chunk(col_count + 1);
    {
        hsize_t start[1] = {static_cast<hsize_t>(col_start)};
        hsize_t count[1] = {static_cast<hsize_t>(col_count + 1)};
        hid_t ms = H5Screate_simple(1, count, nullptr);
        hid_t fs = H5Dget_space(dset_indptr_);
        H5Sselect_hyperslab(fs, H5S_SELECT_SET, start, nullptr, count, nullptr);
        H5Dread(dset_indptr_, H5T_NATIVE_INT64, ms, fs, H5P_DEFAULT,
                old_indptr_chunk.data());
        H5Sclose(fs); H5Sclose(ms);
    }

    int64 old_nnz = old_indptr_chunk[col_count] - old_indptr_chunk[0];

    // Count new NNZ
    int64 new_nnz = 0;
    for (int64 r = 0; r < row_count; ++r) {
        for (int64 c = 0; c < col_count; ++c) {
            if (buffer[r * col_count + c] != 0.0) new_nnz++;
        }
    }

    // For simplicity in v0.1: if writing to a layer for the first time,
    // just read all, modify, write all.
    // A proper implementation would do range-based replacement.
    throw std::runtime_error(
        "HDF5CSCMatrix::write_cols: incremental sparse write not yet implemented. "
        "Use write_rows or create a new matrix.");
}

// --- Row chunk I/O (using row_ptr) ---

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

void HDF5CSCMatrix::write_rows(const double* /*buffer*/,
                                int64 /*row_start*/, int64 /*row_count*/,
                                int64 /*col_start*/, int64 /*col_count*/) {
    throw std::runtime_error(
        "HDF5CSCMatrix::write_rows: incremental write not yet implemented.");
}

// --- Matrix-vector products ---

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

// --- Chunk boundary helpers ---

int64 HDF5CSCMatrix::col_nnz(int64 col) const {
    int64 start, end;
    {
        hsize_t s[1] = {static_cast<hsize_t>(col)};
        hsize_t c[1] = {2};
        int64 buf[2];
        hid_t ms = H5Screate_simple(1, c, nullptr);
        hid_t fs = H5Dget_space(dset_indptr_);
        H5Sselect_hyperslab(fs, H5S_SELECT_SET, s, nullptr, c, nullptr);
        H5Dread(dset_indptr_, H5T_NATIVE_INT64, ms, fs, H5P_DEFAULT, buf);
        H5Sclose(fs); H5Sclose(ms);
        start = buf[0]; end = buf[1];
    }
    return end - start;
}

void HDF5CSCMatrix::col_nnz_range(int64 col_start, int64 col_count, int64* out) const {
    std::vector<int64> buf(col_count + 1);
    hsize_t s[1] = {static_cast<hsize_t>(col_start)};
    hsize_t c[1] = {static_cast<hsize_t>(col_count + 1)};
    hid_t ms = H5Screate_simple(1, c, nullptr);
    hid_t fs = H5Dget_space(dset_indptr_);
    H5Sselect_hyperslab(fs, H5S_SELECT_SET, s, nullptr, c, nullptr);
    H5Dread(dset_indptr_, H5T_NATIVE_INT64, ms, fs, H5P_DEFAULT, buf.data());
    H5Sclose(fs); H5Sclose(ms);
    for (int64 i = 0; i < col_count; ++i) {
        out[i] = buf[i + 1] - buf[i];
    }
}

// --- Memory estimation ---

int64 HDF5CSCMatrix::memory_per_row_chunk(int64 n_cols) const {
    double avg_nnz_per_row = (n_rows_ > 0) ? static_cast<double>(nnz_) / n_rows_ : 0.0;
    return static_cast<int64>(avg_nnz_per_row * n_cols * (sizeof(double) + sizeof(int32)));
}

int64 HDF5CSCMatrix::memory_per_col_chunk(int64 n_rows) const {
    double avg_nnz_per_col = (n_cols_ > 0) ? static_cast<double>(nnz_) / n_cols_ : 0.0;
    return static_cast<int64>(avg_nnz_per_col * n_rows * (sizeof(double) + sizeof(int32)));
}

} // namespace sclean
