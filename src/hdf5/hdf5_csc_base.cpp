#include "hdf5_csc_matrix.h"
#include "hdf5_utils.h"
#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace sclean {

// --- full_path ---

std::string HDF5CSCMatrix::full_path(const std::string& name) const {
    return group_path_ + "/" + name;
}

// --- build_row_ptr (static helper) ---

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
//
// Constructor flow:
// 1. Open datasets (data, indices, indptr) from the HDF5 group.
// 2. Read dimensions from the "shape" attribute or dataset extents.
// 3. Optionally build the in-memory row_ptr cache (CSR row pointers) from
//    indices + indptr. This cache enables O(1) row-slice queries at the cost
//    of ~ (n_rows+1) * 8 bytes. It is built lazily (via ensure_row_ptr) and
//    never written back to HDF5, preventing corruption during read operations.
// 4. If thread_file is set (not -1), use a per-thread HDF5 file handle for
//    parallel reads with OpenMP. Thread-local handles prevent HDF5 library
//    contention when HDF5 was NOT built with --enable-threadsafe.

HDF5CSCMatrix::HDF5CSCMatrix(HDF5File* file, const std::string& group_path,
                               hid_t thread_file)
    : file_(file), group_path_(group_path),
      n_rows_(0), n_cols_(0), nnz_(0),
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
    hsize_t indptr_len = hdf5::dset_len(d_indptr);
    n_cols_ = static_cast<int64>(indptr_len) - 1;
    H5Dclose(d_indptr);

    std::string data_path = full_path("data");
    hid_t d_data = H5Dopen2(fid, data_path.c_str(), H5P_DEFAULT);
    nnz_ = static_cast<int64>(hdf5::dset_len(d_data));
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
                hsize_t rp_len = hdf5::dset_len(drp);
                n_rows_ = static_cast<int64>(rp_len) - 1;
                H5Dclose(drp);
            }
        }
    }

    open_datasets(fid);
}

// --- Constructor: create new ---

HDF5CSCMatrix::HDF5CSCMatrix(HDF5File* file, const std::string& group_path,
                               const std::vector<double>& data,
                               const std::vector<int32>& indices,
                               const std::vector<int64>& indptr,
                               int64 n_rows, int64 n_cols, int compression)
    : file_(file), group_path_(group_path),
      n_rows_(n_rows), n_cols_(n_cols), nnz_(static_cast<int64>(data.size())),
      dset_data_(-1), dset_indices_(-1), dset_indptr_(-1), dset_row_ptr_(-1),
      thread_file_(-1) {

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
    dset_data_ = hdf5::create_1d(fid, full_path("data"), H5T_NATIVE_DOUBLE, data.size(), compression);
    hdf5::check_write(H5Dwrite(dset_data_, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()),
                full_path("data"));

    dset_indices_ = hdf5::create_1d(fid, full_path("indices"), H5T_NATIVE_INT32, indices.size(), compression);
    hdf5::check_write(H5Dwrite(dset_indices_, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, indices.data()),
                full_path("indices"));

    dset_indptr_ = hdf5::create_1d(fid, full_path("indptr"), H5T_NATIVE_INT64, indptr.size(), 0);
    hdf5::check_write(H5Dwrite(dset_indptr_, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, indptr.data()),
                full_path("indptr"));

    // Build and write row_ptr (CSR pointer)
    std::vector<int64> row_ptr(n_rows + 1);
    build_row_ptr(n_rows, n_cols, indices.data(), indptr.data(), row_ptr.data());
    dset_row_ptr_ = hdf5::create_1d(fid, full_path("row_ptr"), H5T_NATIVE_INT64, row_ptr.size(), 0);
    hdf5::check_write(H5Dwrite(dset_row_ptr_, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, row_ptr.data()),
                full_path("row_ptr"));

    // Write shape attribute
    int64 shape[2] = {n_rows, n_cols};
    hsize_t shape_dims[1] = {2};
    hid_t space = H5Screate_simple(1, shape_dims, nullptr);
    hid_t d_shape = H5Dcreate2(fid, full_path("shape").c_str(), H5T_NATIVE_INT64,
                                space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hdf5::check_write(H5Dwrite(d_shape, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, shape),
                full_path("shape"));
    H5Dclose(d_shape);
    H5Sclose(space);
}

// --- Destructor ---

HDF5CSCMatrix::~HDF5CSCMatrix() {
    close_datasets();
}

// --- open_datasets / close_datasets ---

void HDF5CSCMatrix::open_datasets(hid_t file_id) {
    dset_data_ = hdf5::open_dset(file_id, full_path("data"));
    dset_indices_ = hdf5::open_dset(file_id, full_path("indices"));
    dset_indptr_ = hdf5::open_dset(file_id, full_path("indptr"));

    std::string rp_path = full_path("row_ptr");
    if (H5Lexists(file_id, rp_path.c_str(), H5P_DEFAULT) > 0) {
        dset_row_ptr_ = hdf5::open_dset(file_id, rp_path);
    }
}

void HDF5CSCMatrix::close_datasets() {
    if (dset_data_ >= 0)    { H5Dclose(dset_data_);    dset_data_ = -1; }
    if (dset_indices_ >= 0) { H5Dclose(dset_indices_); dset_indices_ = -1; }
    if (dset_indptr_ >= 0)  { H5Dclose(dset_indptr_);  dset_indptr_ = -1; }
    if (dset_row_ptr_ >= 0) { H5Dclose(dset_row_ptr_); dset_row_ptr_ = -1; }
    row_ptr_cache_.clear();
}

// --- ensure_row_ptr ---

void HDF5CSCMatrix::ensure_row_ptr() {
    // Already have in-memory cache or HDF5 dataset open
    if (!row_ptr_cache_.empty() || dset_row_ptr_ >= 0) return;

    hid_t fid = thread_file_ >= 0 ? thread_file_ : file_->file_id();
    std::string rp_path = full_path("row_ptr");

    if (H5Lexists(fid, rp_path.c_str(), H5P_DEFAULT) > 0) {
        dset_row_ptr_ = hdf5::open_dset(fid, rp_path);
        return;
    }

    // Build row_ptr and cache in memory -- do NOT write to HDF5,
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

} // namespace sclean
