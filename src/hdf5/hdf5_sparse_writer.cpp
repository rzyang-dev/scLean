#include "hdf5_sparse_writer.h"
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <algorithm>

namespace sclean {

static constexpr int64 SPARSE_CHUNK_NNZ = 1ULL << 20;  // 1M elements per chunk

static void check_write(herr_t status, const char* path) {
    if (status < 0) {
        char buf[512];
        snprintf(buf, sizeof(buf), "HDF5 write failed: %s", path);
        throw std::runtime_error(buf);
    }
}

HDF5SparseWriter::HDF5SparseWriter(HDF5File* file, const std::string& group_path,
                                     int64 n_rows, int64 n_cols, int64 total_nnz,
                                     int64 batch_nnz, int compression)
    : file_(file), group_path_(group_path),
      n_rows_(n_rows), n_cols_(n_cols),
      batch_nnz_(batch_nnz), compression_(compression),
      dset_data_(-1), dset_indices_(-1), dset_indptr_(-1),
      total_written_(0), cols_written_(0) {

    // Reserve buffer capacity
    int64 reserve = std::min(batch_nnz_, total_nnz);
    buf_data_.reserve(reserve);
    buf_indices_.reserve(reserve);
    buf_indptr_.reserve(n_cols_ / 10 + 1);  // rough estimate

    // Start indptr
    buf_indptr_.push_back(0);

    ensure_group_exists();
    create_datasets();
}

void HDF5SparseWriter::ensure_group_exists() {
    // Create all intermediate groups
    std::string g;
    size_t pos = 0;
    hid_t fid = file_->file_id();
    while ((pos = group_path_.find('/', pos + 1)) != std::string::npos) {
        g = group_path_.substr(0, pos);
        if (!g.empty() && H5Lexists(fid, g.c_str(), H5P_DEFAULT) <= 0) {
            hid_t grp = H5Gcreate2(fid, g.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Gclose(grp);
        }
    }
    if (H5Lexists(fid, group_path_.c_str(), H5P_DEFAULT) <= 0) {
        hid_t grp = H5Gcreate2(fid, group_path_.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Gclose(grp);
    }
}

void HDF5SparseWriter::create_datasets() {
    hid_t fid = file_->file_id();

    hsize_t chunk_1d[1] = {static_cast<hsize_t>(SPARSE_CHUNK_NNZ)};

    auto create_1d_ext = [&](const std::string& name, hid_t dtype,
                              hsize_t max_len, int comp) -> hid_t {
        std::string path = group_path_ + "/" + name;
        if (H5Lexists(fid, path.c_str(), H5P_DEFAULT) > 0) {
            H5Ldelete(fid, path.c_str(), H5P_DEFAULT);
        }
        hsize_t dims[1] = {0};  // start empty
        hsize_t maxdims[1] = {max_len};
        hid_t space = H5Screate_simple(1, dims, maxdims);

        // Extensible datasets require chunked storage
        hsize_t chunk_dims[1] = {std::min(max_len, static_cast<hsize_t>(SPARSE_CHUNK_NNZ))};
        hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(plist, 1, chunk_dims);
        if (comp > 0 && max_len > chunk_dims[0]) {
            H5Pset_deflate(plist, comp);
        }

        hid_t dset = H5Dcreate2(fid, path.c_str(), dtype, space,
                                 H5P_DEFAULT, plist, H5P_DEFAULT);
        H5Pclose(plist);
        H5Sclose(space);
        return dset;
    };

    // For indptr: chunked but no compression (small dataset)
    hsize_t indptr_max = static_cast<hsize_t>(n_cols_ + 1);
    std::string ip = group_path_ + "/indptr";
    if (H5Lexists(fid, ip.c_str(), H5P_DEFAULT) > 0) H5Ldelete(fid, ip.c_str(), H5P_DEFAULT);
    hsize_t ip0[1] = {0};
    hsize_t ipmax[1] = {indptr_max};
    hid_t ipspace = H5Screate_simple(1, ip0, ipmax);
    hsize_t ip_chunk[1] = {std::min(indptr_max, static_cast<hsize_t>(SPARSE_CHUNK_NNZ))};
    hid_t ip_plist = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(ip_plist, 1, ip_chunk);
    dset_indptr_ = H5Dcreate2(fid, ip.c_str(), H5T_NATIVE_INT64,
                                ipspace, H5P_DEFAULT, ip_plist, H5P_DEFAULT);
    H5Pclose(ip_plist);
    H5Sclose(ipspace);

    // data and indices start at 0, extend without limit (centering/scaling
    // can make a sparse matrix dense, blowing past any initial nnz estimate)
    hsize_t max_nnz = static_cast<hsize_t>(n_rows_) * static_cast<hsize_t>(n_cols_);
    if (max_nnz == 0) max_nnz = H5S_UNLIMITED;
    dset_data_    = create_1d_ext("data",    H5T_NATIVE_DOUBLE, max_nnz, compression_);
    dset_indices_ = create_1d_ext("indices", H5T_NATIVE_INT32,  max_nnz, compression_);

    // Write shape attribute
    int64 shape[2] = {n_rows_, n_cols_};
    hsize_t sdim[1] = {2};
    hid_t sspace = H5Screate_simple(1, sdim, nullptr);
    std::string spath = group_path_ + "/shape";
    if (H5Lexists(fid, spath.c_str(), H5P_DEFAULT) > 0) H5Ldelete(fid, spath.c_str(), H5P_DEFAULT);
    hid_t sdset = H5Dcreate2(fid, spath.c_str(), H5T_NATIVE_INT64,
                              sspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    check_write(H5Dwrite(sdset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, shape), spath.c_str());
    H5Dclose(sdset);
    H5Sclose(sspace);
}

void HDF5SparseWriter::flush_buffer() {
    if (buf_data_.empty()) return;

    // Extend datasets
    hsize_t new_size = static_cast<hsize_t>(total_written_ + static_cast<int64>(buf_data_.size()));

    // Extend data
    if (H5Dset_extent(dset_data_, &new_size) < 0)
        throw std::runtime_error("Failed to extend data dataset to " + std::to_string(new_size));
    hid_t filespace_d = H5Dget_space(dset_data_);
    hsize_t start_d[1] = {static_cast<hsize_t>(total_written_)};
    hsize_t count_d[1] = {static_cast<hsize_t>(buf_data_.size())};
    H5Sselect_hyperslab(filespace_d, H5S_SELECT_SET, start_d, nullptr, count_d, nullptr);
    hid_t memspace_d = H5Screate_simple(1, count_d, nullptr);
    check_write(H5Dwrite(dset_data_, H5T_NATIVE_DOUBLE, memspace_d, filespace_d, H5P_DEFAULT,
                         buf_data_.data()), (group_path_ + "/data").c_str());
    H5Sclose(memspace_d);
    H5Sclose(filespace_d);

    // Extend indices
    if (H5Dset_extent(dset_indices_, &new_size) < 0)
        throw std::runtime_error("Failed to extend indices dataset to " + std::to_string(new_size));
    hid_t filespace_i = H5Dget_space(dset_indices_);
    H5Sselect_hyperslab(filespace_i, H5S_SELECT_SET, start_d, nullptr, count_d, nullptr);
    hid_t memspace_i = H5Screate_simple(1, count_d, nullptr);
    check_write(H5Dwrite(dset_indices_, H5T_NATIVE_INT32, memspace_i, filespace_i, H5P_DEFAULT,
                         buf_indices_.data()), (group_path_ + "/indices").c_str());
    H5Sclose(memspace_i);
    H5Sclose(filespace_i);

    // Extend indptr — write all entries including leading 0 offset by base
    hsize_t indptr_new = static_cast<hsize_t>(cols_written_ + buf_indptr_.size());
    if (H5Dset_extent(dset_indptr_, &indptr_new) < 0)
        throw std::runtime_error("Failed to extend indptr dataset to " + std::to_string(indptr_new));
    hid_t filespace_p = H5Dget_space(dset_indptr_);
    hsize_t start_p[1] = {static_cast<hsize_t>(cols_written_)};
    hsize_t count_p[1] = {static_cast<hsize_t>(buf_indptr_.size())};
    H5Sselect_hyperslab(filespace_p, H5S_SELECT_SET, start_p, nullptr, count_p, nullptr);

    std::vector<int64> global_indptr;
    global_indptr.reserve(buf_indptr_.size());
    int64 base = total_written_;
    for (size_t i = 0; i < buf_indptr_.size(); ++i) {
        global_indptr.push_back(buf_indptr_[i] + base);
    }

    hid_t memspace_p = H5Screate_simple(1, count_p, nullptr);
    check_write(H5Dwrite(dset_indptr_, H5T_NATIVE_INT64, memspace_p, filespace_p, H5P_DEFAULT,
                         global_indptr.data()), (group_path_ + "/indptr").c_str());
    H5Sclose(memspace_p);
    H5Sclose(filespace_p);

    // Update totals
    total_written_ += static_cast<int64>(buf_data_.size());
    cols_written_ += static_cast<int64>(buf_indptr_.size()) - 1;

    // Clear buffers
    buf_data_.clear();
    buf_indices_.clear();
    buf_indptr_.clear();
    buf_indptr_.push_back(0);
}

HDF5SparseWriter::~HDF5SparseWriter() {
    // Note: finalize() should be called before destruction.
    // This is a safety net — flush any remaining data.
    try {
        if (!buf_data_.empty()) {
            flush_buffer();
        }
    } catch (...) {
        // Don't throw in destructor
    }
    if (dset_data_ >= 0)    H5Dclose(dset_data_);
    if (dset_indices_ >= 0) H5Dclose(dset_indices_);
    if (dset_indptr_ >= 0)  H5Dclose(dset_indptr_);
}

void HDF5SparseWriter::write_column(const double* values,
                                     const int32* row_indices,
                                     int64 nnz_in_col) {
    for (int64 k = 0; k < nnz_in_col; ++k) {
        buf_data_.push_back(values[k]);
        buf_indices_.push_back(row_indices[k]);
    }
    buf_indptr_.push_back(buf_indptr_.back() + nnz_in_col);

    if (static_cast<int64>(buf_data_.size()) >= batch_nnz_) {
        flush_buffer();
    }
}

void HDF5SparseWriter::write_dense_columns_sparse(const double* buffer,
                                                    int64 n_rows, int64 n_cols) {
    for (int64 j = 0; j < n_cols; ++j) {
        // Count non-zeros in this column first
        int64 col_nnz = 0;
        for (int64 i = 0; i < n_rows; ++i) {
            if (buffer[i * n_cols + j] != 0.0) col_nnz++;
        }

        // Write sparse column
        buf_indptr_.push_back(buf_indptr_.back() + col_nnz);
        for (int64 i = 0; i < n_rows; ++i) {
            double v = buffer[i * n_cols + j];
            if (v != 0.0) {
                buf_data_.push_back(v);
                buf_indices_.push_back(static_cast<int32>(i));
            }
        }

        if (static_cast<int64>(buf_data_.size()) >= batch_nnz_) {
            flush_buffer();
        }
    }
}

void HDF5SparseWriter::finalize() {
    if (!buf_data_.empty()) {
        flush_buffer();
    }

    // Write the final indptr entry (total nnz) if not already written
    hsize_t expected_indptr = static_cast<hsize_t>(n_cols_ + 1);
    hid_t space = H5Dget_space(dset_indptr_);
    hsize_t current_size;
    H5Sget_simple_extent_dims(space, &current_size, nullptr);
    H5Sclose(space);

    if (current_size < n_cols_ + 1) {
        // Extend to final size and write last entry
        if (H5Dset_extent(dset_indptr_, &expected_indptr) < 0)
            throw std::runtime_error("Failed to extend indptr dataset to final size");
        int64 final_nnz = total_written_;
        hsize_t last_pos[1] = {static_cast<hsize_t>(n_cols_)};
        hid_t ms = H5Screate_simple(1, (hsize_t[]){1}, nullptr);
        hid_t fs = H5Dget_space(dset_indptr_);
        H5Sselect_hyperslab(fs, H5S_SELECT_SET, last_pos, nullptr, (hsize_t[]){1}, nullptr);
        check_write(H5Dwrite(dset_indptr_, H5T_NATIVE_INT64, ms, fs, H5P_DEFAULT, &final_nnz),
                    (group_path_ + "/indptr").c_str());
        H5Sclose(fs); H5Sclose(ms);
    }
}

} // namespace sclean
