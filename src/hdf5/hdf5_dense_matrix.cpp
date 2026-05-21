#include "hdf5_dense_matrix.h"
#include <stdexcept>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <hdf5_hl.h>

namespace sclean {

static void check_h5_status(herr_t status, const char* op, const char* path) {
    if (status < 0) {
        char buf[512];
        snprintf(buf, sizeof(buf), "HDF5 %s failed: %s", op, path);
        throw std::runtime_error(buf);
    }
}

// --- Constructor: open existing ---

HDF5DenseMatrix::HDF5DenseMatrix(HDF5File* file, const std::string& dataset_path,
                                   hid_t thread_file)
    : file_(file), dataset_path_(dataset_path), dset_id_(-1),
      n_rows_(0), n_cols_(0), owns_dataset_(false), thread_file_(thread_file) {

    hid_t fid = get_file_id();
    dset_id_ = H5Dopen2(fid, dataset_path.c_str(), H5P_DEFAULT);
    if (dset_id_ < 0) {
        throw std::runtime_error("Cannot open dense dataset: " + dataset_path);
    }

    hid_t space = H5Dget_space(dset_id_);
    int ndims = H5Sget_simple_extent_ndims(space);
    hsize_t dims[2] = {0, 0};
    H5Sget_simple_extent_dims(space, dims, nullptr);
    H5Sclose(space);

    if (ndims == 1) {
        n_rows_ = static_cast<int64>(dims[0]);
        n_cols_ = 1;
    } else {
        n_rows_ = static_cast<int64>(dims[0]);
        n_cols_ = static_cast<int64>(dims[1]);
    }
}

// --- Constructor: create new ---

HDF5DenseMatrix::HDF5DenseMatrix(HDF5File* file, const std::string& dataset_path,
                                   int64 n_rows, int64 n_cols,
                                   const double* data, int compression)
    : file_(file), dataset_path_(dataset_path), dset_id_(-1),
      n_rows_(n_rows), n_cols_(n_cols), owns_dataset_(true), thread_file_(-1) {

    hid_t fid = file_->file_id();

    // Delete if exists
    if (H5Lexists(fid, dataset_path.c_str(), H5P_DEFAULT) > 0) {
        H5Ldelete(fid, dataset_path.c_str(), H5P_DEFAULT);
    }

    // Create intermediate groups
    std::string g;
    size_t pos = 0;
    while ((pos = dataset_path.find('/', pos + 1)) != std::string::npos) {
        g = dataset_path.substr(0, pos);
        if (H5Lexists(fid, g.c_str(), H5P_DEFAULT) <= 0) {
            hid_t grp = H5Gcreate2(fid, g.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Gclose(grp);
        }
    }

    hsize_t dims[2] = {static_cast<hsize_t>(n_rows), static_cast<hsize_t>(n_cols)};
    hsize_t chunk_dims[2] = {
        std::min(static_cast<hsize_t>(n_rows), static_cast<hsize_t>(1024)),
        std::min(static_cast<hsize_t>(n_cols), static_cast<hsize_t>(64))
    };

    hid_t plist = H5P_DEFAULT;
    if (compression > 0) {
        plist = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(plist, 2, chunk_dims);
        H5Pset_deflate(plist, compression);
    }

    hid_t space = H5Screate_simple(2, dims, nullptr);
    dset_id_ = H5Dcreate2(fid, dataset_path.c_str(), H5T_NATIVE_DOUBLE,
                           space, H5P_DEFAULT, plist, H5P_DEFAULT);
    H5Sclose(space);
    if (compression > 0) H5Pclose(plist);

    if (data) {
        check_h5_status(H5Dwrite(dset_id_, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data),
                        "H5Dwrite", dataset_path.c_str());
    }
}

HDF5DenseMatrix::~HDF5DenseMatrix() {
    if (dset_id_ >= 0) {
        H5Dclose(dset_id_);
    }
}

void HDF5DenseMatrix::read_2d(double* buffer,
                               int64 row_start, int64 row_count,
                               int64 col_start, int64 col_count) {
    hid_t fid = get_file_id();
    // Re-open if we don't own the ID (to get a fresh handle)
    hid_t dset = owns_dataset_ ? dset_id_ :
        H5Dopen2(fid, dataset_path_.c_str(), H5P_DEFAULT);

    hsize_t start[2] = {static_cast<hsize_t>(row_start),
                         static_cast<hsize_t>(col_start)};
    hsize_t count[2] = {static_cast<hsize_t>(row_count),
                         static_cast<hsize_t>(col_count)};

    hid_t memspace = H5Screate_simple(2, count, nullptr);
    hid_t filespace = H5Dget_space(dset);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, count, nullptr);
    check_h5_status(H5Dread(dset, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT, buffer),
                    "H5Dread", dataset_path_.c_str());
    H5Sclose(filespace);
    H5Sclose(memspace);

    if (!owns_dataset_) H5Dclose(dset);
}

void HDF5DenseMatrix::write_2d(const double* buffer,
                                int64 row_start, int64 row_count,
                                int64 col_start, int64 col_count) {
    std::lock_guard<std::mutex> lock(file_->write_mutex_);
    hid_t dset = owns_dataset_ ? dset_id_ :
        H5Dopen2(file_->file_id(), dataset_path_.c_str(), H5P_DEFAULT);

    hsize_t start[2] = {static_cast<hsize_t>(row_start),
                         static_cast<hsize_t>(col_start)};
    hsize_t count[2] = {static_cast<hsize_t>(row_count),
                         static_cast<hsize_t>(col_count)};

    hid_t memspace = H5Screate_simple(2, count, nullptr);
    hid_t filespace = H5Dget_space(dset);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, count, nullptr);
    check_h5_status(H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT, buffer),
                    "H5Dwrite", dataset_path_.c_str());
    H5Sclose(filespace);
    H5Sclose(memspace);

    if (!owns_dataset_) H5Dclose(dset);
}

void HDF5DenseMatrix::read_rows(double* buffer,
                                 int64 row_start, int64 row_count,
                                 int64 col_start, int64 col_count) {
    read_2d(buffer, row_start, row_count, col_start, col_count);
}

void HDF5DenseMatrix::write_rows(const double* buffer,
                                  int64 row_start, int64 row_count,
                                  int64 col_start, int64 col_count) {
    write_2d(buffer, row_start, row_count, col_start, col_count);
}

void HDF5DenseMatrix::read_cols(double* buffer,
                                 int64 row_start, int64 row_count,
                                 int64 col_start, int64 col_count) {
    read_2d(buffer, row_start, row_count, col_start, col_count);
}

void HDF5DenseMatrix::write_cols(const double* buffer,
                                  int64 row_start, int64 row_count,
                                  int64 col_start, int64 col_count) {
    write_2d(buffer, row_start, row_count, col_start, col_count);
}

void HDF5DenseMatrix::matvec(const double* x, double* y) {
    // y = A * x  (A: n_rows x n_cols, x: n_cols)
    std::fill(y, y + n_rows_, 0.0);

    int64 chunk_size = 1024;
    for (int64 r = 0; r < n_rows_; r += chunk_size) {
        int64 rc = std::min(chunk_size, n_rows_ - r);
        for (int64 c = 0; c < n_cols_; c += chunk_size) {
            int64 cc = std::min(chunk_size, n_cols_ - c);
            std::vector<double> block(rc * cc);
            read_2d(block.data(), r, rc, c, cc);
            for (int64 i = 0; i < rc; ++i) {
                double acc = 0.0;
                for (int64 j = 0; j < cc; ++j) {
                    acc += block[i * cc + j] * x[c + j];
                }
                y[r + i] += acc;
            }
        }
    }
}

void HDF5DenseMatrix::rmatvec(const double* x, double* y) {
    // y = A^T * x  (A: n_rows x n_cols, x: n_rows, y: n_cols)
    std::fill(y, y + n_cols_, 0.0);

    int64 chunk_size = 1024;
    for (int64 r = 0; r < n_rows_; r += chunk_size) {
        int64 rc = std::min(chunk_size, n_rows_ - r);
        std::vector<double> block(rc * n_cols_);
        read_2d(block.data(), r, rc, 0, n_cols_);
        for (int64 i = 0; i < rc; ++i) {
            double xi = x[r + i];
            if (xi == 0.0) continue;
            for (int64 j = 0; j < n_cols_; ++j) {
                y[j] += block[i * n_cols_ + j] * xi;
            }
        }
    }
}

void HDF5DenseMatrix::read_full(std::vector<double>& out) {
    out.resize(n_rows_ * n_cols_);
    // Use flat read to avoid row-major/column-major reinterpretation.
    // HDF5 stores raw bytes as-is; both Eigen and R use column-major,
    // so a flat copy preserves the correct element positions.
    hid_t dset = owns_dataset_ ? dset_id_ :
        H5Dopen2(get_file_id(), dataset_path_.c_str(), H5P_DEFAULT);
    check_h5_status(H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()),
                    "H5Dread", dataset_path_.c_str());
    if (!owns_dataset_) H5Dclose(dset);
}

void HDF5DenseMatrix::write_full(const double* data) {
    write_2d(data, 0, n_rows_, 0, n_cols_);
}

} // namespace sclean
