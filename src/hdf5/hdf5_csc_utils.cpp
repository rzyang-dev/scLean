#include "hdf5_csc_matrix.h"
#include "hdf5_utils.h"
#include <vector>

namespace sclean {

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
