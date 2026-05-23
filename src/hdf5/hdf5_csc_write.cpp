#include "hdf5_csc_matrix.h"
#include "hdf5_utils.h"
#include <stdexcept>
#include <vector>
#include <mutex>

namespace sclean {

// --- Column chunk write ---

void HDF5CSCMatrix::write_cols(const double* buffer,
                                int64 row_start, int64 row_count,
                                int64 col_start, int64 col_count) {
    // For writing, we need to update the sparse representation.
    // Read existing data for these columns, merge with new data,
    // and write back. This is expensive for CSC -- use batch rewrite.
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

// --- Row chunk write ---

void HDF5CSCMatrix::write_rows(const double* /*buffer*/,
                                int64 /*row_start*/, int64 /*row_count*/,
                                int64 /*col_start*/, int64 /*col_count*/) {
    throw std::runtime_error(
        "HDF5CSCMatrix::write_rows: incremental write not yet implemented.");
}

} // namespace sclean
