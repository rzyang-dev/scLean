#include "hdf5_file.h"
#include "hdf5_csc_matrix.h"
#include "hdf5_dense_matrix.h"

namespace sclean {

std::unique_ptr<HDF5CSCMatrix> HDF5File::open_csc_matrix(const std::string& group_path,
                                                          hid_t thread_file) {
    return std::make_unique<HDF5CSCMatrix>(this, group_path, thread_file);
}

std::unique_ptr<HDF5DenseMatrix> HDF5File::open_dense_matrix(const std::string& dataset_path) {
    return std::make_unique<HDF5DenseMatrix>(this, dataset_path);
}

std::unique_ptr<HDF5DenseMatrix> HDF5File::create_dense_dataset(
    const std::string& path, int64 n_rows, int64 n_cols) {
    return std::make_unique<HDF5DenseMatrix>(this, path, n_rows, n_cols);
}

std::unique_ptr<HDF5CSCMatrix> HDF5File::create_csc_matrix(
    const std::string& group_path,
    const std::vector<double>& data,
    const std::vector<int32>& indices,
    const std::vector<int64>& indptr,
    int64 n_rows, int64 n_cols) {
    return std::make_unique<HDF5CSCMatrix>(
        this, group_path, data, indices, indptr, n_rows, n_cols);
}

} // namespace sclean
