#ifndef SCLEAN_HDF5_DENSE_MATRIX_H
#define SCLEAN_HDF5_DENSE_MATRIX_H

#include "hdf5_file.h"
#include "core/disk_matrix.h"
#include <string>
#include <vector>

namespace sclean {

class HDF5DenseMatrix : public DiskMatrix {
public:
    HDF5DenseMatrix(HDF5File* file, const std::string& dataset_path,
                    hid_t thread_file = -1);

    // Create a new dense dataset
    HDF5DenseMatrix(HDF5File* file, const std::string& dataset_path,
                    int64 n_rows, int64 n_cols,
                    const double* data = nullptr,
                    int compression = 1);

    ~HDF5DenseMatrix() override;

    int64 n_rows() const override { return n_rows_; }
    int64 n_cols() const override { return n_cols_; }
    int64 nnz()   const override { return n_rows_ * n_cols_; }
    bool is_dense() const override { return true; }

    void read_rows(double* buffer,
                   int64 row_start, int64 row_count,
                   int64 col_start, int64 col_count) override;
    void write_rows(const double* buffer,
                    int64 row_start, int64 row_count,
                    int64 col_start, int64 col_count) override;
    void read_cols(double* buffer,
                   int64 row_start, int64 row_count,
                   int64 col_start, int64 col_count) override;
    void write_cols(const double* buffer,
                    int64 row_start, int64 row_count,
                    int64 col_start, int64 col_count) override;

    void matvec(const double* x, double* y) override;
    void rmatvec(const double* x, double* y) override;

    void read_full(std::vector<double>& out);
    void write_full(const double* data);

private:
    HDF5File* file_;
    std::string dataset_path_;
    hid_t dset_id_;
    int64 n_rows_, n_cols_;
    bool owns_dataset_;
    hid_t thread_file_;

    hid_t get_file_id() const { return thread_file_ >= 0 ? thread_file_ : file_->file_id(); }

    void read_2d(double* buffer,
                 int64 row_start, int64 row_count,
                 int64 col_start, int64 col_count);
    void write_2d(const double* buffer,
                  int64 row_start, int64 row_count,
                  int64 col_start, int64 col_count);
};

} // namespace sclean

#endif // SCLEAN_HDF5_DENSE_MATRIX_H
