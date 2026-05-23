#ifndef SCLEAN_HDF5_CSC_MATRIX_H
#define SCLEAN_HDF5_CSC_MATRIX_H

#include "hdf5_file.h"
#include "core/disk_matrix.h"
#include <vector>
#include <string>

namespace sclean {

class HDF5CSCMatrix : public DiskMatrix {
public:
    // Open existing matrix at group_path inside file
    HDF5CSCMatrix(HDF5File* file, const std::string& group_path,
                  hid_t thread_file = -1);

    // Create new matrix from in-memory CSC data
    HDF5CSCMatrix(HDF5File* file, const std::string& group_path,
                  const std::vector<double>& data,
                  const std::vector<int32>& indices,
                  const std::vector<int64>& indptr,
                  int64 n_rows, int64 n_cols,
                  int compression = 3);

    ~HDF5CSCMatrix() override;

    // --- DiskMatrix interface ---
    int64 n_rows() const override { return n_rows_; }
    int64 n_cols() const override { return n_cols_; }
    int64 nnz()   const override { return nnz_; }
    bool is_sparse() const override { return true; }

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

    // --- Sparse-specific ---
    void read_sparse_rows(
        std::vector<double>& values,
        std::vector<int32>& indices,
        std::vector<int64>& row_starts,
        int64 row_start, int64 row_count) override;

    // --- Chunk boundary queries ---
    int64 col_nnz(int64 col) const;
    void col_nnz_range(int64 col_start, int64 col_count, int64* out) const;

    int64 memory_per_row_chunk(int64 n_cols) const override;
    int64 memory_per_col_chunk(int64 n_rows) const override;

private:
    HDF5File* file_;
    std::string group_path_;

    int64 n_rows_, n_cols_, nnz_;

    // Cached HDF5 dataset IDs
    hid_t dset_data_;
    hid_t dset_indices_;
    hid_t dset_indptr_;
    hid_t dset_row_ptr_;

    // In-memory row_ptr cache (avoids HDF5 write during read ops)
    std::vector<int64> row_ptr_cache_;

    // Per-thread file handle for reads
    hid_t thread_file_;

    void open_datasets(hid_t file_id);
    void close_datasets();
    void ensure_row_ptr();
    std::string full_path(const std::string& name) const;

    static void build_row_ptr(int64 n_rows, int64 n_cols,
                              const int32* indices, const int64* indptr,
                              int64* row_ptr_out);
};

} // namespace sclean

#endif // SCLEAN_HDF5_CSC_MATRIX_H
