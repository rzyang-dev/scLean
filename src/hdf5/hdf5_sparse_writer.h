#ifndef SCLEAN_HDF5_SPARSE_WRITER_H
#define SCLEAN_HDF5_SPARSE_WRITER_H

#include "hdf5_file.h"
#include "scLean_types.h"
#include <string>
#include <vector>

namespace sclean {

// Incremental CSC matrix writer.
// Collects sparse columns in memory buffers and flushes to HDF5 in batches
// to avoid per-column I/O overhead while keeping memory bounded.
class HDF5SparseWriter {
public:
    // batch_nnz: max non-zeros to buffer before flushing (default: 10M)
    HDF5SparseWriter(HDF5File* file, const std::string& group_path,
                     int64 n_rows, int64 n_cols, int64 total_nnz,
                     int64 batch_nnz = 10 * 1024 * 1024,
                     int compression = 3);

    ~HDF5SparseWriter();

    HDF5SparseWriter(const HDF5SparseWriter&) = delete;
    HDF5SparseWriter& operator=(const HDF5SparseWriter&) = delete;

    // Write one sparse column. values and row_indices must be sorted by row.
    void write_column(const double* values, const int32* row_indices,
                      int64 nnz_in_col);

    // Write a range of dense columns as sparse (zero entries are skipped).
    // Writes `ncol` columns from a dense column-major buffer.
    // Column j is at buffer[j * n_rows : (j+1) * n_rows].
    void write_dense_columns_sparse(const double* buffer,
                                     int64 n_rows, int64 n_cols);

    // Flush remaining buffered data and finalize the HDF5 file.
    // Must be called before destruction.
    void finalize();

    int64 total_nnz_written() const { return total_written_; }
    int64 columns_written() const { return cols_written_; }
    int64 buf_nnz() const { return buf_data_.size(); }

private:
    HDF5File* file_;
    std::string group_path_;
    int64 n_rows_, n_cols_;
    int64 batch_nnz_;
    int compression_;

    // Buffered sparse data
    std::vector<double> buf_data_;
    std::vector<int32>  buf_indices_;
    std::vector<int64>  buf_indptr_;  // indptr for buffered columns

    // HDF5 dataset IDs
    hid_t dset_data_;
    hid_t dset_indices_;
    hid_t dset_indptr_;

    // Totals written to HDF5 so far
    int64 total_written_;   // nnz written
    int64 cols_written_;    // columns written

    void create_datasets();
    void flush_buffer();
    void ensure_group_exists();
};

} // namespace sclean

#endif // SCLEAN_HDF5_SPARSE_WRITER_H
