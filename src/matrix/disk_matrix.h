#ifndef SCLEAN_DISK_MATRIX_H
#define SCLEAN_DISK_MATRIX_H

#include <cstdint>
#include <vector>
#include "scLean_types.h"

namespace sclean {

enum class ChunkAxis { Rows = 0, Columns = 1 };

class DiskMatrix {
public:
    virtual ~DiskMatrix() = default;

    virtual int64 n_rows() const = 0;
    virtual int64 n_cols() const = 0;
    virtual int64 nnz()   const = 0;
    virtual bool  is_dense() const { return false; }
    virtual bool  is_sparse() const { return false; }

    // Dense chunked I/O
    virtual void read_rows(double* buffer,
                           int64 row_start, int64 row_count,
                           int64 col_start, int64 col_count) = 0;
    virtual void write_rows(const double* buffer,
                            int64 row_start, int64 row_count,
                            int64 col_start, int64 col_count) = 0;
    virtual void read_cols(double* buffer,
                           int64 row_start, int64 row_count,
                           int64 col_start, int64 col_count) = 0;
    virtual void write_cols(const double* buffer,
                            int64 row_start, int64 row_count,
                            int64 col_start, int64 col_count) = 0;

    // Matrix-vector products (critical for IRLBA)
    virtual void matvec(const double* x, double* y) = 0;   // y = A * x
    virtual void rmatvec(const double* x, double* y) = 0;  // y = A^T * x

    // Sparse-specific row access
    virtual void read_sparse_rows(
        std::vector<double>& values,
        std::vector<int32>& indices,
        std::vector<int64>& row_starts,
        int64 row_start, int64 row_count) {}

    // Memory estimation for chunk scheduling
    virtual int64 memory_per_row_chunk(int64 n_cols) const {
        return n_cols * sizeof(double);
    }
    virtual int64 memory_per_col_chunk(int64 n_rows) const {
        return n_rows * sizeof(double);
    }
};

} // namespace sclean

#endif // SCLEAN_DISK_MATRIX_H
