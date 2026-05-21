#ifndef SCLEAN_MTX_READER_H
#define SCLEAN_MTX_READER_H

#include <string>
#include <vector>
#include <cstdio>
#include <zlib.h>
#include "sclean_types.h"

namespace sclean {

class HDF5SparseWriter;
class HDF5File;

struct MtxHeader {
    int64 n_rows;
    int64 n_cols;
    int64 nnz;
    bool is_pattern;   // "pattern" matrix has no values (all 1.0)
    bool is_integer;   // "integer" matrix (values are ints, cast to double)
};

// Per-column accumulation buffer during streaming parse.
struct ColBuffer {
    int64 current_col = -1;
    std::vector<double> values;
    std::vector<int32> row_indices;

    void clear() {
        current_col = -1;
        values.clear();
        row_indices.clear();
    }
};

// Streaming reader for 10X-format directories.
// Reads matrix.mtx(.gz), barcodes.tsv(.gz), features.tsv(.gz)
// and streams sparse columns through an HDF5SparseWriter.
class MtxReader {
public:
    explicit MtxReader(const std::string& tenx_dir);
    ~MtxReader();

    MtxReader(const MtxReader&) = delete;
    MtxReader& operator=(const MtxReader&) = delete;

    // Parse MatrixMarket header. Returns dimensions and nnz.
    // May be called independently before stream_to_writer().
    MtxHeader parse_header();

    // Stream all coordinate triples through the writer.
    // Populates row_nnz (size n_rows) with per-row NNZ counts
    // for subsequent row_ptr construction.
    void stream_to_writer(HDF5SparseWriter& writer,
                          std::vector<int64>& row_nnz);

    // Read barcodes.tsv(.gz) and features.tsv(.gz) first column.
    std::vector<std::string> read_barcodes();
    std::vector<std::string> read_features();

private:
    std::string tenx_dir_;
    std::string mtx_path_;
    bool mtx_gzipped_;
    gzFile gz_file_;
    std::FILE* plain_file_;

    void open_mtx();
    void close_mtx();
    bool read_line(std::string& line);
    void flush_column(ColBuffer& buf, HDF5SparseWriter& writer,
                      std::vector<int64>& row_nnz);

    // Read a specific column (0-indexed) of a TSV/TSV.GZ file.
    std::vector<std::string> read_tsv_column(
        const std::string& filename, int col_idx = 0);

    // Find matrix.mtx or matrix.mtx.gz in tenx_dir_.
    std::string find_mtx_path();
};

// Build CSR row_ptr from per-row NNZ counts.
std::vector<int64> build_row_ptr_from_counts(
    const std::vector<int64>& row_nnz, int64 n_rows);

} // namespace sclean

#endif // SCLEAN_MTX_READER_H
