#include "mtx_reader.h"
#include "hdf5/hdf5_sparse_writer.h"
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

namespace sclean {

// === File helpers ===

static bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

// === MtxReader ===

MtxReader::MtxReader(const std::string& tenx_dir)
    : tenx_dir_(tenx_dir), mtx_gzipped_(false),
      gz_file_(nullptr), plain_file_(nullptr) {

    mtx_path_ = find_mtx_path();
    if (mtx_path_.empty()) {
        throw std::runtime_error(
            "No matrix.mtx or matrix.mtx.gz found in: " + tenx_dir_);
    }
}

MtxReader::~MtxReader() {
    close_mtx();
}

std::string MtxReader::find_mtx_path() {
    std::string plain = tenx_dir_ + "/matrix.mtx";
    if (file_exists(plain)) {
        mtx_gzipped_ = false;
        return plain;
    }
    std::string gz = tenx_dir_ + "/matrix.mtx.gz";
    if (file_exists(gz)) {
        mtx_gzipped_ = true;
        return gz;
    }
    return "";
}

// === MTX file I/O ===

void MtxReader::open_mtx() {
    if (mtx_gzipped_) {
        gz_file_ = gzopen(mtx_path_.c_str(), "rb");
        if (!gz_file_) {
            throw std::runtime_error("Cannot open: " + mtx_path_);
        }
    } else {
#ifdef _WIN32
        fopen_s(&plain_file_, mtx_path_.c_str(), "r");
#else
        plain_file_ = std::fopen(mtx_path_.c_str(), "r");
#endif
        if (!plain_file_) {
            throw std::runtime_error("Cannot open: " + mtx_path_);
        }
    }
}

void MtxReader::close_mtx() {
    if (gz_file_) {
        gzclose(gz_file_);
        gz_file_ = nullptr;
    }
    if (plain_file_) {
        std::fclose(plain_file_);
        plain_file_ = nullptr;
    }
}

bool MtxReader::read_line(std::string& line) {
    line.clear();
    char buf[65536];
    if (mtx_gzipped_) {
        if (gzgets(gz_file_, buf, sizeof(buf)) == Z_NULL) return false;
    } else {
        if (std::fgets(buf, sizeof(buf), plain_file_) == nullptr) return false;
    }
    line = buf;
    // Trim trailing whitespace / CRLF
    while (!line.empty() && (static_cast<unsigned char>(line.back()) <= ' ')) {
        line.pop_back();
    }
    return true;
}

// === Header parsing ===

MtxHeader MtxReader::parse_header() {
    open_mtx();
    MtxHeader hdr = {};
    hdr.is_pattern = false;
    hdr.is_integer = false;

    std::string line;
    bool found_format = false;
    bool found_dims = false;

    while (!found_dims && read_line(line)) {
        if (line.empty()) continue;

        if (line[0] == '%') {
            // %%MatrixMarket format line
            if (line.size() > 15 && line.rfind("%%MatrixMarket", 0) == 0) {
                found_format = true;
                if (line.find("pattern") != std::string::npos) {
                    hdr.is_pattern = true;
                }
                if (line.find("integer") != std::string::npos) {
                    hdr.is_integer = true;
                }
            }
            continue;
        }

        // This is the dimensions line (first non-comment line)
        std::istringstream iss(line);
        if (!(iss >> hdr.n_rows >> hdr.n_cols >> hdr.nnz)) {
            close_mtx();
            throw std::runtime_error("Failed to parse MTX dimensions: " + line);
        }
        found_dims = true;
    }

    close_mtx();

    if (!found_dims) {
        throw std::runtime_error("MTX file is missing dimensions line");
    }

    return hdr;
}

// === Column flushing ===

void MtxReader::flush_column(ColBuffer& buf, HDF5SparseWriter& writer,
                              std::vector<int64>& row_nnz) {
    int64 nnz = static_cast<int64>(buf.values.size());
    if (nnz == 0) {
        writer.write_column(nullptr, nullptr, 0);
        return;
    }

    // Verify rows are sorted within this column (10X guarantees this,
    // but be defensive against malformed files).
    bool sorted = true;
    for (int64 i = 1; i < nnz; ++i) {
        if (buf.row_indices[i] < buf.row_indices[i - 1]) {
            sorted = false;
            break;
        }
    }

    if (!sorted) {
        // Reorder pairs by row index
        std::vector<std::pair<int32, double>> pairs(static_cast<size_t>(nnz));
        for (int64 i = 0; i < nnz; ++i) {
            pairs[static_cast<size_t>(i)] = {buf.row_indices[static_cast<size_t>(i)],
                                             buf.values[static_cast<size_t>(i)]};
        }
        std::sort(pairs.begin(), pairs.end());
        for (int64 i = 0; i < nnz; ++i) {
            buf.row_indices[static_cast<size_t>(i)] = pairs[static_cast<size_t>(i)].first;
            buf.values[static_cast<size_t>(i)] = pairs[static_cast<size_t>(i)].second;
        }
    }

    writer.write_column(buf.values.data(), buf.row_indices.data(), nnz);
}

// === Streaming parser ===

void MtxReader::stream_to_writer(HDF5SparseWriter& writer,
                                  std::vector<int64>& row_nnz) {
    MtxHeader hdr = parse_header();
    row_nnz.assign(static_cast<size_t>(hdr.n_rows), 0);

    open_mtx();

    // Skip header lines (format + dimensions) before streaming data.
    // parse_header() already consumed these; re-opening the file
    // restarts from the beginning, so we must skip them again.
    {
        std::string skip_line;
        bool skipped_format = false;
        bool skipped_dims = false;
        while (!skipped_dims && read_line(skip_line)) {
            if (skip_line.empty()) continue;
            if (skip_line[0] == '%') {
                if (!skipped_format && skip_line.rfind("%%MatrixMarket", 0) == 0) {
                    skipped_format = true;
                }
                continue;
            }
            // First non-comment line after format is the dimensions line
            skipped_dims = true;
        }
    }

    ColBuffer buf;
    std::string line;
    int64 entries_parsed = 0;

    while (read_line(line)) {
        if (line.empty() || line[0] == '%') continue;

        // Fast parse: row col [value]  (1-based indices in MTX)
        const char* p = line.c_str();
        char* end = nullptr;
        errno = 0;
        int64 row = strtoll(p, &end, 10);
        if (errno || end == p) {
            throw std::runtime_error("Bad MTX line (expected 'row col val'): " + line);
        }
        int64 col = strtoll(end, &end, 10);
        if (errno) {
            throw std::runtime_error("Bad MTX line (bad column): " + line);
        }
        double value = 1.0;
        if (!hdr.is_pattern) {
            value = strtod(end, nullptr);
            if (errno == ERANGE) {
                throw std::runtime_error("Bad MTX line (value overflow): " + line);
            }
        }

        int32 row0 = static_cast<int32>(row - 1);
        int64 col0 = col - 1;

        if (row0 < 0 || static_cast<int64>(row0) >= hdr.n_rows ||
            col0 < 0 || col0 >= hdr.n_cols) {
            throw std::runtime_error(
                "MTX entry out of bounds: row=" + std::to_string(row) +
                " col=" + std::to_string(col) +
                " (expected rows in [1," + std::to_string(hdr.n_rows) +
                "], cols in [1," + std::to_string(hdr.n_cols) + "])");
        }

        // Column boundary crossed — flush previous column
        if (col0 != buf.current_col) {
            if (buf.current_col >= 0) {
                flush_column(buf, writer, row_nnz);
            }
            // Fill empty columns between previous and current
            for (int64 c = buf.current_col + 1; c < col0; ++c) {
                writer.write_column(nullptr, nullptr, 0);
            }
            buf.current_col = col0;
            buf.values.clear();
            buf.row_indices.clear();
        }

        buf.values.push_back(value);
        buf.row_indices.push_back(row0);
        row_nnz[static_cast<size_t>(row0)]++;
        entries_parsed++;
    }

    // Flush last column
    if (buf.current_col >= 0) {
        flush_column(buf, writer, row_nnz);
    }

    // Fill trailing empty columns
    for (int64 c = buf.current_col + 1; c < hdr.n_cols; ++c) {
        writer.write_column(nullptr, nullptr, 0);
    }

    close_mtx();
    writer.finalize();

    if (entries_parsed != hdr.nnz) {
        throw std::runtime_error(
            "MTX entry count mismatch: expected " + std::to_string(hdr.nnz) +
            ", found " + std::to_string(entries_parsed));
    }
}

// === TSV reading ===

std::vector<std::string> MtxReader::read_tsv_column(
    const std::string& filename, int col_idx) {

    std::string path = tenx_dir_ + "/" + filename;
    std::vector<std::string> result;

    // Check for .gz variant if plain doesn't exist
    bool gz = false;
    if (!file_exists(path)) {
        std::string gz_path = path + ".gz";
        if (file_exists(gz_path)) {
            path = gz_path;
            gz = true;
        } else {
            return result;  // neither .tsv nor .tsv.gz exists
        }
    }

    auto trim = [](std::string& s) {
        while (!s.empty() &&
               (static_cast<unsigned char>(s.back()) <= ' ')) s.pop_back();
    };

    auto extract_field = [&](const std::string& line, int target_col) -> std::string {
        if (target_col <= 0) {
            size_t tab = line.find('\t');
            return (tab == std::string::npos) ? line : line.substr(0, tab);
        }
        size_t start = 0;
        for (int c = 0; c < target_col; ++c) {
            size_t tab = line.find('\t', start);
            if (tab == std::string::npos) return "";  // requested column doesn't exist
            start = tab + 1;
        }
        size_t next_tab = line.find('\t', start);
        return (next_tab == std::string::npos) ? line.substr(start) : line.substr(start, next_tab - start);
    };

    if (gz) {
        gzFile f = gzopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("Cannot open: " + path);
        char buf[65536];
        while (gzgets(f, buf, sizeof(buf)) != Z_NULL) {
            std::string line(buf);
            trim(line);
            if (line.empty()) continue;
            result.push_back(extract_field(line, col_idx));
        }
        gzclose(f);
    } else {
        std::FILE* f = nullptr;
#ifdef _WIN32
        fopen_s(&f, path.c_str(), "r");
#else
        f = std::fopen(path.c_str(), "r");
#endif
        if (!f) throw std::runtime_error("Cannot open: " + path);
        char buf[65536];
        while (std::fgets(buf, sizeof(buf), f) != nullptr) {
            std::string line(buf);
            trim(line);
            if (line.empty()) continue;
            result.push_back(extract_field(line, col_idx));
        }
        std::fclose(f);
    }
    return result;
}

std::vector<std::string> MtxReader::read_barcodes() {
    return read_tsv_column("barcodes.tsv", 0);
}

std::vector<std::string> MtxReader::read_features() {
    // 10X V3 uses "features.tsv", V2 uses "genes.tsv".
    // Both typically have gene symbol in column 2 (index 1).
    // Prefer gene symbol for consistency with Seurat::Read10X.
    const char* filenames[] = {"features.tsv", "genes.tsv"};
    for (auto fname : filenames) {
        std::string path = tenx_dir_ + "/" + fname;
        if (!file_exists(path) && !file_exists(path + ".gz")) continue;

        std::vector<std::string> sym = read_tsv_column(fname, 1);
        if (!sym.empty() && !sym[0].empty()) return sym;

        // Column 1 was empty — only 1 column exists, use column 0
        return read_tsv_column(fname, 0);
    }
    return {};
}

// === row_ptr builder ===

std::vector<int64> build_row_ptr_from_counts(
    const std::vector<int64>& row_nnz, int64 n_rows) {
    std::vector<int64> row_ptr(static_cast<size_t>(n_rows) + 1, 0);
    for (int64 i = 0; i < n_rows; ++i) {
        row_ptr[static_cast<size_t>(i) + 1] =
            row_ptr[static_cast<size_t>(i)] +
            row_nnz[static_cast<size_t>(i)];
    }
    return row_ptr;
}

} // namespace sclean
