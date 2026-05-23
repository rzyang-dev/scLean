#include <Rcpp.h>
#include <RcppEigen.h>
#include <string>
#include <vector>
#include <unordered_set>
#include <unistd.h>

#include "hdf5/hdf5_file.h"
#include "hdf5/hdf5_csc_matrix.h"
#include "hdf5/hdf5_dense_matrix.h"
#include "hdf5/hdf5_sparse_writer.h"
#include "io/mtx_reader.h"

using namespace Rcpp;
using namespace sclean;

// ============================================================
// HDF5 I/O helpers
// ============================================================


// [[Rcpp::export]]
void write_csc_to_hdf5(std::string hdf5_path, std::string group_path,
                        std::vector<double> data, std::vector<int> indices,
                        std::vector<double> indptr,
                        int n_rows, int n_cols) {
    std::vector<int64> indptr64(indptr.begin(), indptr.end());
    std::vector<int32> idx(indices.begin(), indices.end());

    // Check if file exists — create if new, open if existing
    bool file_exists = (access(hdf5_path.c_str(), F_OK) == 0);
    FileMode mode = file_exists ? FileMode::ReadWrite : FileMode::Create;
    HDF5File file(hdf5_path, mode);

    // Create intermediate groups (skip root "/")
    std::string g;
    size_t pos = 1;  // start after leading "/"
    while ((pos = group_path.find('/', pos)) != std::string::npos) {
        g = group_path.substr(0, pos);
        if (!g.empty() && !file.exists(g)) file.create_group(g);
        pos++;
    }
    if (!file.exists(group_path)) file.create_group(group_path);

    file.create_csc_matrix(group_path, data, idx, indptr64, n_rows, n_cols);
}


// [[Rcpp::export]]
IntegerVector read_hdf5_shape(std::string hdf5_path, std::string group_path) {
    HDF5File file(hdf5_path, FileMode::ReadOnly);
    auto mat = file.open_csc_matrix(group_path);
    IntegerVector out(2);
    out[0] = static_cast<int>(mat->n_rows());
    out[1] = static_cast<int>(mat->n_cols());
    return out;
}


// [[Rcpp::export]]
NumericMatrix read_hdf5_dense_chunk(std::string hdf5_path, std::string group_path,
                                     int row_start, int row_count,
                                     int col_start, int col_count) {
    HDF5File file(hdf5_path, FileMode::ReadOnly);

    // Detect whether path is a dense dataset or a CSC group.
    // CSC groups contain an "indptr" child; dense datasets do not.
    hid_t fid = file.file_id();
    bool is_csc = (H5Lexists(fid, (group_path + "/indptr").c_str(), H5P_DEFAULT) > 0);

    NumericMatrix out(row_count, col_count);
    double* rdata = REAL(out);

    if (!is_csc) {
        // Read the full dense dataset as a flat array.
        // HDF5 stores raw bytes as-is; Eigen writes column-major,
        // so flat[pos] = Eigen(pos % nRows, pos / nRows).
        // R also uses column-major, so we index flat the same way.
        hid_t dset = H5Dopen2(fid, group_path.c_str(), H5P_DEFAULT);
        hid_t space = H5Dget_space(dset);
        hsize_t dims[2];
        H5Sget_simple_extent_dims(space, dims, nullptr);
        H5Sclose(space);
        int64 n_rows_full = static_cast<int64>(dims[0]);
        int64 n_cols_full = static_cast<int64>(dims[1]);
        hsize_t total = static_cast<hsize_t>(n_rows_full * n_cols_full);

        std::vector<double> flat(total);
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data());
        H5Dclose(dset);

        // Index into the flat column-major array
        for (int r = 0; r < row_count; ++r)
            for (int c = 0; c < col_count; ++c)
                rdata[r + c * row_count] =
                    flat[(row_start + r) + (col_start + c) * n_rows_full];
    } else {
        auto mat = file.open_csc_matrix(group_path);
        std::vector<double> buf(row_count * col_count);
        mat->read_rows(buf.data(), row_start, row_count, col_start, col_count);
        // read_rows fills row-major; convert to column-major (R)
        for (int r = 0; r < row_count; ++r)
            for (int c = 0; c < col_count; ++c)
                rdata[r + c * row_count] = buf[r * col_count + c];
    }

    return out;
}


// [[Rcpp::export]]
List read_hdf5_as_dgCMatrix(std::string hdf5_path, std::string group_path) {
    HDF5File file(hdf5_path, FileMode::ReadOnly);
    auto mat = file.open_csc_matrix(group_path);

    int64 n_rows = mat->n_rows();
    int64 n_cols = mat->n_cols();
    int64 nnz = mat->nnz();

    // Read all data
    // We need to read the underlying datasets directly
    // For now, read via sparse_rows for each chunk
    std::vector<double> all_values;
    std::vector<int> all_indices; // R uses int for dgCMatrix i slot
    std::vector<int64> all_indptr;
    all_indptr.push_back(0);

    int64 chunk_size = 1024;
    for (int64 c = 0; c < n_cols; c += chunk_size) {
        int64 cc = std::min(chunk_size, n_cols - c);

        // Read indptr slice
        hid_t fid = file.file_id();
        std::string indptr_path = group_path + "/indptr";
        hid_t d = H5Dopen2(fid, indptr_path.c_str(), H5P_DEFAULT);

        std::vector<int64> ip_chunk(cc + 1);
        hsize_t start[1] = {static_cast<hsize_t>(c)};
        hsize_t count[1] = {static_cast<hsize_t>(cc + 1)};
        hid_t ms = H5Screate_simple(1, count, nullptr);
        hid_t fs = H5Dget_space(d);
        H5Sselect_hyperslab(fs, H5S_SELECT_SET, start, nullptr, count, nullptr);
        H5Dread(d, H5T_NATIVE_INT64, ms, fs, H5P_DEFAULT, ip_chunk.data());
        H5Sclose(fs); H5Sclose(ms); H5Dclose(d);

        int64 chunk_nnz = ip_chunk[cc] - ip_chunk[0];
        int64 base = static_cast<int64>(all_values.size()); // before appending
        if (chunk_nnz > 0) {
            // Read data and indices
            std::string data_path = group_path + "/data";
            std::string idx_path = group_path + "/indices";

            std::vector<double> dv(chunk_nnz);
            std::vector<int32> iv(chunk_nnz);

            hid_t ds = H5Dopen2(fid, data_path.c_str(), H5P_DEFAULT);
            hsize_t ds_start[1] = {static_cast<hsize_t>(ip_chunk[0])};
            hsize_t ds_count[1] = {static_cast<hsize_t>(chunk_nnz)};
            hid_t data_ms = H5Screate_simple(1, ds_count, nullptr);
            hid_t dfs = H5Dget_space(ds);
            H5Sselect_hyperslab(dfs, H5S_SELECT_SET, ds_start, nullptr, ds_count, nullptr);
            H5Dread(ds, H5T_NATIVE_DOUBLE, data_ms, dfs, H5P_DEFAULT, dv.data());
            H5Sclose(dfs); H5Sclose(data_ms); H5Dclose(ds);

            hid_t is = H5Dopen2(fid, idx_path.c_str(), H5P_DEFAULT);
            hid_t ifs = H5Dget_space(is);
            H5Sselect_hyperslab(ifs, H5S_SELECT_SET, ds_start, nullptr, ds_count, nullptr);
            hid_t idx_ms = H5Screate_simple(1, ds_count, nullptr);
            H5Dread(is, H5T_NATIVE_INT32, idx_ms, ifs, H5P_DEFAULT, iv.data());
            H5Sclose(ifs); H5Dclose(is);
            H5Sclose(idx_ms);

            all_values.insert(all_values.end(), dv.begin(), dv.end());
            all_indices.insert(all_indices.end(), iv.begin(), iv.end());
        }

        // Add indptr values offset by NNZ accumulated so far
        for (int64 j = 1; j <= cc; ++j) {
            all_indptr.push_back(base + (ip_chunk[j] - ip_chunk[0]));
        }
    }

    // Build dgCMatrix in R
    List out;
    out["i"] = IntegerVector(all_indices.begin(), all_indices.end());
    out["p"] = IntegerVector(all_indptr.begin(), all_indptr.end());
    out["x"] = NumericVector(all_values.begin(), all_values.end());
    out["Dim"] = IntegerVector::create(
        static_cast<int>(n_rows), static_cast<int>(n_cols));
    out.attr("class") = "dgCMatrix";

    return out;
}


// [[Rcpp::export]]
std::vector<double> hdf5_matvec(std::string hdf5_path, std::string group_path,
                                 std::vector<double> x) {
    HDF5File file(hdf5_path, FileMode::ReadOnly);
    auto mat = file.open_csc_matrix(group_path);

    std::vector<double> y(mat->n_rows());
    mat->matvec(x.data(), y.data());
    return y;
}


// [[Rcpp::export]]
void write_strings_to_hdf5(std::string hdf5_path, std::string path,
                            std::vector<std::string> data) {
    HDF5File file(hdf5_path, FileMode::ReadWrite);
    file.write_vector_string(path, data);
}


// [[Rcpp::export]]
std::vector<std::string> read_strings_from_hdf5(std::string hdf5_path,
                                                  std::string path) {
    HDF5File file(hdf5_path, FileMode::ReadOnly);
    return file.read_vector_string(path);
}


// [[Rcpp::export]]
std::vector<int> read_hdf5_int32(std::string hdf5_path, std::string path) {
    HDF5File file(hdf5_path, FileMode::ReadOnly);
    auto v = file.read_vector_int32(path);
    return std::vector<int>(v.begin(), v.end());
}


// [[Rcpp::export]]
std::vector<int> read_vector_int8(std::string hdf5_path, std::string path) {
    HDF5File file(hdf5_path, FileMode::ReadOnly);
    auto v = file.read_vector_int32(path);
    return std::vector<int>(v.begin(), v.end());
}


// [[Rcpp::export]]
std::vector<double> read_hdf5_double(std::string hdf5_path, std::string path) {
    HDF5File file(hdf5_path, FileMode::ReadOnly);
    return file.read_vector_double(path);
}


// [[Rcpp::export]]
std::vector<double> read_hdf5_float32(std::string hdf5_path, std::string path) {
    HDF5File file(hdf5_path, FileMode::ReadOnly);
    // HDF5 stores float32 but we read as double for R compatibility
    return file.read_vector_double(path);
}


// [[Rcpp::export]]
NumericMatrix read_dense_matrix(std::string hdf5_path, std::string dataset_path) {
    HDF5File file(hdf5_path, FileMode::ReadOnly);
    auto mat = file.open_dense_matrix(dataset_path);

    NumericMatrix out(mat->n_rows(), mat->n_cols());

    std::vector<double> buf(mat->n_rows() * mat->n_cols());
    mat->read_full(buf);
    std::copy(buf.begin(), buf.end(), REAL(out));

    return out;
}


// [[Rcpp::export]]
std::vector<std::string> list_hdf5_groups(std::string hdf5_path,
                                            std::string parent_path) {
    HDF5File file(hdf5_path, FileMode::ReadOnly);
    std::vector<std::string> groups;

    hid_t fid = file.file_id();
    if (!file.exists(parent_path)) return groups;

    hid_t group = H5Gopen2(fid, parent_path.c_str(), H5P_DEFAULT);
    hsize_t n_objs;
    H5Gget_num_objs(group, &n_objs);

    char name_buf[1024];
    for (hsize_t i = 0; i < n_objs; ++i) {
        H5Gget_objname_by_idx(group, i, name_buf, sizeof(name_buf));
        int objtype = H5Gget_objtype_by_idx(group, i);
        if (objtype == H5G_GROUP || objtype == H5G_DATASET) {
            groups.push_back(std::string(name_buf));
        }
    }
    H5Gclose(group);
    return groups;
}


// ============================================================
// Streaming 10X import
// ============================================================

// [[Rcpp::export]]
void stream_10x_to_hdf5(std::string tenx_dir,
                         std::string hdf5_path,
                         std::string assay = "RNA") {

    HDF5File file(hdf5_path, FileMode::Create);

    MtxReader reader(tenx_dir);
    MtxHeader hdr = reader.parse_header();

    std::string counts_group = "/assays/" + assay + "/layers/counts";

    HDF5SparseWriter writer(&file, counts_group,
                            hdr.n_rows, hdr.n_cols, hdr.nnz,
                            10 * 1024 * 1024,  // 10M batch_nnz
                            3);                // compression

    std::vector<int64> row_nnz;
    reader.stream_to_writer(writer, row_nnz);

    // Write row_ptr (CSR pointer — matches create_csc_matrix output)
    std::vector<int64> row_ptr = build_row_ptr_from_counts(row_nnz, hdr.n_rows);
    file.write_vector_int64(counts_group + "/row_ptr", row_ptr);

    // Write cell/feature metadata
    std::vector<std::string> barcodes = reader.read_barcodes();
    file.write_vector_string("/cells/names", barcodes);

    std::vector<std::string> features = reader.read_features();

    // Deduplicate gene symbols (mimics R's make.unique behavior).
    // Seurat sets gene symbols as dgCMatrix row names; R auto-appends
    // .1, .2 suffixes to duplicate dimnames. We replicate this here.
    {
        std::unordered_set<std::string> seen;
        for (size_t i = 0; i < features.size(); ++i) {
            std::string orig = features[i];
            if (seen.find(orig) == seen.end()) {
                seen.insert(orig);
            } else {
                int suffix = 1;
                std::string candidate;
                do {
                    candidate = orig + "." + std::to_string(suffix++);
                } while (seen.find(candidate) != seen.end());
                features[i] = candidate;
                seen.insert(candidate);
            }
        }
    }

    file.write_vector_string("/features/names", features);
}
