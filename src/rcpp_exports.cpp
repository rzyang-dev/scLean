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
#include "normalize/normalize_operator.h"
#include "scale/scale_operator.h"
#include "normalize/vst_operator.h"
#include "pca/pca_operator.h"
#include "neighbors/neighbor_operator.h"
#include "neighbors/cluster_operator.h"
#include "markers/marker_operator.h"
#include "integration/integration_operator.h"
#include "io/mtx_reader.h"
#include "utils/memory_probe.h"
#include "utils/chunk_scheduler.h"
#include "utils/resource_monitor.h"
#include "utils/parallel.h"

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
// Pipeline implementations
// ============================================================

static ChunkScheduler make_scheduler() {
    int64 ram = 0;
    Environment base = Environment::base_namespace();
    if (base.exists("getOption")) {
        Function getOpt = base["getOption"];
        RObject ram_opt = getOpt("scLean.max_ram");
        if (!ram_opt.isNULL()) {
            NumericVector v = as<NumericVector>(ram_opt);
            if (v.size() > 0) ram = static_cast<int64>(v[0]);
        }
    }
    return ChunkScheduler(ram);
}


// [[Rcpp::export]]
List cpp_normalize(std::string hdf5_path, std::string assay_group,
                   int method, double scale_factor,
                   int chunk_size, bool in_memory) {
    int64 t0 = sclean::wall_time_ns();
    int64 rss0 = sclean::current_rss_bytes();

    HDF5File file(hdf5_path, FileMode::ReadWrite);
    auto scheduler = make_scheduler();
    if (chunk_size > 0) scheduler.set_chunk_size(chunk_size);
    std::string input_group = assay_group + "/layers/counts";
    std::string output_group = assay_group + "/layers/data";
    NormalizeOperator op(static_cast<NormalizeMethod>(method), scale_factor);
    // HDF5-backed read ops need --enable-threadsafe for multi-threading
    int n_threads = get_num_threads();
    NormalizeResult res = op.run(&file, input_group, output_group, scheduler, n_threads);

    // Write size factors to file
    file.write_vector_double(assay_group + "/layers/data/size_factors",
                              res.size_factors);

    int64 t1 = sclean::wall_time_ns();
    int64 rss1 = sclean::current_rss_bytes();
    REprintf("[NormalizeData] %d cells x %d genes | %.2f s | RSS %.0f -> %.0f MB\n",
             static_cast<int>(res.n_cells), static_cast<int>(res.n_genes),
             static_cast<double>(t1 - t0) / 1e9,
             static_cast<double>(rss0) / 1048576.0,
             static_cast<double>(rss1) / 1048576.0);

    return List::create(
        _["n_cells"] = res.n_cells,
        _["n_genes"] = res.n_genes
    );
}


// [[Rcpp::export]]
List cpp_scale(std::string hdf5_path, std::string assay_group,
               bool do_scale, bool do_center, int chunk_size, bool in_memory) {
    int64 t0 = sclean::wall_time_ns();
    int64 rss0 = sclean::current_rss_bytes();

    HDF5File file(hdf5_path, FileMode::ReadWrite);
    auto scheduler = make_scheduler();
    if (chunk_size > 0) scheduler.set_chunk_size(chunk_size);

    std::string input_group = assay_group + "/layers/data";

    ScaleOperator op(do_scale, do_center);
    // Only compute mean/sd — don't materialize scale.data on disk.
    // PCA will apply centering on-the-fly during Lanczos.
    auto input = file.open_csc_matrix(input_group);
    int64 n_genes = input->n_rows();
    int64 n_cells = input->n_cols();
    // HDF5-backed read ops need --enable-threadsafe for multi-threading
    int n_threads = get_num_threads();
    ScaleResult res = op.compute_stats(&file, input_group, n_genes, n_cells,
                                       scheduler, n_threads);

    // Write gene means/SDs as per-gene metadata (tiny: ~16 bytes/gene)
    file.write_vector_double(assay_group + "/features/residual_mean",
                              res.gene_means);
    file.write_vector_double(assay_group + "/features/residual_sd",
                              res.gene_sds);

    int64 t1 = sclean::wall_time_ns();
    int64 rss1 = sclean::current_rss_bytes();
    REprintf("[ScaleData] %d genes x %d cells | %.2f s | RSS %.0f -> %.0f MB\n",
             static_cast<int>(res.n_genes), static_cast<int>(res.n_cells),
             static_cast<double>(t1 - t0) / 1e9,
             static_cast<double>(rss0) / 1048576.0,
             static_cast<double>(rss1) / 1048576.0);

    return List::create(
        _["n_genes"] = res.n_genes,
        _["n_cells"] = res.n_cells
    );
}


// [[Rcpp::export]]
List cpp_vst(std::string hdf5_path, std::string assay_group, int n_top) {
    HDF5File file(hdf5_path, FileMode::ReadWrite);
    auto scheduler = make_scheduler();

    auto mat = file.open_csc_matrix(assay_group + "/layers/counts");
    int64 n_genes = mat->n_rows();
    int64 n_cells = mat->n_cols();

    VSTOperator op(n_top);
    // HDF5-backed read ops need --enable-threadsafe for multi-threading
    int n_threads = get_num_threads();
    VSTResult res = op.run(&file, assay_group + "/layers/counts",
                             n_genes, n_cells, scheduler, n_threads);

    // Write results to HDF5
    file.write_vector_double(assay_group + "/features/mean", res.gene_means);
    file.write_vector_double(assay_group + "/features/variance", res.gene_variances);
    file.write_vector_double(assay_group + "/features/vst_mean", {}); // placeholder
    file.write_vector_double(assay_group + "/features/vst_variance", res.vst_variances);

    std::vector<int32> var_int;
    for (auto v : res.variable_features) var_int.push_back(v ? 1 : 0);
    file.write_vector_int32(assay_group + "/features/variable", var_int);

    return List::create(
        _["n_variable"] = res.n_variable
    );
}


// [[Rcpp::export]]
List cpp_pca(std::string hdf5_path, std::string assay_group,
             int npcs, double tol, int max_iter) {
    int64 t0 = sclean::wall_time_ns();
    int64 rss0 = sclean::current_rss_bytes();

    HDF5File file(hdf5_path, FileMode::ReadWrite);
    auto scheduler = make_scheduler();

    // Determine input data and whether to use on-the-fly centering.
    // Priority: 1) residual_mean/sd (on-the-fly), 2) scale.data (pre-centered), 3) data (raw)
    std::string input_group;
    std::vector<double> means, sds;
    const double* p_means = nullptr;
    const double* p_sds = nullptr;

    std::string mean_path = assay_group + "/features/residual_mean";
    std::string sd_path   = assay_group + "/features/residual_sd";

    if (file.exists(mean_path) && file.exists(sd_path)) {
        means = file.read_vector_double(mean_path);
        sds   = file.read_vector_double(sd_path);
        p_means = means.data();
        p_sds   = sds.data();
        input_group = assay_group + "/layers/data";
    } else if (file.exists(assay_group + "/layers/scale.data")) {
        // Legacy: pre-centered scale.data (no on-the-fly centering needed)
        input_group = assay_group + "/layers/scale.data";
    } else {
        input_group = assay_group + "/layers/data";
    }

    auto mat = file.open_csc_matrix(input_group);

    PCAOperator op(npcs, true, false, tol, max_iter);
    int n_threads = get_num_threads();
    PCAResult res = op.run(mat.get(), scheduler, n_threads, p_means, p_sds);

    // Write PCA results to HDF5
    std::string pca_group = assay_group + "/reductions/pca";
    file.create_group(pca_group);

    // Write loadings as dense matrix
    {
        std::vector<double> ld(res.loadings.rows() * res.loadings.cols());
        Eigen::Map<Eigen::MatrixXd>(ld.data(), res.loadings.rows(),
                                     res.loadings.cols()) = res.loadings;
        file.create_dense_dataset(pca_group + "/loadings",
                                   res.loadings.rows(),
                                   res.loadings.cols())->write_full(ld.data());
    }

    // Write embeddings
    {
        std::vector<double> emb(res.embeddings.rows() * res.embeddings.cols());
        Eigen::Map<Eigen::MatrixXd>(emb.data(), res.embeddings.rows(),
                                     res.embeddings.cols()) = res.embeddings;
        file.create_dense_dataset(pca_group + "/embeddings",
                                   res.embeddings.rows(),
                                   res.embeddings.cols())->write_full(emb.data());
    }

    // Write stdev
    std::vector<double> stdev_vec(res.stdev.data(),
                                   res.stdev.data() + res.stdev.size());
    file.write_vector_double(pca_group + "/stdev", stdev_vec);

    file.set_attr_double(pca_group, "total_variance", res.total_variance);

    int64 t1 = sclean::wall_time_ns();
    int64 rss1 = sclean::current_rss_bytes();
    REprintf("[RunPCA] %d PCs, %d cells | %.2f s | RSS %.0f -> %.0f MB\n",
             static_cast<int>(res.stdev.size()),
             static_cast<int>(res.embeddings.rows()),
             static_cast<double>(t1 - t0) / 1e9,
             static_cast<double>(rss0) / 1048576.0,
             static_cast<double>(rss1) / 1048576.0);

    return List::create(
        _["npcs"] = static_cast<int>(res.stdev.size()),
        _["total_variance"] = res.total_variance
    );
}


// [[Rcpp::export]]
void cpp_find_neighbors(NumericMatrix embeddings, int k,
                        std::string metric, int n_trees,
                        std::string hdf5_path, std::string assay_group) {
    int64 t0 = sclean::wall_time_ns();
    int64 rss0 = sclean::current_rss_bytes();

    HDF5File file(hdf5_path, FileMode::ReadWrite);

    int n_cells = embeddings.nrow();
    int dims = embeddings.ncol();

    // Convert R matrix to Eigen
    Eigen::MatrixXd emb_eigen(n_cells, dims);
    for (int i = 0; i < n_cells; ++i) {
        for (int j = 0; j < dims; ++j) {
            emb_eigen(i, j) = embeddings(i, j);
        }
    }

    NeighborOperator op(k, n_trees);
    ChunkScheduler scheduler;
    NeighborResult knn = op.run(emb_eigen, k, scheduler);

    // Build SNN graph
    std::vector<double> snn_data;
    std::vector<int32> snn_indices;
    std::vector<int64> snn_indptr;
    op.build_snn(knn, snn_data, snn_indices, snn_indptr, 1.0 / 15.0, get_num_threads());

    // Write KNN to HDF5
    std::string nn_group = assay_group + "/graphs/nn";
    file.create_group(nn_group);
    file.write_vector_int32(nn_group + "/indices", knn.nn_idx);
    // Convert float to double for HDF5 write
    std::vector<double> nn_dists_d(knn.nn_dists.begin(), knn.nn_dists.end());
    file.write_vector_double(nn_group + "/distances", nn_dists_d);

    // Write SNN to HDF5 as CSC
    std::string snn_group = assay_group + "/graphs/snn";
    file.create_csc_matrix(snn_group, snn_data, snn_indices, snn_indptr,
                            n_cells, n_cells);

    int64 t1 = sclean::wall_time_ns();
    int64 rss1 = sclean::current_rss_bytes();
    REprintf("[FindNeighbors] %d cells, k=%d | %.2f s | RSS %.0f -> %.0f MB\n",
             n_cells, k,
             static_cast<double>(t1 - t0) / 1e9,
             static_cast<double>(rss0) / 1048576.0,
             static_cast<double>(rss1) / 1048576.0);
}


// [[Rcpp::export]]
List cpp_find_clusters(std::string hdf5_path, std::string assay_group,
                       int algorithm, double resolution, int n_iter) {
    int64 t0 = sclean::wall_time_ns();
    int64 rss0 = sclean::current_rss_bytes();

    HDF5File file(hdf5_path, FileMode::ReadWrite);

    std::string snn_group = assay_group + "/graphs/snn";
    if (!file.exists(snn_group)) {
        stop("SNN graph not found. Run FindNeighbors first.");
    }

    auto snn = file.open_csc_matrix(snn_group);
    int n_cells = static_cast<int>(snn->n_cols());

    std::string algo = (algorithm == 1) ? "leiden" : "louvain";

    ClusterOperator op(algo, resolution, n_iter);
    ChunkScheduler scheduler;
    ClusterResult res = op.run_from_hdf5(&file, snn_group, n_cells, scheduler);

    // Write cluster assignments
    std::string algo_name = (algorithm == 1) ? "leiden" : "louvain";
    std::string cluster_path = assay_group + "/clusters/" + algo_name;
    file.write_vector_int32(cluster_path, res.assignments);

    int64 t1 = sclean::wall_time_ns();
    int64 rss1 = sclean::current_rss_bytes();
    REprintf("[FindClusters] %d cells, %d clusters, Q=%.4f | %.2f s | RSS %.0f -> %.0f MB\n",
             n_cells, res.n_clusters, res.modularity,
             static_cast<double>(t1 - t0) / 1e9,
             static_cast<double>(rss0) / 1048576.0,
             static_cast<double>(rss1) / 1048576.0);

    return List::create(
        _["n_clusters"] = res.n_clusters,
        _["modularity"] = res.modularity,
        _["assignments"] = std::vector<int>(res.assignments.begin(),
                                             res.assignments.end())
    );
}


// [[Rcpp::export]]
List cpp_find_markers(std::string hdf5_path, std::string assay_group,
                      std::vector<int> clusters, int ident_1, int ident_2,
                      int test_type, double logfc_threshold, double min_pct) {
    HDF5File file(hdf5_path, FileMode::ReadOnly);

    std::string data_group = assay_group + "/layers/data";
    auto mat = file.open_csc_matrix(data_group);
    int64 n_genes = mat->n_rows();
    int64 n_cells = mat->n_cols();

    std::vector<int32> cl_int(clusters.begin(), clusters.end());

    MarkerOperator op(static_cast<DETest>(test_type),
                       logfc_threshold, min_pct);

    ChunkScheduler scheduler;
    // HDF5-backed read ops need --enable-threadsafe for multi-threading
    int n_threads = get_num_threads();
    auto results = op.find_markers(&file, data_group, n_genes, n_cells,
                                    cl_int, static_cast<int32>(ident_1),
                                    static_cast<int32>(ident_2),
                                    scheduler, n_threads);

    // Convert to R list of data frames
    List out(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        auto& r = results[i];
        out[i] = List::create(
            _["gene_idx"] = static_cast<int>(r.gene_idx),
            _["p_val"] = r.p_val,
            _["avg_log2FC"] = r.avg_log2FC,
            _["pct.1"] = r.pct_1,
            _["pct.2"] = r.pct_2,
            _["p_val_adj"] = r.p_val_adj
        );
    }

    return out;
}


// [[Rcpp::export]]
void cpp_integrate(std::string hdf5_path, std::string assay_group,
                   std::vector<int> batch_labels, int n_ccs,
                   int n_mnn = 20, double sigma = 0.1, int max_iter = 2) {
    int64 t0 = sclean::wall_time_ns();
    int64 rss0 = sclean::current_rss_bytes();

    HDF5File file(hdf5_path, FileMode::ReadWrite);

    // Read PCA embeddings from HDF5
    std::string pca_group = assay_group + "/reductions/pca";
    std::string emb_path = pca_group + "/embeddings";

    if (!file.exists(emb_path)) {
        stop("PCA embeddings not found. Run RunPCA first.");
    }

    auto emb_mat = file.open_dense_matrix(emb_path);
    std::vector<double> emb_data(emb_mat->n_rows() * emb_mat->n_cols());
    emb_mat->read_full(emb_data);

    int n_cells = static_cast<int>(emb_mat->n_rows());
    int n_pcs = static_cast<int>(emb_mat->n_cols());

    Eigen::Map<Eigen::MatrixXd> embeddings(emb_data.data(), n_cells, n_pcs);

    std::vector<int32> bl(batch_labels.begin(), batch_labels.end());

    auto scheduler = make_scheduler();
    IntegrationOperator op(n_ccs, n_mnn, sigma, max_iter);
    IntegrationResult res = op.run(embeddings, bl, scheduler);

    // Write corrected embeddings to HDF5
    std::string harmony_group = assay_group + "/reductions/harmony";
    file.create_group(harmony_group);

    std::vector<double> corr_data(
        res.corrected_embeddings.rows() * res.corrected_embeddings.cols());
    Eigen::Map<Eigen::MatrixXd>(corr_data.data(),
                                 res.corrected_embeddings.rows(),
                                 res.corrected_embeddings.cols())
        = res.corrected_embeddings;

    file.create_dense_dataset(harmony_group + "/embeddings",
                               res.corrected_embeddings.rows(),
                               res.corrected_embeddings.cols())
        ->write_full(corr_data.data());

    int64 t1 = sclean::wall_time_ns();
    int64 rss1 = sclean::current_rss_bytes();
    REprintf("[IntegrateLayers] %d cells, %d batches, %d dims | "
             "%.2f s | RSS %.0f -> %.0f MB\n",
             n_cells, res.n_batches, std::min(n_ccs, n_pcs),
             static_cast<double>(t1 - t0) / 1e9,
             static_cast<double>(rss0) / 1048576.0,
             static_cast<double>(rss1) / 1048576.0);
}

// ============================================================
// Performance probes
// ============================================================

// [[Rcpp::export]]
double cpp_current_rss() {
    return static_cast<double>(sclean::current_rss_bytes());
}

// [[Rcpp::export]]
double cpp_wall_time_ns() {
    return static_cast<double>(sclean::wall_time_ns());
}

// [[Rcpp::export]]
void suppress_hdf5_errors() {
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
}

// [[Rcpp::export]]
void cpp_set_threads(int n) {
    set_num_threads(n);
}

// [[Rcpp::export]]
Rcpp::List cpp_resource_snapshot() {
    ResourceMonitor monitor;
    auto snap = monitor.snapshot();
    return Rcpp::List::create(
        Rcpp::Named("total_ram")        = static_cast<double>(snap.total_ram),
        Rcpp::Named("free_ram")         = static_cast<double>(snap.free_ram),
        Rcpp::Named("current_rss")      = static_cast<double>(snap.current_rss),
        Rcpp::Named("available_ram")    = static_cast<double>(snap.available_ram),
        Rcpp::Named("cpu_load_1min")    = snap.cpu_load_1min,
        Rcpp::Named("cpu_load_5min")    = snap.cpu_load_5min,
        Rcpp::Named("physical_cores")   = snap.physical_cores,
        Rcpp::Named("available_cores")  = snap.available_cores,
        Rcpp::Named("memory_pressure")  = snap.memory_pressure,
        Rcpp::Named("cpu_pressure")     = snap.cpu_pressure
    );
}

// [[Rcpp::export]]
std::string cpp_bottleneck_type() {
    ResourceMonitor monitor;
    auto snap = monitor.snapshot();
    Bottleneck b = ResourceMonitor::classify(snap);
    switch (b) {
        case Bottleneck::None:         return "none";
        case Bottleneck::MemoryBound:  return "memory";
        case Bottleneck::ComputeBound: return "compute";
        case Bottleneck::BothBound:    return "both";
    }
    return "none";
}

// [[Rcpp::export]]
void cpp_set_max_dense_chunk_mb(double mb) {
    set_max_dense_chunk_bytes(static_cast<int64>(mb * 1024.0 * 1024.0));
}

// [[Rcpp::export]]
double cpp_get_max_dense_chunk_mb() {
    return static_cast<double>(get_max_dense_chunk_bytes()) / (1024.0 * 1024.0);
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
