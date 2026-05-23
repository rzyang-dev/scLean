#include <Rcpp.h>
#include <RcppEigen.h>
#include <string>
#include <vector>
#include <unordered_set>
#include <unistd.h>

#include "hdf5/hdf5_file.h"
#include "hdf5/hdf5_csc_matrix.h"
#include "hdf5/hdf5_dense_matrix.h"
#include "normalize/normalize_operator.h"
#include "scale/scale_operator.h"
#include "normalize/vst_operator.h"
#include "pca/pca_operator.h"
#include "neighbors/neighbor_operator.h"
#include "neighbors/cluster_operator.h"
#include "markers/marker_operator.h"
#include "integration/integration_operator.h"
#include "utils/memory_probe.h"
#include "utils/chunk_scheduler.h"
#include "utils/parallel.h"

using namespace Rcpp;
using namespace sclean;

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
             int npcs, double tol, int max_iter,
             Rcpp::Nullable<Rcpp::IntegerVector> feature_indices = R_NilValue) {
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
    int64 n_genes = mat->n_rows();

    PCAOperator op(npcs, true, false, tol, max_iter);
    int n_threads = get_num_threads();
    PCAResult res;
    std::vector<int> feat_idx;
    if (feature_indices.isNotNull()) {
        feat_idx = Rcpp::as<std::vector<int>>(feature_indices);
    }
    if (!feat_idx.empty()) {
        std::vector<int64> indices64(feat_idx.begin(), feat_idx.end());
        res = op.run_on_subset(mat.get(), indices64, scheduler, n_threads);
        if (res.loadings.rows() < n_genes) {
            Eigen::MatrixXd full_loadings = Eigen::MatrixXd::Zero(n_genes, res.loadings.cols());
            for (int64 i = 0; i < static_cast<int64>(indices64.size()); ++i) {
                full_loadings.row(indices64[i]) = res.loadings.row(i);
            }
            res.loadings = full_loadings;
        }
    } else {
        res = op.run(mat.get(), scheduler, n_threads, p_means, p_sds);
    }

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
