#include "cluster_operator.h"
#include "cluster_internal.h"
#include "hdf5/hdf5_file.h"
#include "hdf5/hdf5_csc_matrix.h"
#include "utils/progress.h"
#include "utils/resource_monitor.h"
#include <algorithm>
#include <stdexcept>

namespace sclean {

ClusterOperator::ClusterOperator(const std::string& algorithm,
                                   double resolution,
                                   int n_iterations,
                                   int random_seed)
    : algorithm_(algorithm), resolution_(resolution),
      n_iterations_(n_iterations), random_seed_(random_seed) {}

// ============================================================
// Public run: delegates to pure algorithm functions
// ============================================================

ClusterResult ClusterOperator::run(
    const std::vector<double>& graph_data,
    const std::vector<int32>& graph_indices,
    const std::vector<int64>& graph_indptr,
    int n_nodes) {

    std::vector<int32> assignments;
    ProgressReporter progress("FindClusters", 1,
                              ProgressReporter::is_verbose());

    if (algorithm_ == "leiden") {
        progress.message("Running Leiden clustering...");
        assignments = leiden_cluster(graph_data, graph_indices, graph_indptr,
                                       n_nodes, resolution_, n_iterations_,
                                       random_seed_);
    } else if (algorithm_ == "louvain") {
        progress.message("Running Louvain clustering...");
        assignments = louvain_cluster(graph_data, graph_indices, graph_indptr,
                                        n_nodes, resolution_, n_iterations_,
                                        random_seed_);
    } else {
        throw std::runtime_error("Unknown clustering algorithm: " + algorithm_);
    }

    progress.step();
    progress.done();

    ClusterResult result;
    result.assignments = assignments;
    result.n_clusters = assignments.empty() ? 0 :
        (*std::max_element(assignments.begin(), assignments.end()) + 1);
    result.modularity = compute_graph_modularity(graph_data, graph_indices,
                                                  graph_indptr, n_nodes,
                                                  assignments, resolution_);
    return result;
}

// ============================================================
// run_from_hdf5: load graph from HDF5, delegate to run()
// ============================================================

ClusterResult ClusterOperator::run_from_hdf5(
    HDF5File* file, const std::string& graph_group, int n_cells,
    ChunkScheduler& scheduler) {

    // Read SNN graph from HDF5
    auto snn = file->open_csc_matrix(graph_group);
    int64 nnz = snn->nnz();

    // Memory estimate: data (8B) + indices (4B) + indptr (8B)
    int64 graph_mem_est = nnz * (sizeof(double) + sizeof(int32))
                          + (n_cells + 1) * sizeof(int64);

    scheduler.refresh_available_ram();
    ResourceMonitor monitor;
    auto snap = monitor.snapshot();
    int64 available = ChunkScheduler::worst_case_available_ram(
        snap.free_ram, snap.current_rss, 1);

    if (graph_mem_est > available && n_cells > 100000) {
        REprintf("[scLean] FindClusters: graph memory estimate %.0f MB "
                 "exceeds available %.0f MB -- clustering may be slow or OOM\n",
                 graph_mem_est / 1048576.0, available / 1048576.0);
    }

    // Read all graph data
    std::vector<double> graph_data(nnz);
    std::vector<int32> graph_indices(nnz);

    hid_t fid = file->file_id();
    std::string data_path = graph_group + "/data";
    std::string idx_path = graph_group + "/indices";
    std::string indptr_path = graph_group + "/indptr";

    // Read data
    {
        hid_t d = H5Dopen2(fid, data_path.c_str(), H5P_DEFAULT);
        H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, graph_data.data());
        H5Dclose(d);
    }
    {
        hid_t d = H5Dopen2(fid, idx_path.c_str(), H5P_DEFAULT);
        H5Dread(d, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, graph_indices.data());
        H5Dclose(d);
    }

    // Read indptr
    std::vector<int64> graph_indptr(n_cells + 1);
    {
        hid_t d = H5Dopen2(fid, indptr_path.c_str(), H5P_DEFAULT);
        H5Dread(d, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, graph_indptr.data());
        H5Dclose(d);
    }

    return run(graph_data, graph_indices, graph_indptr, n_cells);
}

} // namespace sclean
