#ifndef SCLEAN_CLUSTER_OPERATOR_H
#define SCLEAN_CLUSTER_OPERATOR_H

#include <cstdint>
#include <vector>
#include <string>
#include "../scLean_types.h"
#include "../utils/chunk_scheduler.h"

namespace sclean {

class HDF5File;

struct ClusterResult {
    std::vector<int32> assignments;  // (n_cells,)
    int n_clusters;
    double modularity;
};

class ClusterOperator {
public:
    ClusterOperator(const std::string& algorithm = "leiden",
                    double resolution = 0.8,
                    int n_iterations = 10,
                    int random_seed = 42);

    // Run clustering on a graph given in CSC format
    ClusterResult run(const std::vector<double>& graph_data,
                      const std::vector<int32>& graph_indices,
                      const std::vector<int64>& graph_indptr,
                      int n_nodes);

    // Alternative: load SNN graph from HDF5 and run.
    // Scheduler provides memory checks for large graphs.
    ClusterResult run_from_hdf5(HDF5File* file,
                                 const std::string& graph_group,
                                 int n_cells,
                                 ChunkScheduler& scheduler);

private:
    std::string algorithm_;
    double resolution_;
    int n_iterations_;
    int random_seed_;
};

} // namespace sclean

#endif // SCLEAN_CLUSTER_OPERATOR_H
