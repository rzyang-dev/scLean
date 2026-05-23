#ifndef SCLEAN_NEIGHBOR_OPERATOR_H
#define SCLEAN_NEIGHBOR_OPERATOR_H

#include <cstdint>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include "../scLean_types.h"
#include "../utils/chunk_scheduler.h"

namespace sclean {

class HDF5File;

struct NeighborResult {
    std::vector<int32> nn_idx;      // (n_cells * k,) flattened, row-major
    std::vector<float> nn_dists;    // (n_cells * k,)
    int k;
    int n_cells;
};

class NeighborOperator {
public:
    NeighborOperator(int k = 20, int n_trees = 50);

    // Find KNN on PCA embeddings (n_cells x dims dense matrix from HDF5).
    // Uses Annoy approximate NN for large datasets, falls back to block-based
    // brute force when memory-bound or n_trees == 0.
    // Scheduler provides block-size guidance and bottleneck detection.
    NeighborResult run(const Eigen::MatrixXd& embeddings, int k,
                       ChunkScheduler& scheduler);

    // Build SNN graph from KNN graph using Jaccard similarity.
    // Delegates to the free function build_snn_graph() in neighbor_snn.cpp.
    // Returns sparse CSC representation via output parameters.
    void build_snn(const NeighborResult& knn,
                   std::vector<double>& snn_data,
                   std::vector<int32>& snn_indices,
                   std::vector<int64>& snn_indptr,
                   double prune = 1.0 / 15.0,
                   int n_threads = 1);

private:
    int n_trees_;
};

} // namespace sclean

#endif // SCLEAN_NEIGHBOR_OPERATOR_H
