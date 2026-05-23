#ifndef SCLEAN_NEIGHBOR_INTERNAL_H
#define SCLEAN_NEIGHBOR_INTERNAL_H

#include <cstdint>
#include <vector>
#include <Eigen/Dense>
#include "scLean_types.h"

namespace sclean {

// Block-based brute-force KNN using Eigen.
// Splits cells into query blocks to limit memory; uses distance decomposition
// ||x_q - x_r||^2 = ||x_q||^2 + ||x_r||^2 - 2 * x_q^T * x_r.
// Returns k nearest neighbors per cell (excluding self), stored row-major.
void knn_blocked(const Eigen::MatrixXd& data,
                 int k, int block_size,
                 std::vector<int32>& idx,
                 std::vector<float>& dists);

// Annoy-based approximate KNN for large datasets.
// Builds an Annoy index with Euclidean distance, queries k+1 neighbors
// (to exclude self), and returns k nearest neighbors per cell.
void knn_annoy(const Eigen::MatrixXd& data,
               int k, int n_trees,
               std::vector<int32>& idx,
               std::vector<float>& dists);

// Build SNN graph from KNN graph using Jaccard similarity.
// J(i,j) = |NN(i) intersect NN(j)| / |NN(i) union NN(j)|
//        = intersect / (2k - intersect)
// Returns sparse CSC representation: data (weights), indices (column indices),
// and indptr (column pointers). Only edges with Jaccard > prune are retained.
void build_snn_graph(const std::vector<int32>& nn_idx,
                     const std::vector<float>& nn_dists,
                     int k, int n_cells, double prune, int n_threads,
                     std::vector<double>& snn_data,
                     std::vector<int32>& snn_indices,
                     std::vector<int64>& snn_indptr);

} // namespace sclean

#endif // SCLEAN_NEIGHBOR_INTERNAL_H
