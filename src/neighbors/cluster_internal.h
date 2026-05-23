#ifndef SCLEAN_CLUSTER_INTERNAL_H
#define SCLEAN_CLUSTER_INTERNAL_H

#include <cstdint>
#include <vector>
#include "../scLean_types.h"

namespace sclean {

// ============================================================
// Pure functions for modularity optimization and community
// detection. These operate on in-memory CSC graph data and
// have no side effects (no HDF5 I/O, no console output).
// ============================================================

// Compute modularity of a graph partition (Newman-Girvan Q).
double compute_graph_modularity(
    const std::vector<double>& graph_data,
    const std::vector<int32>& graph_indices,
    const std::vector<int64>& graph_indptr,
    int n_nodes,
    const std::vector<int32>& communities,
    double resolution);

// One pass of Louvain local moving optimization.
// Moves nodes between communities to maximize modularity.
// Returns true if any node moved.
bool louvain_local_moving_pass(
    const std::vector<double>& graph_data,
    const std::vector<int32>& graph_indices,
    const std::vector<int64>& graph_indptr,
    int n_nodes,
    std::vector<int32>& communities,  // in/out: community assignments
    double resolution,
    int random_seed);

// Full Louvain modularity optimization.
// Returns community assignments (0-based, sorted by size descending).
std::vector<int32> louvain_cluster(
    const std::vector<double>& graph_data,
    const std::vector<int32>& graph_indices,
    const std::vector<int64>& graph_indptr,
    int n_nodes,
    double resolution,
    int n_iterations,
    int random_seed);

// Leiden refinement phase: split communities into internally
// well-connected sub-communities (Traag, Waltman & van Eck, 2019).
// Takes partitions from a Louvain-style local-moving pass and
// guarantees that each resulting community is internally connected.
std::vector<int32> leiden_refine(
    const std::vector<double>& graph_data,
    const std::vector<int32>& graph_indices,
    const std::vector<int64>& graph_indptr,
    int n_nodes,
    const std::vector<int32>& communities);

// Full Leiden clustering (local moving + refinement).
// Returns community assignments (0-based, sorted by size descending).
std::vector<int32> leiden_cluster(
    const std::vector<double>& graph_data,
    const std::vector<int32>& graph_indices,
    const std::vector<int64>& graph_indptr,
    int n_nodes,
    double resolution,
    int n_iterations,
    int random_seed);

} // namespace sclean

#endif // SCLEAN_CLUSTER_INTERNAL_H
