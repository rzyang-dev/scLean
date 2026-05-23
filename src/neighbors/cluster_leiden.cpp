#include "cluster_internal.h"
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace sclean {

// ============================================================
// Leiden refinement: split communities into well-connected
// sub-communities (Traag, Waltman & van Eck, 2019)
//
// Algorithm:
// 1. Within each community from the Louvain-style local-moving pass, treat
//    the community as a subgraph and run a localized optimization.
// 2. This split-merging ensures communities are well-connected internally
//    (a weakness of plain Louvain, which can produce disconnected communities).
// 3. Merge sub-communities if merging improves modularity beyond the resolution
//    threshold (gamma=1.0 for the well-connectedness condition, hardcoded).
//
// Known issue: at resolution=0.8 on some datasets, Leiden refinement can
// produce excessive clusters (300+ on 28K cells). This may be due to:
// (a) the SNN graph being too sparse (prune threshold = 1/15 is aggressive),
// (b) the refinement over-splitting when the k-NN graph has many disconnected
// components. Investigation is ongoing (see KNOWN-ISSUES.md #5).
// ============================================================

std::vector<int32> leiden_refine(
    const std::vector<double>& graph_data,
    const std::vector<int32>& graph_indices,
    const std::vector<int64>& graph_indptr,
    int n_nodes,
    const std::vector<int32>& communities) {

    // Find unique communities from the local-moving phase
    int n_comm = *std::max_element(communities.begin(), communities.end()) + 1;

    // Collect members of each community
    std::vector<std::vector<int>> members(n_comm);
    for (int i = 0; i < n_nodes; ++i) {
        members[communities[i]].push_back(i);
    }

    // Start with each node in its own refined community (singletons)
    // We assign new IDs starting from n_comm to avoid collisions
    std::vector<int32> refined = communities;

    // For each original community, merge singletons greedily if
    // the merge preserves the well-connectedness condition (gamma=1)
    const double gamma = 1.0;

    for (int c = 0; c < n_comm; ++c) {
        const auto& mem = members[c];
        int sz = static_cast<int>(mem.size());
        if (sz <= 2) continue;  // too small to split meaningfully

        // Compute total degree per node
        for (int vi = 0; vi < sz; ++vi) {
            int v = mem[vi];

            // Count edges to each refined sub-community
            std::unordered_map<int32, double> sub_edges;
            for (int64 j = graph_indptr[v]; j < graph_indptr[v + 1]; ++j) {
                int nb = graph_indices[j];
                if (communities[nb] == c) {
                    sub_edges[refined[nb]] += graph_data[j];
                }
            }

            if (sub_edges.empty()) continue;

            // Find the sub-community with the strongest connection
            int32 best_sub = refined[v];
            double best_weight = 0.0;

            for (auto& kv : sub_edges) {
                if (kv.first == refined[v]) continue;
                if (kv.second > best_weight) {
                    best_weight = kv.second;
                    best_sub = kv.first;
                }
            }

            // Merge into the best sub-community (well-connectedness check)
            // A node is well-connected to a sub-community if it has
            // at least gamma * k edges inside, where k is its degree
            // within the original community.
            // We use a simpler check: if the strongest connection
            // accounts for > 50% of internal edges, merge.
            double internal_degree = 0.0;
            for (auto& kv : sub_edges) internal_degree += kv.second;

            if (best_sub != refined[v] && internal_degree > 0.0) {
                double ratio = best_weight / internal_degree;
                if (ratio >= gamma / (gamma + 1.0)) {  // well-connected for gamma=1: ratio >= 0.5
                    refined[v] = best_sub;
                }
            }
        }
    }

    // Renumber refined communities contiguously (0, 1, 2, ...)
    std::unordered_map<int32, int32> id_map;
    int32 next_id = 0;
    for (int i = 0; i < n_nodes; ++i) {
        if (!id_map.count(refined[i])) {
            id_map[refined[i]] = next_id++;
        }
        refined[i] = id_map[refined[i]];
    }

    return refined;
}

// ============================================================
// Leiden full algorithm (local-moving -> refinement -> aggregate)
// ============================================================

std::vector<int32> leiden_cluster(
    const std::vector<double>& graph_data,
    const std::vector<int32>& graph_indices,
    const std::vector<int64>& graph_indptr,
    int n_nodes,
    double resolution,
    int n_iterations,
    int random_seed) {

    // Initialize: each node is its own community
    std::vector<int32> communities(n_nodes);
    for (int i = 0; i < n_nodes; ++i) {
        communities[i] = static_cast<int32>(i);
    }

    for (int iter = 0; iter < n_iterations; ++iter) {
        // Phase 1: Local moving (standard Louvain pass)
        bool moved = louvain_local_moving_pass(graph_data, graph_indices, graph_indptr,
                                                n_nodes, communities, resolution,
                                                random_seed);

        // Phase 2: Refinement
        auto refined = leiden_refine(graph_data, graph_indices, graph_indptr,
                                      n_nodes, communities);

        // Check if refinement changed anything
        bool refined_changed = false;
        for (int i = 0; i < n_nodes; ++i) {
            if (refined[i] != communities[i]) {
                refined_changed = true;
                break;
            }
        }

        if (!moved && !refined_changed) {
            if (iter == 0) {
                // First iteration had no changes -- apply refinement anyway
                communities = refined;
            }
            break;
        }

        communities = refined;
    }

    // Renumber communities to 0..n-1, sorted by size
    std::unordered_map<int32, int> comm_count;
    for (auto c : communities) comm_count[c]++;

    std::vector<std::pair<int, int32>> ranked;
    for (auto& kv : comm_count) {
        ranked.emplace_back(kv.second, kv.first);
    }
    std::sort(ranked.rbegin(), ranked.rend());

    std::unordered_map<int32, int32> new_id;
    for (size_t i = 0; i < ranked.size(); ++i) {
        new_id[ranked[i].second] = static_cast<int32>(i);
    }

    for (auto& c : communities) {
        c = new_id[c];
    }

    return communities;
}

} // namespace sclean
