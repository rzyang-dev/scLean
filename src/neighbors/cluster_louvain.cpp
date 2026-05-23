#include "cluster_internal.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>

namespace sclean {

// ============================================================
// Modularity computation (Newman-Girvan)
// ============================================================

double compute_graph_modularity(
    const std::vector<double>& graph_data,
    const std::vector<int32>& graph_indices,
    const std::vector<int64>& graph_indptr,
    int n_nodes,
    const std::vector<int32>& communities,
    double resolution) {

    // Total edge weight (m)
    double m = 0.0;
    for (auto w : graph_data) m += w;
    if (m == 0.0) return 0.0;

    // Degree per node
    std::vector<double> degree(n_nodes, 0.0);
    for (int i = 0; i < n_nodes; ++i) {
        for (int64 j = graph_indptr[i]; j < graph_indptr[i + 1]; ++j) {
            degree[i] += graph_data[j];
        }
    }

    // Community stats: sigma_in (internal weight) and sigma_tot (total degree)
    int n_comm = *std::max_element(communities.begin(), communities.end()) + 1;
    std::vector<double> sigma_in(n_comm, 0.0);
    std::vector<double> sigma_tot(n_comm, 0.0);

    for (int i = 0; i < n_nodes; ++i) {
        int ci = communities[i];
        sigma_tot[ci] += degree[i];
        for (int64 j = graph_indptr[i]; j < graph_indptr[i + 1]; ++j) {
            int nb = graph_indices[j];
            if (communities[nb] == ci) {
                sigma_in[ci] += graph_data[j];
            }
        }
    }

    // Each internal edge counted twice
    double Q = 0.0;
    for (int c = 0; c < n_comm; ++c) {
        Q += sigma_in[c] / (2.0 * m) -
             resolution * (sigma_tot[c] / (2.0 * m)) * (sigma_tot[c] / (2.0 * m));
    }

    return Q;
}

// ============================================================
// Louvain one-pass (local moving)
// ============================================================

bool louvain_local_moving_pass(
    const std::vector<double>& graph_data,
    const std::vector<int32>& graph_indices,
    const std::vector<int64>& graph_indptr,
    int n_nodes,
    std::vector<int32>& communities,
    double resolution,
    int random_seed) {

    // Total edge weight
    double m = 0.0;
    for (auto w : graph_data) m += w;
    if (m == 0.0) return false;

    // Degree per node
    std::vector<double> degree(n_nodes, 0.0);
    for (int i = 0; i < n_nodes; ++i) {
        for (int64 j = graph_indptr[i]; j < graph_indptr[i + 1]; ++j) {
            degree[i] += graph_data[j];
        }
    }

    // Community stats
    int max_comm = n_nodes;
    std::vector<double> sigma_tot(max_comm, 0.0);
    for (int i = 0; i < n_nodes; ++i) {
        sigma_tot[communities[i]] += degree[i];
    }

    // Random node ordering
    std::vector<int> order(n_nodes);
    for (int i = 0; i < n_nodes; ++i) order[i] = i;
    std::mt19937 rng(random_seed);
    std::shuffle(order.begin(), order.end(), rng);

    bool moved = false;
    int n_passes = 0;
    bool changed = true;

    while (changed && n_passes < 10) {
        changed = false;
        n_passes++;

        for (int idx : order) {
            int cur_comm = communities[idx];

            // Compute k_i,in per neighboring community.
            // Use sorted vector of pairs instead of unordered_map --
            // per-node neighbor communities are few (< 64 in practice),
            // so linear scan on a vector is faster than hash table overhead.
            std::vector<std::pair<int32, double>> k_in;
            double k_i = degree[idx];

            for (int64 j = graph_indptr[idx]; j < graph_indptr[idx + 1]; ++j) {
                int nb_comm = communities[graph_indices[j]];
                double w = graph_data[j];

                // Linear scan to find existing entry for this community
                bool found = false;
                for (auto& kv : k_in) {
                    if (kv.first == nb_comm) {
                        kv.second += w;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    k_in.emplace_back(nb_comm, w);
                }
            }

            // Remove node from current community
            sigma_tot[cur_comm] -= k_i;

            // Find best community
            int best_comm = cur_comm;
            double best_delta = 0.0;

            // Look up ki_in for the current community (used in delta calc)
            double ki_in_cur = 0.0;
            int cur_comm_found = -1;
            for (size_t p = 0; p < k_in.size(); ++p) {
                if (k_in[p].first == cur_comm) {
                    ki_in_cur = k_in[p].second;
                    cur_comm_found = static_cast<int>(p);
                    break;
                }
            }

            for (auto& kv : k_in) {
                int target = kv.first;
                if (target == cur_comm) continue;

                double ki_in = kv.second;
                double sigma_tot_target = sigma_tot[target];

                // delta-Q = ki_in/m - resolution * sigma_tot_target * ki / (2*m^2)
                double delta = ki_in / m -
                    resolution * sigma_tot_target * k_i / (2.0 * m * m);

                // Also remove current community contribution
                if (cur_comm_found >= 0) {
                    double sigma_tot_cur = sigma_tot[cur_comm];
                    delta -= ki_in_cur / m -
                        resolution * sigma_tot_cur * k_i / (2.0 * m * m);
                } else {
                    delta -= resolution * k_i * k_i / (4.0 * m * m);
                }

                if (delta > best_delta) {
                    best_delta = delta;
                    best_comm = target;
                }
            }

            if (best_comm != cur_comm) {
                communities[idx] = static_cast<int32>(best_comm);
                sigma_tot[best_comm] += k_i;
                moved = true;
                changed = true;
            } else {
                sigma_tot[cur_comm] += k_i;
            }
        }
    }

    return moved;
}

// ============================================================
// Louvain full algorithm
// ============================================================

std::vector<int32> louvain_cluster(
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
        bool moved = louvain_local_moving_pass(graph_data, graph_indices, graph_indptr,
                                                n_nodes, communities, resolution,
                                                random_seed);
        if (!moved) break;

        // Re-number communities to be contiguous
        std::unordered_map<int32, int32> comm_map;
        int next_id = 0;
        for (int i = 0; i < n_nodes; ++i) {
            int32 c = communities[i];
            if (!comm_map.count(c)) {
                comm_map[c] = static_cast<int32>(next_id++);
            }
            communities[i] = comm_map[c];
        }
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
