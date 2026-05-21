#include "cluster_operator.h"
#include "hdf5/hdf5_file.h"
#include "hdf5/hdf5_csc_matrix.h"
#include "utils/progress.h"
#include "utils/resource_monitor.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sclean {

ClusterOperator::ClusterOperator(const std::string& algorithm,
                                   double resolution,
                                   int n_iterations,
                                   int random_seed)
    : algorithm_(algorithm), resolution_(resolution),
      n_iterations_(n_iterations), random_seed_(random_seed) {}

// ============================================================
// Modularity computation
// ============================================================

double ClusterOperator::compute_modularity(
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

    // Community stats: Σ_in (internal weight) and Σ_tot (total degree)
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
// Louvain one-pass
// ============================================================

bool ClusterOperator::louvain_pass(
    const std::vector<double>& graph_data,
    const std::vector<int32>& graph_indices,
    const std::vector<int64>& graph_indptr,
    int n_nodes,
    std::vector<int32>& communities,
    double resolution) {

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
    std::mt19937 rng(random_seed_);
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
            // Use sorted vector of pairs instead of unordered_map —
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

                // ΔQ = ki_in/m - resolution * sigma_tot_target * ki / (2*m^2)
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

ClusterResult ClusterOperator::louvain(
    const std::vector<double>& graph_data,
    const std::vector<int32>& graph_indices,
    const std::vector<int64>& graph_indptr,
    int n_nodes) {

    // Initialize: each node is its own community
    std::vector<int32> communities(n_nodes);
    for (int i = 0; i < n_nodes; ++i) {
        communities[i] = static_cast<int32>(i);
    }

    ProgressReporter progress("FindClusters", n_iterations_,
                              ProgressReporter::is_verbose());
    for (int iter = 0; iter < n_iterations_; ++iter) {
        progress.message("Louvain pass " + std::to_string(iter + 1) + "...");
        bool moved = louvain_pass(graph_data, graph_indices, graph_indptr,
                                   n_nodes, communities, resolution_);
        progress.step();
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

    progress.done();

    ClusterResult result;
    result.assignments = communities;
    result.n_clusters = static_cast<int>(ranked.size());
    result.modularity = compute_modularity(graph_data, graph_indices, graph_indptr,
                                            n_nodes, communities, resolution_);
    return result;
}

// ============================================================
// Leiden refinement: split communities into well-connected
// sub-communities (Traag, Waltman & van Eck, 2019)
// ============================================================

std::vector<int32> ClusterOperator::leiden_refine(
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
// Leiden full algorithm (local-moving → refinement → aggregate)
// ============================================================

ClusterResult ClusterOperator::leiden(
    const std::vector<double>& graph_data,
    const std::vector<int32>& graph_indices,
    const std::vector<int64>& graph_indptr,
    int n_nodes) {

    // Initialize: each node is its own community
    std::vector<int32> communities(n_nodes);
    for (int i = 0; i < n_nodes; ++i) {
        communities[i] = static_cast<int32>(i);
    }

    ProgressReporter progress("FindClusters", n_iterations_,
                              ProgressReporter::is_verbose());
    for (int iter = 0; iter < n_iterations_; ++iter) {
        progress.message("Leiden iteration " + std::to_string(iter + 1) + "/" +
                         std::to_string(n_iterations_) + "...");

        // Phase 1: Local moving (standard Louvain pass)
        progress.message("  local moving...");
        bool moved = louvain_pass(graph_data, graph_indices, graph_indptr,
                                   n_nodes, communities, resolution_);

        // Phase 2: Refinement
        progress.message("  refinement...");
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
                // First iteration had no changes — apply refinement anyway
                communities = refined;
            }
            break;
        }

        communities = refined;
        progress.step();
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

    progress.done();

    ClusterResult result;
    result.assignments = communities;
    result.n_clusters = static_cast<int>(ranked.size());
    result.modularity = compute_modularity(graph_data, graph_indices, graph_indptr,
                                            n_nodes, communities, resolution_);
    return result;
}

// ============================================================
// Public run methods
// ============================================================

ClusterResult ClusterOperator::run(
    const std::vector<double>& graph_data,
    const std::vector<int32>& graph_indices,
    const std::vector<int64>& graph_indptr,
    int n_nodes) {

    if (algorithm_ == "leiden") {
        return leiden(graph_data, graph_indices, graph_indptr, n_nodes);
    } else if (algorithm_ == "louvain") {
        return louvain(graph_data, graph_indices, graph_indptr, n_nodes);
    }

    throw std::runtime_error("Unknown clustering algorithm: " + algorithm_);
}

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
                 "exceeds available %.0f MB — clustering may be slow or OOM\n",
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
