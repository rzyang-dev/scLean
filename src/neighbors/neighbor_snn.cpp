#include "neighbor_internal.h"
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sclean {

// ============================================================
// SNN Graph Construction (Jaccard similarity)
// ============================================================

void build_snn_graph(
    const std::vector<int32>& nn_idx,
    const std::vector<float>& nn_dists,
    int k, int n_cells, double prune, int n_threads,
    std::vector<double>& snn_data,
    std::vector<int32>& snn_indices,
    std::vector<int64>& snn_indptr) {

    // Build reverse index: for each cell, which cells have it as a neighbor
    std::vector<std::vector<int32>> reverse_nn(n_cells);
    for (int i = 0; i < n_cells; ++i) {
        int base = i * k;
        for (int j = 0; j < k; ++j) {
            int nb = nn_idx[base + j];
            if (nb != i) {
                reverse_nn[nb].push_back(static_cast<int32>(i));
            }
        }
    }

    // Compute Jaccard similarities
    // For each cell i, for each neighbor j (j > i), compute:
    //   J(i,j) = |NN(i) intersect NN(j)| / |NN(i) union NN(j)|
    //          = |NN(i) intersect NN(j)| / (2k - |NN(i) intersect NN(j)|)

    // Use vector of pairs for each cell to accumulate connections
    std::vector<std::vector<std::pair<int32, double>>> edges(n_cells);

    #pragma omp parallel num_threads(n_threads)
    {
        // Per-thread reusable buffers to avoid per-cell allocations
        std::vector<int32> candidates;
        std::vector<int32> nn_i_sorted;

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n_cells; ++i) {
            int base_i = i * k;

            // Collect candidate cells that share a neighbor with i
            candidates.clear();
            for (int j = 0; j < k; ++j) {
                int nb = nn_idx[base_i + j];
                for (int nb2 : reverse_nn[nb]) {
                    if (nb2 > i) candidates.push_back(static_cast<int32>(nb2));
                }
                for (int jj = 0; jj < k; ++jj) {
                    int nb2 = nn_idx[nb * k + jj];
                    if (nb2 > i) candidates.push_back(static_cast<int32>(nb2));
                }
            }

            // Deduplicate candidates via sort + unique
            if (candidates.empty()) continue;
            std::sort(candidates.begin(), candidates.end());
            candidates.erase(std::unique(candidates.begin(), candidates.end()),
                           candidates.end());

            // Build sorted neighbor list for binary-search intersection
            nn_i_sorted.clear();
            for (int j = 0; j < k; ++j) {
                nn_i_sorted.push_back(nn_idx[base_i + j]);
            }
            std::sort(nn_i_sorted.begin(), nn_i_sorted.end());

            for (int32 cand : candidates) {
                int base_j = cand * k;

                int intersect = 0;
                for (int j = 0; j < k; ++j) {
                    int32 nb = nn_idx[base_j + j];
                    if (std::binary_search(nn_i_sorted.begin(),
                                           nn_i_sorted.end(), nb)) {
                        intersect++;
                    }
                }

                // Jaccard = intersect / (2k - intersect)
                double jaccard = static_cast<double>(intersect) / (2.0 * k - intersect);

                if (jaccard > prune) {
                    edges[i].emplace_back(cand, jaccard);
                    edges[cand].emplace_back(static_cast<int32>(i), jaccard);
                }
            }
        }
    }

    // Convert to CSC format
    snn_indptr.assign(n_cells + 1, 0);
    for (int i = 0; i < n_cells; ++i) {
        snn_indptr[i + 1] = snn_indptr[i] + static_cast<int64>(edges[i].size());
    }

    int64 total_edges = snn_indptr[n_cells];
    snn_data.resize(total_edges);
    snn_indices.resize(total_edges);

    for (int i = 0; i < n_cells; ++i) {
        int64 offset = snn_indptr[i];
        // Sort by neighbor index for CSC
        std::sort(edges[i].begin(), edges[i].end());
        for (size_t j = 0; j < edges[i].size(); ++j) {
            snn_data[offset + j] = edges[i][j].second;
            snn_indices[offset + j] = edges[i][j].first;
        }
    }
}

} // namespace sclean
