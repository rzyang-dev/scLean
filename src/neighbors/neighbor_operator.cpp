#include "neighbor_operator.h"
#include "utils/progress.h"
#include "utils/resource_monitor.h"
#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>

// Raw Annoy headers — avoid RcppAnnoy.h which conflicts with Rcpp.h includes
#include "annoylib.h"
#include "kissrandom.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sclean {

NeighborOperator::NeighborOperator(int k, int n_trees)
    : k_(k), n_trees_(n_trees) {}

// ============================================================
// Block-based KNN using Eigen
// ============================================================

void NeighborOperator::knn_blocked(
    const Eigen::MatrixXd& data,
    int k, int block_size,
    std::vector<int32>& idx,
    std::vector<float>& dists) {

    int n_cells = static_cast<int>(data.rows());
    int d = static_cast<int>(data.cols());

    idx.assign(n_cells * k, 0);
    dists.assign(n_cells * k, 0.0f);

    if (n_cells <= 1) {
        if (n_cells == 1) {
            for (int j = 0; j < std::min(k, 1); ++j) {
                idx[j] = 0;
                dists[j] = 0.0f;
            }
        }
        return;
    }

    // Process in query blocks
    for (int q_start = 0; q_start < n_cells; q_start += block_size) {
        int q_end = std::min(q_start + block_size, n_cells);
        int n_queries = q_end - q_start;

        // Compute distances from query block to all cells
        // D[q, r] = ||data[q_start+q] - data[r]||^2
        // = ||x_q||^2 + ||x_r||^2 - 2 * x_q^T * x_r

        // Precompute ||x_r||^2 for all cells
        Eigen::VectorXd norms = data.rowwise().squaredNorm();

        // Query norms
        Eigen::MatrixXd query_block = data.middleRows(q_start, n_queries);
        Eigen::VectorXd q_norms = query_block.rowwise().squaredNorm();

        // For each query cell, compute distances to ALL reference cells
        for (int qi = 0; qi < n_queries; ++qi) {
            int cell_idx = q_start + qi;
            double q_norm = q_norms(qi);

            // Use min-heap of max size k (store negative distances for max-heap behavior)
            // Actually use a max-heap for the k smallest: store pairs of (-dist, idx)
            std::priority_queue<std::pair<double, int>> heap;

            // Process reference cells in blocks — match query block size
            int ref_block = std::min(block_size, 2048);
            for (int r_start = 0; r_start < n_cells; r_start += ref_block) {
                int r_end = std::min(r_start + ref_block, n_cells);
                Eigen::MatrixXd ref_block_data = data.middleRows(r_start, r_end - r_start);

                // Compute dot products: query_block.row(qi) * ref_block^T
                Eigen::VectorXd dots = ref_block_data * query_block.row(qi).transpose();

                for (int r = 0; r < dots.size(); ++r) {
                    int ref_idx = r_start + r;
                    if (ref_idx == cell_idx) continue;  // skip self

                    double dist_sq = q_norm + norms(ref_idx) - 2.0 * dots(r);
                    dist_sq = std::max(0.0, dist_sq);

                    if (static_cast<int>(heap.size()) < k) {
                        heap.push({dist_sq, ref_idx});
                    } else if (dist_sq < heap.top().first) {
                        heap.pop();
                        heap.push({dist_sq, ref_idx});
                    }
                }
            }

            // Extract k nearest neighbors (heap is max-heap, so extract in reverse)
            int heap_sz = static_cast<int>(heap.size());
            std::vector<std::pair<double, int>> neighbors(heap_sz);
            for (int j = heap_sz - 1; j >= 0; --j) {
                neighbors[j] = heap.top();
                heap.pop();
            }

            int actual_k = std::min(k, heap_sz);
            int base = cell_idx * k;
            for (int j = 0; j < actual_k; ++j) {
                idx[base + j] = static_cast<int32>(neighbors[j].second);
                dists[base + j] = static_cast<float>(std::sqrt(neighbors[j].first));
            }
        }
    }
}

void NeighborOperator::knn_annoy(
    const Eigen::MatrixXd& data,
    int k, int n_trees,
    std::vector<int32>& idx,
    std::vector<float>& dists) {

    int n_cells = static_cast<int>(data.rows());
    int d = static_cast<int>(data.cols());

    idx.assign(n_cells * k, 0);
    dists.assign(n_cells * k, 0.0f);

    if (n_cells <= 1) {
        if (n_cells == 1) {
            for (int j = 0; j < std::min(k, 1); ++j) {
                idx[j] = 0;
                dists[j] = 0.0f;
            }
        }
        return;
    }

    // Build Annoy index (raw headers, no RcppAnnoy wrapper)
    using AnnoyIdx = Annoy::AnnoyIndex<int32, float, Annoy::Euclidean,
        Kiss64Random, Annoy::AnnoyIndexSingleThreadedBuildPolicy>;
    AnnoyIdx index(d);

    for (int i = 0; i < n_cells; ++i) {
        std::vector<float> row(d);
        Eigen::VectorXd ei = data.row(i);
        for (int j = 0; j < d; ++j) {
            row[j] = static_cast<float>(ei(j));
        }
        index.add_item(i, row.data());
    }

    index.build(n_trees);

    int search_k = std::max(n_trees * k, 0);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_cells; ++i) {
        std::vector<int32> nn_result;
        std::vector<float> nn_dist;
        index.get_nns_by_item(i, k + 1, search_k, &nn_result, &nn_dist);

        int base = i * k;
        int written = 0;
        for (size_t j = 0; j < nn_result.size() && written < k; ++j) {
            if (nn_result[j] != i) {
                idx[base + written] = nn_result[j];
                dists[base + written] = std::sqrt(
                    std::max(0.0f, nn_dist[j]));
                ++written;
            }
        }
    }

    index.unbuild();
}

// ============================================================

NeighborResult NeighborOperator::run(const Eigen::MatrixXd& embeddings, int k,
                                     ChunkScheduler& scheduler) {
    int n_cells = static_cast<int>(embeddings.rows());
    int dims = static_cast<int>(embeddings.cols());

    NeighborResult result;
    result.k = k;
    result.n_cells = n_cells;

    scheduler.refresh_available_ram();
    auto cfg = scheduler.schedule(n_cells, dims, OperationType::FindNeighbors, 1);

    ProgressReporter progress("FindNeighbors", 2,
                              ProgressReporter::is_verbose());
    progress.message("Finding KNN...");

    bool use_annoy = (n_trees_ > 0 && n_cells >= 5000 &&
                      cfg.bottleneck != Bottleneck::MemoryBound &&
                      cfg.bottleneck != Bottleneck::BothBound);

    if (use_annoy) {
        // Annoy approximate NN for large datasets
        knn_annoy(embeddings, k, n_trees_, result.nn_idx, result.nn_dists);
    } else {
        // Brute-force exact NN — use scheduler-informed block size
        int block_size = std::max(512,
            std::min(static_cast<int>(cfg.chunk_size), n_cells));
        if (cfg.bottleneck == Bottleneck::MemoryBound ||
            cfg.bottleneck == Bottleneck::BothBound) {
            block_size = std::min(block_size, 512);
        }
        knn_blocked(embeddings, k, block_size, result.nn_idx, result.nn_dists);
    }
    progress.step();

    // Include self as first neighbor (Seurat convention: self is NN)
    progress.message("Formatting neighbor indices...");
    std::vector<int32> idx_with_self(n_cells * k);
    std::vector<float> dists_with_self(n_cells * k);

    for (int i = 0; i < n_cells; ++i) {
        int base = i * k;
        idx_with_self[base] = static_cast<int32>(i);
        dists_with_self[base] = 0.0f;

        // Copy k-1 neighbors (skip the last from original)
        for (int j = 0; j < k - 1; ++j) {
            idx_with_self[base + j + 1] = result.nn_idx[base + j];
            dists_with_self[base + j + 1] = result.nn_dists[base + j];
        }
    }

    result.nn_idx = std::move(idx_with_self);
    result.nn_dists = std::move(dists_with_self);
    progress.done();
    return result;
}

// ============================================================
// SNN Graph Construction (Jaccard similarity)
// ============================================================

void NeighborOperator::build_snn(
    const NeighborResult& knn,
    std::vector<double>& snn_data,
    std::vector<int32>& snn_indices,
    std::vector<int64>& snn_indptr,
    double prune,
    int n_threads) {

    int n_cells = knn.n_cells;
    int k = knn.k;

    ProgressReporter progress("BuildSNN", 4,
                              ProgressReporter::is_verbose());

    // Build reverse index: for each cell, which cells have it as a neighbor
    progress.message("Building reverse neighbor index...");
    // This is a list of lists; use adjacency for efficiency
    std::vector<std::vector<int32>> reverse_nn(n_cells);
    for (int i = 0; i < n_cells; ++i) {
        int base = i * k;
        for (int j = 0; j < k; ++j) {
            int nb = knn.nn_idx[base + j];
            if (nb != i) {
                reverse_nn[nb].push_back(static_cast<int32>(i));
            }
        }
    }

    progress.step();

    // Compute Jaccard similarities
    // For each cell i, for each neighbor j (j > i), compute:
    progress.message("Computing Jaccard similarities...");
    //   J(i,j) = |NN(i) ∩ NN(j)| / |NN(i) ∪ NN(j)|
    //          = |NN(i) ∩ NN(j)| / (2k - |NN(i) ∩ NN(j)|)

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
                int nb = knn.nn_idx[base_i + j];
                for (int nb2 : reverse_nn[nb]) {
                    if (nb2 > i) candidates.push_back(static_cast<int32>(nb2));
                }
                for (int jj = 0; jj < k; ++jj) {
                    int nb2 = knn.nn_idx[nb * k + jj];
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
                nn_i_sorted.push_back(knn.nn_idx[base_i + j]);
            }
            std::sort(nn_i_sorted.begin(), nn_i_sorted.end());

            for (int32 cand : candidates) {
                int base_j = cand * k;

                int intersect = 0;
                for (int j = 0; j < k; ++j) {
                    int32 nb = knn.nn_idx[base_j + j];
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

    progress.step();
    progress.message("Converting to CSC format...");

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

    progress.done();
}

} // namespace sclean
