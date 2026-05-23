#include "neighbor_internal.h"
#include <cmath>
#include <algorithm>
#include <queue>

// Raw Annoy headers — avoid RcppAnnoy.h which conflicts with Rcpp.h includes
#ifndef STRICT_R_HEADERS
#define STRICT_R_HEADERS
#endif
#include "annoylib.h"
#include "kissrandom.h"

namespace sclean {

// ============================================================
// Block-based KNN using Eigen
// ============================================================

void knn_blocked(
    const Eigen::MatrixXd& data,
    int k, int block_size,
    std::vector<int32>& idx,
    std::vector<float>& dists) {

    int n_cells = static_cast<int>(data.rows());
    int d = static_cast<int>(data.cols());
    (void)d;  // retained for API symmetry with knn_annoy

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

// ============================================================
// Annoy-based approximate KNN
// ============================================================

void knn_annoy(
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

} // namespace sclean
