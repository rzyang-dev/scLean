#include "integration_internal.h"
#include <queue>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sclean {

// ============================================================
// MNN pair finding
//
// Brute-force mutual nearest neighbor search: O(n1 * n2 * d).
// For each cell in batch1, finds k nearest neighbors in batch2 using a
// max-heap (priority_queue). Then checks reciprocity: cell i in batch1 and
// cell j in batch2 are MNN pairs if i is in j's top-k AND j is in i's top-k.
//
// Performance note: this is O(n^2 * d) and becomes the bottleneck for large
// batches. A future optimization could use Annoy for approximate MNN finding
// when n_cells exceeds a threshold (cf. FindNeighbors threshold at 5000).
// ============================================================

std::vector<std::pair<int, int>> find_mnn_pairs(
    const Eigen::MatrixXd& batch1_emb,
    const Eigen::MatrixXd& batch2_emb,
    int k) {

    int n1 = static_cast<int>(batch1_emb.rows());
    int n2 = static_cast<int>(batch2_emb.rows());

    // For each cell in batch1, find k nearest neighbors in batch2
    std::vector<std::vector<int>> nn_b1_to_b2(n1);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < n1; ++i) {
        std::priority_queue<std::pair<double, int>> heap;
        for (int j = 0; j < n2; ++j) {
            double dist_sq = (batch1_emb.row(i) - batch2_emb.row(j)).squaredNorm();
            if (static_cast<int>(heap.size()) < k) {
                heap.push({dist_sq, j});
            } else if (dist_sq < heap.top().first) {
                heap.pop();
                heap.push({dist_sq, j});
            }
        }
        std::vector<int>& nn_list = nn_b1_to_b2[i];
        while (!heap.empty()) {
            nn_list.push_back(heap.top().second);
            heap.pop();
        }
        std::reverse(nn_list.begin(), nn_list.end());
    }

    // For each cell in batch2, find k nearest neighbors in batch1
    std::vector<std::vector<int>> nn_b2_to_b1(n2);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int j = 0; j < n2; ++j) {
        std::priority_queue<std::pair<double, int>> heap;
        for (int i = 0; i < n1; ++i) {
            double dist_sq = (batch2_emb.row(j) - batch1_emb.row(i)).squaredNorm();
            if (static_cast<int>(heap.size()) < k) {
                heap.push({dist_sq, i});
            } else if (dist_sq < heap.top().first) {
                heap.pop();
                heap.push({dist_sq, i});
            }
        }
        std::vector<int>& nn_list = nn_b2_to_b1[j];
        while (!heap.empty()) {
            nn_list.push_back(heap.top().second);
            heap.pop();
        }
        std::reverse(nn_list.begin(), nn_list.end());
    }

    // Find mutual nearest neighbors
    std::vector<std::pair<int, int>> mnn_pairs;

    for (int i = 0; i < n1; ++i) {
        for (int j : nn_b1_to_b2[i]) {
            for (int i2 : nn_b2_to_b1[j]) {
                if (i2 == i) {
                    mnn_pairs.emplace_back(i, j);
                    break;
                }
            }
        }
    }

    return mnn_pairs;
}

} // namespace sclean
