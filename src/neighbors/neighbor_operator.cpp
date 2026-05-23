#include "neighbor_operator.h"
#include "neighbor_internal.h"
#include "utils/progress.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sclean {

NeighborOperator::NeighborOperator(int k, int n_trees)
    :  n_trees_(n_trees) { (void)k; }

// ============================================================
// KNN + SNN orchestration
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
// SNN Graph orchestration
// ============================================================

void NeighborOperator::build_snn(
    const NeighborResult& knn,
    std::vector<double>& snn_data,
    std::vector<int32>& snn_indices,
    std::vector<int64>& snn_indptr,
    double prune,
    int n_threads) {

    // Delegate to the free function in neighbor_snn.cpp
    build_snn_graph(knn.nn_idx, knn.nn_dists, knn.k, knn.n_cells,
                    prune, n_threads,
                    snn_data, snn_indices, snn_indptr);
}

} // namespace sclean
