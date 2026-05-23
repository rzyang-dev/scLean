#include "integration_operator.h"
#include "utils/progress.h"
#include "utils/resource_monitor.h"
#include "utils/parallel.h"
#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <stdexcept>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sclean {

IntegrationOperator::IntegrationOperator(int n_dims, int n_mnn,
                                           double sigma, int max_iter)
    : n_dims_(n_dims), n_mnn_(n_mnn),
      sigma_(sigma), max_iter_(max_iter) {}

// ============================================================
// Backward-compatible run() — delegates to scheduler overload
// ============================================================

IntegrationResult IntegrationOperator::run(
    const Eigen::MatrixXd& pca_embeddings,
    const std::vector<int32>& batch_labels) {
    ChunkScheduler scheduler;
    return run(pca_embeddings, batch_labels, scheduler);
}

// ============================================================
// adapt_params: adjust algorithm based on resource bottleneck
// ============================================================

IntegrationOperator::IntegrationParams
IntegrationOperator::adapt_params(const ChunkConfig& cfg) const {
    IntegrationParams p;
    p.n_mnn = n_mnn_;
    p.sigma = sigma_;
    p.max_iter = max_iter_;
    p.use_chunked_smoothing = false;

    if (cfg.fits_in_memory && cfg.bottleneck == Bottleneck::None) {
        return p;
    }

    if (cfg.bottleneck == Bottleneck::MemoryBound ||
        cfg.bottleneck == Bottleneck::BothBound) {
        p.n_mnn = std::max(5, n_mnn_ / 2);
        p.use_chunked_smoothing = true;
    }

    if (cfg.bottleneck == Bottleneck::ComputeBound ||
        cfg.bottleneck == Bottleneck::BothBound) {
        p.max_iter = std::max(1, max_iter_ - 1);
    }

    return p;
}

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

std::vector<std::pair<int, int>> IntegrationOperator::find_mnn_pairs(
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

// ============================================================
// Original compute_correction (delegates to config overload)
// ============================================================

Eigen::MatrixXd IntegrationOperator::compute_correction(
    const Eigen::MatrixXd& embeddings,
    const std::vector<int32>& batch_labels,
    int reference_batch) {
    ChunkConfig default_cfg;
    default_cfg.fits_in_memory = true;
    default_cfg.bottleneck = Bottleneck::None;
    return compute_correction(embeddings, batch_labels, reference_batch,
                              default_cfg, 1);
}

// ============================================================
// compute_correction with scheduler config
// ============================================================

Eigen::MatrixXd IntegrationOperator::compute_correction(
    const Eigen::MatrixXd& embeddings,
    const std::vector<int32>& batch_labels,
    int reference_batch,
    const ChunkConfig& cfg,
    int n_threads) {

    int n_cells = static_cast<int>(embeddings.rows());
    int d = static_cast<int>(embeddings.cols());

    // Identify reference and query cells
    std::vector<int> ref_idx, query_idx;
    for (int i = 0; i < n_cells; ++i) {
        if (batch_labels[i] == reference_batch) {
            ref_idx.push_back(i);
        } else {
            query_idx.push_back(i);
        }
    }

    int n_ref = static_cast<int>(ref_idx.size());
    int n_query = static_cast<int>(query_idx.size());

    Eigen::MatrixXd correction = Eigen::MatrixXd::Zero(n_cells, d);

    if (n_ref == 0 || n_query == 0) return correction;

    // Extract reference and query embeddings
    Eigen::MatrixXd ref_emb(n_ref, d);
    Eigen::MatrixXd query_emb(n_query, d);
    for (int i = 0; i < n_ref; ++i) {
        ref_emb.row(i) = embeddings.row(ref_idx[i]);
    }
    for (int i = 0; i < n_query; ++i) {
        query_emb.row(i) = embeddings.row(query_idx[i]);
    }

    // Find MNN pairs between reference and query
    auto mnn_pairs = find_mnn_pairs(ref_emb, query_emb, n_mnn_);

    if (mnn_pairs.empty()) return correction;

    // Compute raw correction for query cells that have MNN pairs
    std::unordered_map<int, std::vector<Eigen::VectorXd>> query_corrections;
    for (auto& pair : mnn_pairs) {
        int ref_cell = pair.first;
        int query_cell = pair.second;

        Eigen::VectorXd vec = ref_emb.row(ref_cell) - query_emb.row(query_cell);
        if (!query_corrections.count(query_cell)) {
            query_corrections[query_cell] = std::vector<Eigen::VectorXd>();
        }
        query_corrections[query_cell].push_back(vec);
    }

    // Average correction per query cell
    for (auto& kv : query_corrections) {
        int q = kv.first;
        Eigen::VectorXd avg_correction = Eigen::VectorXd::Zero(d);
        for (auto& v : kv.second) {
            avg_correction += v;
        }
        avg_correction /= static_cast<double>(kv.second.size());

        correction.row(query_idx[q]) = avg_correction;
    }

    // Smooth correction using adaptive path selection
    Eigen::MatrixXd smoothed_query = smooth_correction_adaptive(
        query_emb, correction, sigma_, cfg, n_threads);

    // Map smoothed corrections back to global cell indices
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n_cells, d);
    for (int i = 0; i < n_query; ++i) {
        result.row(query_idx[i]) = smoothed_query.row(i);
    }

    return result;
}

// ============================================================
// Gaussian kernel smoothing (original full-matrix path)
// ============================================================

Eigen::MatrixXd IntegrationOperator::smooth_correction(
    const Eigen::MatrixXd& query_emb,
    const Eigen::MatrixXd& raw_correction,
    double sigma) {

    int n = static_cast<int>(query_emb.rows());
    int d = static_cast<int>(query_emb.cols());

    Eigen::MatrixXd smoothed = Eigen::MatrixXd::Zero(n, d);

    // Only smooth for cells that have a non-zero correction
    std::vector<int> cells_with_correction;
    for (int i = 0; i < n; ++i) {
        if (raw_correction.row(i).squaredNorm() > 0) {
            cells_with_correction.push_back(i);
        }
    }

    if (cells_with_correction.empty()) return smoothed;

    // Gaussian kernel smoothing
    //
    // Bandwidth = sigma * ||first_row_of_query_emb||
    // This is a data-driven heuristic: the bandwidth scales with the embedding
    // magnitude so that the smoothing radius adapts to the data scale. The user
    // parameter `sigma` acts as a relative multiplier on this base bandwidth.
    // A fixed absolute bandwidth would be either too narrow (overfitting) or
    // too wide (over-smoothing) depending on the PCA embedding magnitude.
    double bandwidth = sigma * query_emb.row(0).stableNorm();

    if (bandwidth < 1e-6) bandwidth = 1.0;

    for (int i = 0; i < n; ++i) {
        double weight_sum = 0.0;
        Eigen::VectorXd weighted_correction = Eigen::VectorXd::Zero(d);

        for (int j : cells_with_correction) {
            double dist_sq = (query_emb.row(i) - query_emb.row(j)).squaredNorm();
            double w = std::exp(-dist_sq / (2.0 * bandwidth * bandwidth));
            weighted_correction += w * raw_correction.row(j).transpose();
            weight_sum += w;
        }

        if (weight_sum > 1e-10) {
            smoothed.row(i) = weighted_correction / weight_sum;
        }
    }

    return smoothed;
}

// ============================================================
// Adaptive smoothing: selects full or chunked path
// ============================================================

Eigen::MatrixXd IntegrationOperator::smooth_correction_adaptive(
    const Eigen::MatrixXd& query_emb,
    const Eigen::MatrixXd& raw_correction,
    double sigma,
    const ChunkConfig& cfg,
    int n_threads) {

    if (cfg.fits_in_memory &&
        cfg.bottleneck != Bottleneck::MemoryBound &&
        cfg.bottleneck != Bottleneck::BothBound) {
        return smooth_correction(query_emb, raw_correction, sigma);
    }

    int64 chunk_size = cfg.chunk_size;
    if (chunk_size <= 0) chunk_size = 1024;
    return smooth_correction_chunked(query_emb, raw_correction,
                                      sigma, chunk_size, n_threads);
}

// ============================================================
// Chunked Gaussian kernel smoothing
// ============================================================

Eigen::MatrixXd IntegrationOperator::smooth_correction_chunked(
    const Eigen::MatrixXd& query_emb,
    const Eigen::MatrixXd& raw_correction,
    double sigma,
    int64 chunk_size,
    int n_threads) {

    int n = static_cast<int>(query_emb.rows());
    int d = static_cast<int>(query_emb.cols());

    // Identify source cells (those with non-zero corrections)
    std::vector<int> src_cells;
    src_cells.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (raw_correction.row(i).squaredNorm() > 0) {
            src_cells.push_back(i);
        }
    }

    int n_src = static_cast<int>(src_cells.size());
    if (n_src == 0) return Eigen::MatrixXd::Zero(n, d);

    // Pre-extract source embeddings and correction vectors
    Eigen::MatrixXd src_emb(n_src, d);
    Eigen::MatrixXd src_correction(n_src, d);
    for (int k = 0; k < n_src; ++k) {
        int idx = src_cells[k];
        src_emb.row(k) = query_emb.row(idx);
        src_correction.row(k) = raw_correction.row(idx);
    }

    double bandwidth = sigma * query_emb.row(0).stableNorm();
    if (bandwidth < 1e-6) bandwidth = 1.0;
    double inv_two_sigma2 = 1.0 / (2.0 * bandwidth * bandwidth);

    Eigen::MatrixXd smoothed = Eigen::MatrixXd::Zero(n, d);

    int actual_threads = (n_threads > 1) ? n_threads : 1;
    int64 csize = chunk_size;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) num_threads(actual_threads)
#endif
    for (int t_start = 0; t_start < n; t_start += csize) {
        int t_end = std::min(t_start + static_cast<int>(csize), n);

        for (int i = t_start; i < t_end; ++i) {
            double weight_sum = 0.0;
            Eigen::VectorXd weighted = Eigen::VectorXd::Zero(d);

            for (int k = 0; k < n_src; ++k) {
                double dist_sq = (query_emb.row(i) - src_emb.row(k)).squaredNorm();
                double w = std::exp(-dist_sq * inv_two_sigma2);
                weighted.noalias() += w * src_correction.row(k).transpose();
                weight_sum += w;
            }

            if (weight_sum > 1e-10) {
                smoothed.row(i) = weighted / weight_sum;
            }
        }
    }

    return smoothed;
}

// ============================================================
// Main integration routine (scheduler-aware)
//
// Algorithm flow:
// 1. Find unique batches; if only 1 batch, return PCA embeddings as-is.
// 2. Schedule resources via ChunkScheduler; adapt algorithm parameters under
//    resource pressure (reduce n_mnn, enable chunked smoothing).
// 3. Reference batch = batch with the most cells (ties resolved by first
//    encountered in the sorted batch list).
// 4. Iterative correction (max_iter rounds):
//    a. For each non-reference batch, find MNN pairs with the reference.
//    b. Compute correction vectors from MNN pairs.
//    c. Smooth correction vectors via Gaussian kernel (full or chunked path,
//       selected adaptively based on available resources).
//    d. Add smoothed correction to the batch's embeddings.
// 5. OOM recovery: catch bad_alloc, call scheduler.shrink_and_retry(),
//    retry recursively.
//
// NOTE ON NAMING: The corrected embeddings are stored under the key "harmony"
// in the HDF5 file (/reductions/harmony/embeddings) and in the Seurat object
// as a "harmony" DimReduc. This naming is for Seurat ecosystem compatibility:
// Seurat's HarmonyIntegration stores corrected embeddings under "harmony",
// and downstream functions (DimPlot, FeaturePlot) look for this key. The
// algorithm used here is MNN (mutual nearest neighbors), NOT Harmony — the
// two are different methods but share the same output slot convention.
// ============================================================

IntegrationResult IntegrationOperator::run(
    const Eigen::MatrixXd& pca_embeddings,
    const std::vector<int32>& batch_labels,
    ChunkScheduler& scheduler) {

    int n_cells = static_cast<int>(pca_embeddings.rows());
    int n_pcs = static_cast<int>(pca_embeddings.cols());

    int use_dims = std::min(n_dims_, n_pcs);

    // Find unique batches
    std::vector<int32> unique_batches = batch_labels;
    std::sort(unique_batches.begin(), unique_batches.end());
    unique_batches.erase(std::unique(unique_batches.begin(),
                                      unique_batches.end()),
                          unique_batches.end());

    int n_batches = static_cast<int>(unique_batches.size());

    IntegrationResult result;
    result.batch_labels = batch_labels;
    result.n_batches = n_batches;

    if (n_batches <= 1) {
        result.corrected_embeddings = pca_embeddings.leftCols(use_dims);
        return result;
    }

    // --- Resource scheduling ---
    scheduler.refresh_available_ram();
    int n_threads = static_cast<int>(get_num_threads());
    auto cfg = scheduler.schedule(n_cells, use_dims,
                                   OperationType::Integration, n_threads);

    IntegrationParams params = adapt_params(cfg);

    // Save and override instance fields for this run
    int saved_n_mnn = n_mnn_;
    int saved_max_iter = max_iter_;
    n_mnn_ = params.n_mnn;
    max_iter_ = params.max_iter;

    ProgressReporter progress("IntegrateLayers", params.max_iter,
                              ProgressReporter::is_verbose());

    // Use the largest batch as reference (ties broken by first encountered).
    // Cells without MNN pairs receive zero correction vectors.
    int ref_batch = unique_batches[0];
    int ref_count = 0;
    for (int32 b : unique_batches) {
        int cnt = 0;
        for (auto bl : batch_labels) if (bl == b) cnt++;
        if (cnt > ref_count) {
            ref_count = cnt;
            ref_batch = b;
        }
    }

    Eigen::MatrixXd corrected = pca_embeddings.leftCols(use_dims);

    bool oom_occurred = false;
    try {
        for (int iter = 0; iter < params.max_iter; ++iter) {
            progress.message("Iteration " + std::to_string(iter + 1) + "/" +
                             std::to_string(params.max_iter) + "...");
            Eigen::MatrixXd total_correction = Eigen::MatrixXd::Zero(n_cells, use_dims);

            for (int32 batch : unique_batches) {
                if (batch == ref_batch) continue;

                auto batch_correction = compute_correction(
                    corrected, batch_labels, ref_batch, cfg, n_threads);

                for (int i = 0; i < n_cells; ++i) {
                    if (batch_labels[i] == batch) {
                        total_correction.row(i) = batch_correction.row(i);
                    }
                }
            }

            corrected += total_correction;
            progress.step();
        }
    } catch (const std::bad_alloc&) {
        oom_occurred = true;
    }

    // Restore instance fields
    n_mnn_ = saved_n_mnn;
    max_iter_ = saved_max_iter;

    if (oom_occurred) {
        auto snap = ResourceMonitor().snapshot();
        if (!scheduler.shrink_and_retry(n_cells, use_dims,
                OperationType::Integration, n_threads, cfg)) {
            REprintf("[scLean] IntegrateLayers: FATAL after OOM "
                     "(free: %lld MB, RSS: %lld MB)\n",
                     (long long)(snap.free_ram >> 20),
                     (long long)(snap.current_rss >> 20));
            throw;
        }
        REprintf("[scLean] IntegrateLayers: OOM, shrinking chunk to %lld, "
                 "retrying\n", (long long)cfg.chunk_size);
        return run(pca_embeddings, batch_labels, scheduler);
    }

    result.corrected_embeddings = corrected;
    progress.done();
    return result;
}

} // namespace sclean
