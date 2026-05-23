#include "integration_operator.h"
#include "integration_internal.h"
#include "utils/progress.h"
#include "utils/resource_monitor.h"
#include "utils/parallel.h"
#include <algorithm>
#include <cmath>
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
// Backward-compatible run() -- delegates to scheduler overload
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
// compute_correction (backward-compatible -- delegates to
// scheduler overload with default config)
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
//
// Orchestrates the correction pipeline by delegating to pure
// functions from integration_internal.h:
//   1. Split embeddings into reference / query
//   2. find_mnn_pairs()     → integration_mnn.cpp
//   3. compute_raw_correction() → integration_correct.cpp
//   4. smooth_correction_gaussian() or _chunked()
//   5. Map back to global index space
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

    // Find MNN pairs between reference and query (pure function)
    auto mnn_pairs = find_mnn_pairs(ref_emb, query_emb, n_mnn_);

    if (mnn_pairs.empty()) return correction;

    // Compute raw correction vectors from MNN pairs (pure function)
    Eigen::MatrixXd raw_correction = compute_raw_correction(
        ref_emb, query_emb, mnn_pairs);

    // Smooth correction: select full or chunked path based on config
    Eigen::MatrixXd smoothed_query;
    if (cfg.fits_in_memory &&
        cfg.bottleneck != Bottleneck::MemoryBound &&
        cfg.bottleneck != Bottleneck::BothBound) {
        smoothed_query = smooth_correction_gaussian(
            query_emb, raw_correction, sigma_);
    } else {
        int64 chunk_size = cfg.chunk_size;
        if (chunk_size <= 0) chunk_size = 1024;
        smoothed_query = smooth_correction_gaussian_chunked(
            query_emb, raw_correction, sigma_, chunk_size, n_threads);
    }

    // Map smoothed corrections back to global cell indices
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n_cells, d);
    for (int i = 0; i < n_query; ++i) {
        result.row(query_idx[i]) = smoothed_query.row(i);
    }

    return result;
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
//    a. For each non-reference batch, compute correction via
//       compute_correction() which delegates to pure functions.
//    b. Accumulate corrections and add to embeddings.
// 5. OOM recovery: catch bad_alloc, call scheduler.shrink_and_retry(),
//    retry recursively.
//
// NOTE ON NAMING: The corrected embeddings are stored under the key "harmony"
// in the HDF5 file (/reductions/harmony/embeddings) and in the Seurat object
// as a "harmony" DimReduc. This naming is for Seurat ecosystem compatibility:
// Seurat's HarmonyIntegration stores corrected embeddings under "harmony",
// and downstream functions (DimPlot, FeaturePlot) look for this key. The
// algorithm used here is MNN (mutual nearest neighbors), NOT Harmony -- the
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
