#include "integration_internal.h"
#include <cmath>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sclean {

// ============================================================
// Compute raw correction vectors from MNN pairs
// ============================================================

Eigen::MatrixXd compute_raw_correction(
    const Eigen::MatrixXd& ref_emb,
    const Eigen::MatrixXd& query_emb,
    const std::vector<std::pair<int, int>>& mnn_pairs) {

    int n_query = static_cast<int>(query_emb.rows());
    int d = static_cast<int>(query_emb.cols());

    // Group correction vectors by query cell
    std::unordered_map<int, std::vector<Eigen::VectorXd>> query_corrections;
    for (const auto& pair : mnn_pairs) {
        int ref_cell = pair.first;
        int query_cell = pair.second;

        Eigen::VectorXd vec = ref_emb.row(ref_cell) - query_emb.row(query_cell);
        if (!query_corrections.count(query_cell)) {
            query_corrections[query_cell] = std::vector<Eigen::VectorXd>();
        }
        query_corrections[query_cell].push_back(vec);
    }

    // Average correction per query cell
    Eigen::MatrixXd correction = Eigen::MatrixXd::Zero(n_query, d);
    for (const auto& kv : query_corrections) {
        int q = kv.first;
        Eigen::VectorXd avg_correction = Eigen::VectorXd::Zero(d);
        for (const auto& v : kv.second) {
            avg_correction += v;
        }
        avg_correction /= static_cast<double>(kv.second.size());
        correction.row(q) = avg_correction;
    }

    return correction;
}

// ============================================================
// Gaussian kernel smoothing (full-matrix path)
// ============================================================

Eigen::MatrixXd smooth_correction_gaussian(
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
// Chunked Gaussian kernel smoothing
// ============================================================

Eigen::MatrixXd smooth_correction_gaussian_chunked(
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

} // namespace sclean
