#include "pca_operator.h"
#include "pca_internal.h"
#include "utils/progress.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <Eigen/SVD>

namespace sclean {

PCAOperator::PCAOperator(int npcs, bool center, bool scale,
                           double tol, int max_iter)
    : npcs_(npcs), center_(center), scale_(scale),
      tol_(tol), max_iter_(max_iter) {}

// ============================================================
// Main run method
// ============================================================

PCAResult PCAOperator::run(
    DiskMatrix* matrix,
    ChunkScheduler& scheduler,
    int n_threads,
    const double* means,
    const double* sds) {

    // Refresh resource state
    scheduler.refresh_available_ram();

    int64 n_genes = matrix->n_rows();
    int64 n_cells = matrix->n_cols();

    // Determine Lanczos steps: use more than npcs for accuracy
    int k = std::min(static_cast<int64>(npcs_ + 10),
                     std::min(n_genes, n_cells));

    PCAResult result;
    result.n_iter = 0;

    ProgressReporter progress("RunPCA", 3,
                              ProgressReporter::is_verbose());

    // For very small matrices, just do dense PCA via Eigen.
    // Centering is done in-memory here; means/sds are ignored for this path.
    //
    // Threshold: n_genes <= 2000 && n_cells <= 5000
    // Rationale: 2000x5000 = 10M doubles approx 80 MB fits comfortably in a single
    // dense allocation on 8 GB. Above this, we prefer Lanczos to avoid OOM.
    // The dense SVD path uses BDCSVD (fast for moderate matrices) with the
    // full matrix read into memory and centered row-wise before SVD.
    // If the dense allocation fails with bad_alloc, we fall through to Lanczos.
    if (n_genes <= 2000 && n_cells <= 5000) {
        progress.message("Computing PCA on small matrix (dense SVD)...");
        // Read full matrix into memory (with OOM guard)
        std::vector<double> full;
        try {
            full.resize(n_genes * n_cells);
            matrix->read_rows(full.data(), 0, n_genes, 0, n_cells);
        } catch (const std::bad_alloc&) {
            REprintf("[scLean] RunPCA: dense SVD OOM, falling back to Lanczos\n");
            // Fall through to Lanczos path below
            full.clear();
        }
        if (!full.empty()) {
        Eigen::Map<Eigen::MatrixXd> A(full.data(), n_genes, n_cells);

        // Center
        Eigen::VectorXd row_means = A.rowwise().mean();
        Eigen::MatrixXd centered = A;
        for (int64 i = 0; i < n_genes; ++i) {
            centered.row(i).array() -= row_means(i);
        }

        // SVD
        Eigen::BDCSVD<Eigen::MatrixXd> svd(centered,
            Eigen::ComputeThinU | Eigen::ComputeThinV);
        int actual_npcs = std::min(npcs_, static_cast<int>(svd.singularValues().size()));

        result.loadings = svd.matrixU().leftCols(actual_npcs);
        result.embeddings = svd.matrixV().leftCols(actual_npcs) *
                            svd.singularValues().head(actual_npcs).asDiagonal();
        result.stdev = svd.singularValues().head(actual_npcs) /
                       std::sqrt(std::max(1.0, static_cast<double>(n_cells - 1)));
        result.total_variance = svd.singularValues().squaredNorm() / (n_cells - 1);
        result.n_iter = 1;
        progress.done();
        return result;
    }
    }  // end if (n_genes <= 2000 && n_cells <= 5000)

    // Lanczos bidiagonalization (with optional on-the-fly centering)
    progress.message("Running Lanczos bidiagonalization...");
    Eigen::MatrixXd U, V;
    Eigen::VectorXd alpha, beta_vec;
    lanczos_bidiag(matrix, k, U, V, alpha, beta_vec, means, sds);
    progress.step();

    // Build bidiagonal matrix B (k x k) from alpha and beta
    progress.message("Computing SVD of bidiagonal matrix...");
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(k, k);
    for (int i = 0; i < k; ++i) {
        B(i, i) = alpha(i);
        if (i + 1 < k) {
            B(i + 1, i) = beta_vec(i + 1);
        }
    }

    // Check for NaN/Inf in alpha/beta (can happen with extreme sparsity)
    bool has_nan = false;
    for (int i = 0; i < k && !has_nan; ++i) {
        if (!std::isfinite(alpha(i)) || (i + 1 < k && !std::isfinite(beta_vec(i + 1)))) {
            has_nan = true;
        }
    }

    // SVD of bidiagonal B — prefer BDCSVD for speed, fall back to JacobiSVD
    // when BDCSVD fails. JacobiSVD is slower but correctly handles near-zero
    // singular values from rank-deficient or extremely sparse matrices.
    // NaN/Inf check (above): extreme sparsity can produce degenerate Lanczos
    // vectors; when detected we return zero results rather than propagating NaN.
    int actual_npcs;
    if (has_nan) {
        // Degenerate Lanczos: return zero results
        actual_npcs = 0;
        result.loadings = Eigen::MatrixXd::Zero(n_genes, npcs_);
        result.embeddings = Eigen::MatrixXd::Zero(n_cells, npcs_);
        result.stdev = Eigen::VectorXd::Zero(npcs_);
        result.total_variance = 0.0;
        result.n_iter = 0;
        progress.done();
        return result;
    }

    Eigen::BDCSVD<Eigen::MatrixXd> svd_b(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (svd_b.info() != Eigen::Success) {
        // Fallback: JacobiSVD is slower but more robust
        Eigen::JacobiSVD<Eigen::MatrixXd> jsvd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
        actual_npcs = std::min(npcs_, static_cast<int>(jsvd.singularValues().size()));
        result.loadings = U.leftCols(k) * jsvd.matrixU().leftCols(actual_npcs);
        result.stdev = jsvd.singularValues().head(actual_npcs) /
                       std::sqrt(std::max(1.0, static_cast<double>(n_cells - 1)));
        result.embeddings = V.leftCols(k) * jsvd.matrixV().leftCols(actual_npcs) *
                           jsvd.singularValues().head(actual_npcs).asDiagonal();
    } else {
        actual_npcs = std::min(npcs_, static_cast<int>(svd_b.singularValues().size()));
        result.loadings = U.leftCols(k) * svd_b.matrixU().leftCols(actual_npcs);
        result.stdev = svd_b.singularValues().head(actual_npcs) /
                       std::sqrt(std::max(1.0, static_cast<double>(n_cells - 1)));
        result.embeddings = V.leftCols(k) * svd_b.matrixV().leftCols(actual_npcs) *
                           svd_b.singularValues().head(actual_npcs).asDiagonal();
    }

    result.total_variance = svd_b.singularValues().squaredNorm() / (n_cells - 1);
    result.n_iter = k;

    progress.done();
    return result;
}

PCAResult PCAOperator::run_on_subset(
    DiskMatrix* matrix,
    const std::vector<int64>& feature_indices,
    ChunkScheduler& scheduler,
    int n_threads) {

    // Use variable features only — compute PCA restricted to a gene subset.
    //
    // If the subset has <= 2000 genes, read them into a dense matrix and
    // compute full SVD in-memory (fast path). Row-wise centering is applied
    // in-memory; no on-the-fly centering via means/sds.
    //
    // For larger subsets (>2000 genes), fall back to full Lanczos PCA on the
    // complete matrix passing nullptr for means/sds — subset-restricted Lanczos
    // with on-the-fly centering is not yet implemented. This is a known
    // limitation: PCA on large variable-feature subsets always runs on all genes.
    int64 n_subset = static_cast<int64>(feature_indices.size());
    int64 n_cells = matrix->n_cols();

    if (n_subset <= 2000) {
        // Read selected rows
        std::vector<double> buf(n_subset * n_cells);
        for (size_t i = 0; i < feature_indices.size(); ++i) {
            int64 row = feature_indices[i];
            matrix->read_rows(buf.data() + i * n_cells, row, 1, 0, n_cells);
        }

        Eigen::Map<Eigen::MatrixXd> A(buf.data(), n_subset, n_cells);
        Eigen::VectorXd row_means = A.rowwise().mean();
        for (int64 i = 0; i < n_subset; ++i) {
            A.row(i).array() -= row_means(i);
        }

        Eigen::BDCSVD<Eigen::MatrixXd> svd(A,
            Eigen::ComputeThinU | Eigen::ComputeThinV);

        int actual_npcs = std::min(npcs_, static_cast<int>(svd.singularValues().size()));

        PCAResult result;
        result.loadings.resize(n_subset, actual_npcs);
        result.loadings = svd.matrixU().leftCols(actual_npcs);
        result.embeddings = svd.matrixV().leftCols(actual_npcs) *
                            svd.singularValues().head(actual_npcs).asDiagonal();
        result.stdev = svd.singularValues().head(actual_npcs) /
                       std::sqrt(std::max(1.0, static_cast<double>(n_cells - 1)));
        result.total_variance = svd.singularValues().squaredNorm() / (n_cells - 1);
        result.n_iter = 1;
        return result;
    }

    // For larger subsets, fall back to full PCA (Lanczos on the full matrix
    // with variable feature subset not yet supported in Lanczos path).
    // Pass nullptr for means/sds — subset centering not yet implemented.
    return run(matrix, scheduler, n_threads, nullptr, nullptr);
}

} // namespace sclean
