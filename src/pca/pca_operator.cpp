#include "pca_operator.h"
#include "utils/progress.h"
#include "utils/resource_monitor.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <Eigen/SVD>
#include <Eigen/QR>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sclean {

PCAOperator::PCAOperator(int npcs, bool center, bool scale,
                           double tol, int max_iter)
    : npcs_(npcs), center_(center), scale_(scale),
      tol_(tol), max_iter_(max_iter) {}

// ============================================================
// Lanczos Bidiagonalization
// ============================================================

void PCAOperator::lanczos_bidiag(
    DiskMatrix* A, int k,
    Eigen::MatrixXd& U, Eigen::MatrixXd& V,
    Eigen::VectorXd& alpha, Eigen::VectorXd& beta,
    ChunkScheduler& scheduler, int n_threads,
    const double* means, const double* sds) {

    int64 m = A->n_rows();  // n_genes
    int64 n = A->n_cols();  // n_cells
    bool do_otf = (means != nullptr && sds != nullptr);

    U.resize(m, k + 1);
    V.resize(n, k);
    alpha.resize(k);
    beta.resize(k);
    U.setZero();
    V.setZero();
    alpha.setZero();
    beta.setZero();

    // Initialize with random vector
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::VectorXd p = Eigen::VectorXd::Zero(n);
    for (int64 i = 0; i < n; ++i) {
        p(i) = dist(rng);
    }
    double p_norm = p.norm();
    if (p_norm > 0) p /= p_norm;

    // beta_1 * u_1 = A * p
    std::vector<double> p_vec(n);
    Eigen::VectorXd::Map(p_vec.data(), n) = p;
    std::vector<double> u_vec(m);
    A->matvec(p_vec.data(), u_vec.data());

    if (do_otf) {
        // u[i] = (A*p)[i] / sigma[i] - mu[i]/sigma[i] * sum(p)
        double sum_p = 0.0;
        for (int64 i = 0; i < n; ++i) sum_p += p_vec[i];
        for (int64 i = 0; i < m; ++i) {
            if (sds[i] > 0.0) {
                u_vec[i] = (u_vec[i] - means[i] * sum_p) / sds[i];
            } else {
                // Zero-variance gene: row is all zeros after centering
                u_vec[i] = 0.0;
            }
        }
    }
    Eigen::VectorXd u = Eigen::VectorXd::Map(u_vec.data(), m);

    beta(0) = u.norm();
    if (beta(0) > 0) {
        U.col(0) = u / beta(0);
    }

    // Lanczos iteration
    ProgressReporter lanczos_progress("  Lanczos", k,
                                      ProgressReporter::is_verbose());
    for (int j = 0; j < k; ++j) {
        // r = A^T * u_j - beta_j * v_{j-1}  (v_{-1} = 0)
        std::vector<double> uj_vec(m);
        Eigen::VectorXd::Map(uj_vec.data(), m) = U.col(j);
        std::vector<double> r_vec(n);

        if (do_otf) {
            // r[j] = sum_i ((A[i,j]-mu[i])/sigma[i]) * u[i]
            //      = sum_i A[i,j] * u[i]/sigma[i] - sum_i mu[i]/sigma[i]*u[i]
            double constant = 0.0;
            for (int64 i = 0; i < m; ++i) {
                if (sds[i] > 0.0) {
                    constant += means[i] / sds[i] * uj_vec[i];
                    uj_vec[i] /= sds[i];
                } else {
                    uj_vec[i] = 0.0;  // zero-variance gene: no contribution
                }
            }
            A->rmatvec(uj_vec.data(), r_vec.data());
            for (int64 i = 0; i < n; ++i) {
                r_vec[i] -= constant;
            }
        } else {
            A->rmatvec(uj_vec.data(), r_vec.data());
        }
        Eigen::VectorXd r = Eigen::VectorXd::Map(r_vec.data(), n);

        if (j > 0) {
            r -= beta(j) * V.col(j - 1);
        }

        // Reorthogonalize against previous V columns
        for (int i = 0; i < j; ++i) {
            r -= (r.dot(V.col(i))) * V.col(i);
        }

        alpha(j) = r.norm();
        if (alpha(j) > 0) {
            V.col(j) = r / alpha(j);
        }

        if (j == k - 1) {
            lanczos_progress.step();
            break;  // Last step: don't compute next u
        }

        // p = A * v_j - alpha_j * u_j
        std::vector<double> vj_vec(n);
        Eigen::VectorXd::Map(vj_vec.data(), n) = V.col(j);
        std::vector<double> p_vec2(m);
        A->matvec(vj_vec.data(), p_vec2.data());

        if (do_otf) {
            double sum_v = 0.0;
            for (int64 i = 0; i < n; ++i) sum_v += vj_vec[i];
            for (int64 i = 0; i < m; ++i) {
                if (sds[i] > 0.0) {
                    p_vec2[i] = (p_vec2[i] - means[i] * sum_v) / sds[i];
                } else {
                    p_vec2[i] = 0.0;
                }
            }
        }
        Eigen::VectorXd p_next = Eigen::VectorXd::Map(p_vec2.data(), m);

        p_next -= alpha(j) * U.col(j);

        // Reorthogonalize against previous U columns
        for (int i = 0; i <= j; ++i) {
            p_next -= (p_next.dot(U.col(i))) * U.col(i);
        }

        beta(j + 1) = p_next.norm();
        if (beta(j + 1) > 0) {
            U.col(j + 1) = p_next / beta(j + 1);
        }
        lanczos_progress.step();
    }
    lanczos_progress.done();
}

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
    lanczos_bidiag(matrix, k, U, V, alpha, beta_vec, scheduler, n_threads,
                   means, sds);
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

    // SVD of bidiagonal B — use JacobiSVD for robustness when BDCSVD may
    // struggle with near-zero singular values from extremely sparse matrices
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

    // Use variable features only
    // For the MVP, read the subset rows, compute in-memory, then return
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
