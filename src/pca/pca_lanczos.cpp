#include "pca_internal.h"
#include "utils/progress.h"
#include <cmath>
#include <random>
#include <Eigen/Dense>

namespace sclean {

// ============================================================
// Lanczos Bidiagonalization
//
// Implements implicit restarted Lanczos bidiagonalization (IRLBA).
// The algorithm builds a kxk bidiagonal matrix B such that A approx U * B * V^T.
//
// Initialization:
//   - Random start vector p ~ N(0,1) with fixed seed=42 for reproducibility.
//     Reproducibility matters because PCA sign flips between runs confuse users.
//   - First Lanczos vector: u_1 = A*p / ||A*p||.
//
// On-the-fly centering (do_otf):
//   When means/sds != nullptr, each matvec applies:
//     u[i] = (A*p)[i] / sigma[i] - mu[i]/sigma[i] * sum(p)
//   This avoids materializing the full (genes x cells) scaled matrix on disk.
//   Zero-variance genes (sds[i] <= 0) contribute zero rows after centering.
//
// Iteration count: Lanczos steps k is determined by the caller (typically
// npcs + 10 for accuracy), capped at min(n_genes, n_cells).
// ============================================================

void lanczos_bidiag(
    DiskMatrix* A, int k,
    Eigen::MatrixXd& U, Eigen::MatrixXd& V,
    Eigen::VectorXd& alpha, Eigen::VectorXd& beta,
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

} // namespace sclean
