#ifndef SCLEAN_PCA_INTERNAL_H
#define SCLEAN_PCA_INTERNAL_H

#include <Eigen/Dense>
#include "core/disk_matrix.h"

namespace sclean {

// Lanczos bidiagonalization: builds k-step bidiagonal decomposition A ≈ U * B * V^T
// using only matvec/rmatvec operations on the DiskMatrix interface.
//
// Parameters:
//   A       - DiskMatrix providing matvec(y=Ax) and rmatvec(y=A^Tx)
//   k       - number of Lanczos steps
//   U       - output: left Lanczos vectors (n_rows x k+1)
//   V       - output: right Lanczos vectors (n_cols x k)
//   alpha   - output: diagonal entries of bidiagonal B (k)
//   beta    - output: superdiagonal entries of bidiagonal B (k); beta[0] is p_norm
//   means   - gene means for on-the-fly centering (nullptr = no centering)
//   sds     - gene standard deviations for on-the-fly scaling (nullptr = no scaling)
//
// When means/sds != nullptr, each matvec applies (A[i,j] - mu[i])/sigma[i]
// on-the-fly — no dense scale.data needed on disk. Zero-variance genes
// (sds[i] <= 0) contribute zero rows after centering.
void lanczos_bidiag(DiskMatrix* A, int k,
                    Eigen::MatrixXd& U, Eigen::MatrixXd& V,
                    Eigen::VectorXd& alpha, Eigen::VectorXd& beta,
                    const double* means, const double* sds);

} // namespace sclean

#endif // SCLEAN_PCA_INTERNAL_H
