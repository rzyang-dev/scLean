#ifndef SCLEAN_PCA_OPERATOR_H
#define SCLEAN_PCA_OPERATOR_H

#include <cstdint>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include "core/disk_matrix.h"
#include "utils/chunk_scheduler.h"

namespace sclean {

class HDF5File;

struct PCAResult {
    Eigen::MatrixXd loadings;     // (n_genes, npcs)
    Eigen::MatrixXd embeddings;   // (n_cells, npcs)
    Eigen::VectorXd stdev;        // (npcs,)
    double total_variance;
    int n_iter;
};

class PCAOperator {
public:
    PCAOperator(int npcs = 50, bool center = true, bool scale = false,
                double tol = 1e-5, int max_iter = 200);

    // run with optional on-the-fly centering/scaling.
    // When means/sds are provided, Lanczos applies (A[i,j] - mu[i])/sigma[i]
    // on-the-fly during matvec/rmatvec — no dense scale.data needed on disk.
    PCAResult run(DiskMatrix* matrix,
                  ChunkScheduler& scheduler,
                  int n_threads = 1,
                  const double* means = nullptr,
                  const double* sds = nullptr);

    PCAResult run_on_subset(DiskMatrix* matrix,
                             const std::vector<int64>& feature_indices,
                             ChunkScheduler& scheduler,
                             int n_threads = 1);

private:
    int npcs_;
    bool center_, scale_;
    double tol_;
    int max_iter_;
};

} // namespace sclean

#endif // SCLEAN_PCA_OPERATOR_H
