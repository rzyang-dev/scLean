#ifndef SCLEAN_INTEGRATION_OPERATOR_H
#define SCLEAN_INTEGRATION_OPERATOR_H

#include <cstdint>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include "../scLean_types.h"
#include "utils/chunk_scheduler.h"

namespace sclean {

class HDF5File;

struct IntegrationResult {
    Eigen::MatrixXd corrected_embeddings;  // (n_cells, n_dims)
    std::vector<int32> batch_labels;
    int n_batches;
};

class IntegrationOperator {
public:
    IntegrationOperator(int n_dims = 30, int n_mnn = 20,
                         double sigma = 0.1, int max_iter = 2);

    // Run batch correction on PCA embeddings (backward-compatible overload)
    IntegrationResult run(const Eigen::MatrixXd& pca_embeddings,
                          const std::vector<int32>& batch_labels);

    // Run batch correction with scheduler integration
    IntegrationResult run(const Eigen::MatrixXd& pca_embeddings,
                          const std::vector<int32>& batch_labels,
                          ChunkScheduler& scheduler);

private:
    int n_dims_;
    int n_mnn_;
    double sigma_;
    int max_iter_;

    struct IntegrationParams {
        int n_mnn;
        double sigma;
        int max_iter;
        bool use_chunked_smoothing;
    };

    // Adapt algorithm parameters based on resource bottleneck
    IntegrationParams adapt_params(const ChunkConfig& cfg) const;

    // Compute correction vectors for each cell based on MNN pairs
    // (backward-compatible overload -- delegates to scheduler version)
    Eigen::MatrixXd compute_correction(
        const Eigen::MatrixXd& embeddings,
        const std::vector<int32>& batch_labels,
        int reference_batch);

    // Compute correction with scheduler config (selects full/chunked smoothing)
    Eigen::MatrixXd compute_correction(
        const Eigen::MatrixXd& embeddings,
        const std::vector<int32>& batch_labels,
        int reference_batch,
        const ChunkConfig& cfg,
        int n_threads);
};

} // namespace sclean

#endif // SCLEAN_INTEGRATION_OPERATOR_H
