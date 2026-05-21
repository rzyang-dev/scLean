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

    // Find mutual nearest neighbors between two batches
    std::vector<std::pair<int, int>> find_mnn_pairs(
        const Eigen::MatrixXd& batch1_emb,
        const Eigen::MatrixXd& batch2_emb,
        int k);

    // Compute correction vectors for each cell based on MNN pairs
    Eigen::MatrixXd compute_correction(
        const Eigen::MatrixXd& embeddings,
        const std::vector<int32>& batch_labels,
        int reference_batch);

    // Compute correction with scheduler config (selects full/chunked)
    Eigen::MatrixXd compute_correction(
        const Eigen::MatrixXd& embeddings,
        const std::vector<int32>& batch_labels,
        int reference_batch,
        const ChunkConfig& cfg,
        int n_threads);

    // Apply Gaussian kernel smoothing to correction vectors (original)
    Eigen::MatrixXd smooth_correction(
        const Eigen::MatrixXd& query_emb,
        const Eigen::MatrixXd& raw_correction,
        double sigma);

    // Select full-matrix or chunked smoothing based on config
    Eigen::MatrixXd smooth_correction_adaptive(
        const Eigen::MatrixXd& query_emb,
        const Eigen::MatrixXd& raw_correction,
        double sigma,
        const ChunkConfig& cfg,
        int n_threads);

    // Chunked Gaussian kernel smoothing: processes target cells in blocks
    Eigen::MatrixXd smooth_correction_chunked(
        const Eigen::MatrixXd& query_emb,
        const Eigen::MatrixXd& raw_correction,
        double sigma,
        int64 chunk_size,
        int n_threads);
};

} // namespace sclean

#endif // SCLEAN_INTEGRATION_OPERATOR_H
