#ifndef SCLEAN_VST_OPERATOR_H
#define SCLEAN_VST_OPERATOR_H

#include <cstdint>
#include <vector>
#include <string>
#include "utils/chunk_scheduler.h"

namespace sclean {

class HDF5File;

struct VSTResult {
    std::vector<double> gene_means;
    std::vector<double> gene_variances;
    std::vector<double> vst_variances;     // standardized variance
    std::vector<int8_t> variable_features;  // boolean mask
    int n_variable;
};

class VSTOperator {
public:
    VSTOperator(int n_top_features = 2000,
                double loess_span = 0.3,
                int n_bins = 20);

    VSTResult run(HDF5File* file,
                  const std::string& input_group,
                  int64 n_genes, int64 n_cells,
                  ChunkScheduler& scheduler,
                  int n_threads = 1);

private:
    int n_top_features_;
    double loess_span_;
    int n_bins_;

    // Pass 1: per-gene mean and variance (column-chunked dense Welford)
    void compute_mean_variance(HDF5File* file,
                                const std::string& input_group,
                                int64 n_genes, int64 n_cells,
                                std::vector<double>& means,
                                std::vector<double>& variances,
                                ChunkScheduler& scheduler,
                                int n_threads);

    // Sparse variant: reads CSC directly, no dense buffer.
    // Used when MemoryBound — trades some speed for ~90% less memory.
    void compute_mean_variance_sparse(HDF5File* file,
                                       const std::string& input_group,
                                       int64 n_genes, int64 n_cells,
                                       std::vector<double>& means,
                                       std::vector<double>& variances,
                                       ChunkScheduler& scheduler,
                                       int n_threads);

};

} // namespace sclean

#endif // SCLEAN_VST_OPERATOR_H
