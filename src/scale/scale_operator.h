#ifndef SCLEAN_SCALE_OPERATOR_H
#define SCLEAN_SCALE_OPERATOR_H

#include <cstdint>
#include <vector>
#include <string>
#include "utils/chunk_scheduler.h"

namespace sclean {

class HDF5File;
class HDF5SparseWriter;

struct ScaleResult {
    std::vector<double> gene_means;
    std::vector<double> gene_sds;
    int64 n_genes, n_cells;
};

class ScaleOperator {
public:
    ScaleOperator(bool do_scale = true, bool do_center = true,
                  const std::vector<std::vector<double>>* vars_to_regress = nullptr);

    ScaleResult run(HDF5File* file,
                    const std::string& input_group,
                    const std::string& output_group,
                    ChunkScheduler& scheduler,
                    int n_threads = 1);

    // Compute only stats (no residual writing)
    ScaleResult compute_stats(HDF5File* file,
                               const std::string& input_group,
                               int64 n_genes, int64 n_cells,
                               ChunkScheduler& scheduler,
                               int n_threads);

private:
    bool do_scale_;
    bool do_center_;
    const std::vector<std::vector<double>>* vars_to_regress_;

    void compute_moments_row_chunked(HDF5File* file,
                                      const std::string& input_group,
                                      int64 n_genes, int64 n_cells,
                                      std::vector<double>& means,
                                      std::vector<double>& sds,
                                      ChunkScheduler& scheduler,
                                      int n_threads);
};

} // namespace sclean

#endif // SCLEAN_SCALE_OPERATOR_H
