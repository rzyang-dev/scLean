#ifndef SCLEAN_NORMALIZE_OPERATOR_H
#define SCLEAN_NORMALIZE_OPERATOR_H

#include <cstdint>
#include <vector>
#include <string>
#include "core/disk_matrix.h"
#include "utils/chunk_scheduler.h"

namespace sclean {

enum class NormalizeMethod { LogNormalize = 0, CLR = 1, RelativeCounts = 2 };

struct NormalizeResult {
    std::vector<double> size_factors;
    int64 n_cells;
    int64 n_genes;
};

class HDF5File;
class HDF5SparseWriter;

class NormalizeOperator {
public:
    NormalizeOperator(NormalizeMethod method = NormalizeMethod::LogNormalize,
                      double scale_factor = 10000.0,
                      bool do_pseudocount = true);

    // Run normalization: reads sparse columns from input HDF5, writes
    // normalized data to output via HDF5SparseWriter.
    NormalizeResult run(HDF5File* file,
                        const std::string& input_group,
                        const std::string& output_group,
                        ChunkScheduler& scheduler,
                        int n_threads = 1);

private:
    NormalizeMethod method_;
    double scale_factor_;
    bool do_pseudocount_;

    std::vector<double> compute_size_factors(
        HDF5File* file, const std::string& input_group,
        int64 n_cells, int64 n_rows, int64 nnz,
        ChunkScheduler& scheduler, int n_threads);

    std::vector<double> compute_libsize_chunked(
        HDF5File* file, const std::string& input_group,
        int64 n_cells, int64 n_rows,
        ChunkScheduler& scheduler, int n_threads);

    std::vector<double> compute_geometric_means_chunked(
        HDF5File* file, const std::string& input_group,
        int64 n_cells, int64 n_rows,
        ChunkScheduler& scheduler, int n_threads);
};

} // namespace sclean

#endif // SCLEAN_NORMALIZE_OPERATOR_H
