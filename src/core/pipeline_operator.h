#ifndef SCLEAN_PIPELINE_OPERATOR_H
#define SCLEAN_PIPELINE_OPERATOR_H

#include <string>

namespace sclean {

// Forward declarations
class DataSource;
class ChunkScheduler;

// Result status returned by every operator
struct OpResult {
    int status;              // 0 = success, non-zero = error
    std::string message;     // human-readable status/error message

    static OpResult success(const std::string& msg = "") {
        return {0, msg};
    }
    static OpResult error(const std::string& msg) {
        return {-1, msg};
    }
};

// Common interface: every pipeline operator accepts a DataSource and
// ChunkScheduler, performs its computation, and returns an OpResult.
// Results are written to HDF5 via the DataSource, not returned inline.
class PipelineOperator {
public:
    virtual ~PipelineOperator() = default;

    // Execute the pipeline step.
    // @param source     Data access layer (HDF5-backed)
    // @param scheduler  Memory-aware chunk planner
    // @param n_threads  Number of OpenMP threads to use
    // @return OpResult indicating success or failure
    virtual OpResult run(DataSource& source,
                         ChunkScheduler& scheduler,
                         int n_threads) = 0;

    // Human-readable name for logging/progress
    virtual const char* name() const = 0;
};

} // namespace sclean

#endif // SCLEAN_PIPELINE_OPERATOR_H
