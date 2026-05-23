#ifndef SCLEAN_PARALLEL_H
#define SCLEAN_PARALLEL_H

#include <cstdint>
#include "scLean_types.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <Eigen/Core>

namespace sclean {

// Global thread count, settable from R via SetThreads().
// Default 1: safe for HDF5 builds without --enable-threadsafe.
// Users with thread-safe HDF5 can increase via SetThreads(n).
inline int64 g_num_threads = 1;

inline int64 get_num_threads() {
    return g_num_threads;
}

inline void set_num_threads(int64 n) {
    if (n < 1) n = 1;
    g_num_threads = n;
    // Prevent Eigen from spawning its own OpenMP threads which
    // conflict with external BLAS threading (especially on macOS).
    Eigen::setNbThreads(static_cast<int>(n));
#ifdef _OPENMP
    omp_set_num_threads(static_cast<int>(n));
#else
    (void)n;
#endif
}

inline int64 auto_threads(int64 available_ram, int64 mem_per_thread) {
    if (mem_per_thread <= 0) return 1;

    int64 max_cores = g_num_threads;
    int64 mem_limited = available_ram / mem_per_thread;
    if (mem_limited < 1) mem_limited = 1;
    int64 n = std::min(max_cores, mem_limited);
    return std::max(static_cast<int64>(1), n);
}

} // namespace sclean

#endif // SCLEAN_PARALLEL_H
