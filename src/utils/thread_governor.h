#ifndef SCLEAN_THREAD_GOVERNOR_H
#define SCLEAN_THREAD_GOVERNOR_H

#include "utils/resource_monitor.h"
#include <algorithm>

namespace sclean {

class ThreadGovernor {
public:
    ThreadGovernor() : physical_cores_(ResourceMonitor::detect_physical_cores()) {}

    int physical_cores() const { return physical_cores_; }

    // Adjust requested thread count based on current resource snapshot.
    // Returns the recommended thread count (≥1).
    //
    // Adjustment rules (applied in order):
    //
    // Rule 1 - Cap at physical cores:
    //   Logical/hyperthreaded cores ignored because Eigen + HDF5 are memory-bandwidth
    //   bound, not compute-bound, on typical single-cell data.
    //
    // Rule 2 - CPU contention backoff:
    //   When system load average exceeds 80% of physical cores, reduce thread count
    //   by the estimated number of busy cores. Prevents over-subscription when
    //   background processes (browser, IDE, other R sessions) compete for CPU.
    //
    // Rule 3 - Memory pressure halving:
    //   Each thread adds ~64 MB of HDF5 buffer overhead (PER_THREAD_RESERVE_BYTES).
    //   When memory pressure exceeds 0.7, halve thread count (floor 1).
    //
    // Rule 4 - Diminishing returns at 2 threads:
    //   With only 2 threads and moderate memory pressure (>0.5), the parallel speedup
    //   from the second thread is often negated by memory contention — HDF5 read
    //   bandwidth is the bottleneck. Single-threading avoids thrashing.
    //
    // Rule 5 - Emergency single-thread:
    //   When both memory and CPU pressure exceed 90%, force single-threaded operation
    //   regardless of what was requested.
    int adjust(int requested, const ResourceSnapshot& snap) const {
        if (requested < 1) requested = 1;

        int n = std::min(requested, physical_cores_);

        if (snap.cpu_load_1min > physical_cores_ * 0.8) {
            int busy_cores = static_cast<int>(std::ceil(snap.cpu_load_1min));
            n = std::max(1, physical_cores_ - busy_cores);
        }

        if (snap.memory_pressure > 0.7) {
            n = std::max(1, n / 2);
        }

        if (n == 2 && snap.memory_pressure > 0.5) {
            n = 1;
        }

        if (snap.memory_pressure > 0.9 && snap.cpu_pressure > 0.9) {
            n = 1;
        }

        return n;
    }

private:
    int physical_cores_;
};

} // namespace sclean

#endif // SCLEAN_THREAD_GOVERNOR_H
