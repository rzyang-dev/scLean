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
    int adjust(int requested, const ResourceSnapshot& snap) const {
        if (requested < 1) requested = 1;

        // 1. Never exceed physical cores
        int n = std::min(requested, physical_cores_);

        // 2. CPU contention: back off when load exceeds available cores
        if (snap.cpu_load_1min > physical_cores_ * 0.8) {
            int busy_cores = static_cast<int>(std::ceil(snap.cpu_load_1min));
            n = std::max(1, physical_cores_ - busy_cores);
        }

        // 3. Memory pressure: each thread carries ~64 MB reserve
        if (snap.memory_pressure > 0.7) {
            n = std::max(1, n / 2);
        }

        // 4. Diminishing returns: 2 threads with moderate mem pressure → 1
        if (n == 2 && snap.memory_pressure > 0.5) {
            n = 1;
        }

        // 5. Both resources constrained: single-thread
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
