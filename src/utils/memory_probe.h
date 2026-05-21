#ifndef SCLEAN_MEMORY_PROBE_H
#define SCLEAN_MEMORY_PROBE_H

#include <cstdint>

#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace sclean {

using std::int64_t;

#if defined(__APPLE__)

inline int64_t current_rss_bytes() {
    task_basic_info_data_t info;
    mach_msg_type_number_t count = TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
        return 0;
    }
    return static_cast<int64_t>(info.resident_size);
}

#elif defined(_WIN32)

inline int64_t current_rss_bytes() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return 0;
    }
    return static_cast<int64_t>(pmc.WorkingSetSize);
}

#else // Linux

inline int64_t current_rss_bytes() {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
    return static_cast<int64_t>(ru.ru_maxrss) * 1024; // ru_maxrss is in KB on Linux
}

#endif

#if defined(__APPLE__) || defined(__linux__)

#include <time.h>

inline int64_t wall_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000000LL +
           static_cast<int64_t>(ts.tv_nsec);
}

#elif defined(_WIN32)

inline int64_t wall_time_ns() {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return static_cast<int64_t>(
        static_cast<double>(counter.QuadPart) /
        static_cast<double>(freq.QuadPart) * 1e9);
}

#else

// Fallback: std::chrono (less precise but always available)
#include <chrono>

inline int64_t wall_time_ns() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               now.time_since_epoch()).count();
}

#endif

} // namespace sclean

#endif // SCLEAN_MEMORY_PROBE_H
