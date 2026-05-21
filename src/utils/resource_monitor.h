#ifndef SCLEAN_RESOURCE_MONITOR_H
#define SCLEAN_RESOURCE_MONITOR_H

#include "scLean_types.h"
#include "utils/chunk_scheduler.h"  // for Bottleneck enum
#include <cstdint>
#include <algorithm>
#include <cmath>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#elif defined(_WIN32)
#include <windows.h>
#else
#include <sys/sysinfo.h>
#endif

#ifdef __unix__
#include <stdlib.h>  // getloadavg
#endif

namespace sclean {

struct ResourceSnapshot {
    // Memory (bytes)
    int64 total_ram = 0;
    int64 free_ram = 0;
    int64 current_rss = 0;
    int64 available_ram = 0;     // conservative: free_ram - OS_RESERVE

    // CPU
    double cpu_load_1min = 0.0;
    double cpu_load_5min = 0.0;
    int    physical_cores = 1;
    int    available_cores = 1;

    // Derived pressure ratios
    double memory_pressure = 0.0;  // current_rss / available_ram
    double cpu_pressure = 0.0;     // cpu_load_1min / physical_cores
};

class ResourceMonitor {
public:
    ResourceMonitor() {
        physical_cores_ = detect_physical_cores();
    }

    // Collect current resource snapshot
    ResourceSnapshot snapshot() {
        ResourceSnapshot s;
        s.total_ram      = detect_total_ram();
        s.free_ram       = detect_free_ram();
        s.current_rss    = current_rss_bytes();
        s.physical_cores = physical_cores_;

        // Conservative available: free - OS headroom
        s.available_ram = std::max(
            static_cast<int64>(MIN_AVAILABLE_RAM_BYTES),
            s.free_ram - OS_RESERVE_BYTES);

        // CPU load
#ifdef __unix__
        double loads[3];
        if (getloadavg(loads, 3) != -1) {
            s.cpu_load_1min = loads[0];
            s.cpu_load_5min = loads[1];
        }
#elif defined(_WIN32)
        // Windows: approximate via system times (simplified)
        s.cpu_load_1min = 0.0;
        s.cpu_load_5min = 0.0;
#endif

        // Available cores: physical - estimated busy cores
        s.available_cores = std::max(1,
            physical_cores_ - static_cast<int>(std::ceil(s.cpu_load_1min)));

        // Pressure ratios
        s.memory_pressure = (s.available_ram > 0)
            ? static_cast<double>(s.current_rss) / static_cast<double>(s.available_ram)
            : 1.0;
        s.cpu_pressure = (physical_cores_ > 0)
            ? s.cpu_load_1min / static_cast<double>(physical_cores_)
            : 0.0;

        return s;
    }

    // Classify bottleneck from snapshot
    static Bottleneck classify(const ResourceSnapshot& s) {
        if (s.memory_pressure > 0.7 && s.cpu_pressure > 0.7) return Bottleneck::BothBound;
        if (s.memory_pressure > 0.7) return Bottleneck::MemoryBound;
        if (s.cpu_pressure > 0.7) return Bottleneck::ComputeBound;
        return Bottleneck::None;
    }

    // Worst-case available RAM: free - rss - OS_reserve - thread_reserve
    static int64 worst_case_available_ram(const ResourceSnapshot& s, int n_threads) {
        if (n_threads < 1) n_threads = 1;
        int64 available = s.free_ram - s.current_rss - OS_RESERVE_BYTES
                          - n_threads * PER_THREAD_RESERVE_BYTES;
        // High memory pressure → apply additional 0.5× haircut
        if (s.memory_pressure > 0.8) {
            available = static_cast<int64>(available * 0.5);
        }
        if (available < MIN_AVAILABLE_RAM_BYTES) {
            available = MIN_AVAILABLE_RAM_BYTES;
        }
        return available;
    }

    // --- Core count (public, used by ThreadGovernor) ---

    static int detect_physical_cores() {
#ifdef __APPLE__
        int cores;
        size_t sz = sizeof(cores);
        if (sysctlbyname("hw.physicalcpu", &cores, &sz, nullptr, 0) == 0) return std::max(1, cores);
#elif defined(_WIN32)
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return std::max(1, static_cast<int>(si.dwNumberOfProcessors));
#else
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        return std::max(1, static_cast<int>(n));
#endif
        return 1;
    }

    // --- Raw probes (cross-platform) ---

    static int64 detect_total_ram() {
#ifdef __APPLE__
        int64 ram;
        size_t sz = sizeof(ram);
        if (sysctlbyname("hw.memsize", &ram, &sz, nullptr, 0) == 0) return ram;
#elif defined(_WIN32)
        MEMORYSTATUSEX s; s.dwLength = sizeof(s);
        if (GlobalMemoryStatusEx(&s)) return static_cast<int64>(s.ullTotalPhys);
#else
        struct sysinfo info;
        if (sysinfo(&info) == 0) return static_cast<int64>(info.totalram) * info.mem_unit;
#endif
        return 8LL * 1024 * 1024 * 1024;
    }

    static int64 detect_free_ram() {
        int64 free_ram = 0;
#ifdef __APPLE__
        mach_port_t host = mach_host_self();
        vm_size_t page_size;
        host_page_size(host, &page_size);
        vm_statistics64_data_t vm_stat;
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm_stat, &count) == KERN_SUCCESS) {
            free_ram = static_cast<int64>(vm_stat.free_count + vm_stat.inactive_count) * page_size;
        }
        mach_port_deallocate(mach_task_self(), host);
#elif defined(_WIN32)
        MEMORYSTATUSEX s; s.dwLength = sizeof(s);
        if (GlobalMemoryStatusEx(&s)) free_ram = static_cast<int64>(s.ullAvailPhys);
#else
        struct sysinfo info;
        if (sysinfo(&info) == 0) free_ram = static_cast<int64>(info.freeram) * info.mem_unit;
#endif
        int64 total = detect_total_ram();
        int64 floor_val = static_cast<int64>(total * 0.2);
        if (free_ram < floor_val) free_ram = floor_val;
        return free_ram;
    }

    static int64 current_rss_bytes() {
#ifdef __APPLE__
        struct task_basic_info_64 info;
        mach_msg_type_number_t count = TASK_BASIC_INFO_64_COUNT;
        if (task_info(mach_task_self(), TASK_BASIC_INFO_64,
                      reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
            return static_cast<int64>(info.resident_size);
        }
#elif defined(_WIN32)
        PROCESS_MEMORY_COUNTERS pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
            return static_cast<int64>(pmc.WorkingSetSize);
        }
#else
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
            return static_cast<int64>(usage.ru_maxrss) * 1024;
        }
#endif
        return 0;
    }

private:
    int physical_cores_;
};

} // namespace sclean

#endif // SCLEAN_RESOURCE_MONITOR_H
