#include "chunk_platform.h"
#include <algorithm>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#elif defined(_WIN32)
#include <windows.h>
#else
#include <sys/sysinfo.h>
#endif

namespace sclean {
namespace detail {

// --- Per-platform system RAM detection ---

#ifdef __APPLE__
static int64 detect_system_ram_macos() {
    int64 ram;
    size_t sz = sizeof(ram);
    if (sysctlbyname("hw.memsize", &ram, &sz, nullptr, 0) == 0) return ram;
    return 8LL * 1024 * 1024 * 1024;
}
#elif defined(_WIN32)
static int64 detect_system_ram_windows() {
    MEMORYSTATUSEX s; s.dwLength = sizeof(s);
    if (GlobalMemoryStatusEx(&s)) return static_cast<int64>(s.ullTotalPhys);
    return 8LL * 1024 * 1024 * 1024;
}
#else
static int64 detect_system_ram_linux() {
    struct sysinfo info;
    if (sysinfo(&info) == 0) return static_cast<int64>(info.totalram) * info.mem_unit;
    return 8LL * 1024 * 1024 * 1024;
}
#endif

// --- Per-platform free RAM detection ---

#ifdef __APPLE__
static int64 detect_free_ram_macos() {
    mach_port_t host = mach_host_self();
    vm_size_t page_size;
    host_page_size(host, &page_size);
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    int64 free_ram = 0;
    if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm_stat, &count) == KERN_SUCCESS) {
        free_ram = static_cast<int64>(vm_stat.free_count + vm_stat.inactive_count) * page_size;
    }
    mach_port_deallocate(mach_task_self(), host);
    return free_ram;
}
#elif defined(_WIN32)
static int64 detect_free_ram_windows() {
    MEMORYSTATUSEX s; s.dwLength = sizeof(s);
    if (GlobalMemoryStatusEx(&s)) return static_cast<int64>(s.ullAvailPhys);
    return 0;
}
#else
static int64 detect_free_ram_linux() {
    struct sysinfo info;
    if (sysinfo(&info) == 0) return static_cast<int64>(info.freeram) * info.mem_unit;
    return 0;
}
#endif

// --- Unified wrappers ---

int64 detect_system_ram() {
#ifdef __APPLE__
    return detect_system_ram_macos();
#elif defined(_WIN32)
    return detect_system_ram_windows();
#else
    return detect_system_ram_linux();
#endif
}

int64 detect_free_ram() {
    int64 free_ram = 0;
#ifdef __APPLE__
    free_ram = detect_free_ram_macos();
#elif defined(_WIN32)
    free_ram = detect_free_ram_windows();
#else
    free_ram = detect_free_ram_linux();
#endif
    // Floor at 20% of total RAM to avoid excessively small chunks
    // when the system reports near-zero free memory (e.g., due to disk cache).
    int64 total = detect_system_ram();
    int64 floor_val = static_cast<int64>(total * 0.2);
    if (free_ram < floor_val) free_ram = floor_val;
    return free_ram;
}

} // namespace detail
} // namespace sclean
