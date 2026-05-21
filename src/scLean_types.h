#ifndef SCLEAN_TYPES_H
#define SCLEAN_TYPES_H

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>

namespace sclean {

using int64 = int64_t;
using int32 = int32_t;

constexpr double SAFETY_FACTOR = 0.5;
constexpr int64 MIN_CHUNK_ROWS = 128;
constexpr int64 MIN_CHUNK_COLS = 1024;
constexpr int64 HDF5_CHUNK_SIZE_1D = (1 << 20);
constexpr int64 MAX_DENSE_CHUNK_BYTES = 128LL * 1024 * 1024;  // 128 MB hard cap
constexpr int64 OS_RESERVE_BYTES = 512LL * 1024 * 1024;       // 512 MB OS headroom
constexpr int64 PER_THREAD_RESERVE_BYTES = 64LL * 1024 * 1024; // 64 MB per thread
constexpr int64 MIN_AVAILABLE_RAM_BYTES = 128LL * 1024 * 1024; // 128 MB floor

} // namespace sclean

#endif // SCLEAN_TYPES_H
