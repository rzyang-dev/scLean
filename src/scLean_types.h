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

// --- Memory budget constants ---
// Tuned empirically on 8 GB MacBook Pro via Phase 3A benchmarking.
//
// SAFETY_FACTOR = 0.5
//   Only half of detected free RAM is assumed usable. Rationale: (1) HDF5 decompression
//   buffers and Eigen temporaries can spike 2× above the steady-state estimate; (2) macOS
//   aggressively caches filesystem I/O in "inactive" memory, which vm_stat reports as
//   free but cannot be reclaimed instantly by malloc.
//
// MIN_CHUNK_ROWS = 128
//   Floor for row-chunked operations (Scale, PCA matvec). Below 128 rows, the per-chunk
//   HDF5 metadata read overhead dominates the actual data transfer cost. Empirically the
//   I/O-to-compute ratio stays below 5% at this minimum.
//
// MIN_CHUNK_COLS = 1024
//   Floor for column-chunked operations (Normalize, VST). Column access in CSC requires
//   binary search over indptr per chunk, so larger minimum chunks amortize that fixed cost.
//
// HDF5_CHUNK_SIZE_1D = 1<<20 (1M entries)
//   HDF5 internal chunking for 1D datasets (feature/cell names, mean vectors, stdev).
//   1M entries balances per-read overhead against decompression buffer size.
//   Reference: HDF Group recommends chunk sizes between 10 KB and 1 MB.
//
// MAX_DENSE_CHUNK_BYTES = 128 MB
//   Hard cap on any single dense read buffer. Prevents a single chunk from consuming
//   more than ~128 MB even on a machine with abundant RAM, leaving headroom for other
//   processes and Eigen intermediates. Settable at runtime via scLean.max_dense_chunk_mb.
//
// OS_RESERVE_BYTES = 512 MB
//   Reserved for the OS (kernel buffers, window server, background services).
//   Subtracted from detected free RAM to produce the "available_ram" used by the
//   scheduler. This is NOT subtracted from a user-set max_ram cap.
//
// PER_THREAD_RESERVE_BYTES = 64 MB
//   Each OpenMP thread gets its own HDF5 file handle and read buffer. This reserve
//   accounts for per-thread HDF5 internal buffers, thread stack, and local Eigen
//   temporaries.
//
// MIN_AVAILABLE_RAM_BYTES = 128 MB
//   Absolute floor — if worst-case available RAM after all reserves drops below this,
//   we clamp to 128 MB rather than returning zero (which would cause division-by-zero
//   or zero-sized chunks downstream).
constexpr double SAFETY_FACTOR = 0.5;
constexpr int64 MIN_CHUNK_ROWS = 128;
constexpr int64 MIN_CHUNK_COLS = 1024;
constexpr int64 HDF5_CHUNK_SIZE_1D = (1 << 20);
constexpr int64 MAX_DENSE_CHUNK_BYTES = 128LL * 1024 * 1024;
constexpr int64 OS_RESERVE_BYTES = 512LL * 1024 * 1024;
constexpr int64 PER_THREAD_RESERVE_BYTES = 64LL * 1024 * 1024;
constexpr int64 MIN_AVAILABLE_RAM_BYTES = 128LL * 1024 * 1024;

} // namespace sclean

#endif // SCLEAN_TYPES_H
