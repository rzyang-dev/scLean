#include "chunk_scheduler.h"
#include "chunk_platform.h"
#include "resource_monitor.h"
#include "thread_governor.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sclean {

// Static member initialization
int64 ChunkScheduler::max_dense_chunk_bytes_ = MAX_DENSE_CHUNK_BYTES;

// --- Operation multipliers ---
//
// Each multiplier M represents the peak temporary-memory-to-read-buffer ratio.
// The multiplier captures: 1x for the read buffer itself + additional memory
// for intermediate computation buffers and results.
//
// Breakdown:
//   Normalize (M=3):    1x read + 1x log/RLE transform result + 1x Welford accumulators
//   Scale (M=6):        1x read + 1x centered result + 4x regression design matrix intermediates
//   ScaleSimple (M=2):  1x read + 1x centered result (no regression, less intermediate state)
//   VST (M=4):          1x read + 1x binned means + 1x binned variances + 1x fit temporary
//   PCA (M=2.5):        1x read + 0.5x matvec output + 1x Lanczos vector state
//   FindMarkers (M=4):  1x read + 1x group-split arrays + 1x rank buffers + 1x p-value workspace
//   FindNeighbors (M=3): 1x read + 1x distance matrix + 1x index/heap buffers
//   Integration (M=5):  1x read + 1x MNN distance matrix + 1x correction vectors
//                       + 1x smoothed output + 1x Gaussian kernel workspace
//
// These are conservative estimates tuned on gene expression data (sparse, ~5-10% dense).
// Dense data may need higher multipliers; the 128 MB hard cap on dense chunks (set via
// scLean.max_dense_chunk_mb) provides a safety net.

double ChunkScheduler::op_multiplier(OperationType op) {
    switch (op) {
        case OperationType::Normalize:     return 3.0;
        case OperationType::Scale:         return 6.0;
        case OperationType::ScaleSimple:   return 2.0;
        case OperationType::VST:           return 4.0;
        case OperationType::PCA:           return 2.5;
        case OperationType::FindMarkers:   return 4.0;
        case OperationType::FindNeighbors: return 3.0;
        case OperationType::Integration:   return 5.0;
    }
    return 3.0;
}

ChunkAxis ChunkScheduler::op_axis(OperationType op) {
    switch (op) {
        case OperationType::Normalize:
        case OperationType::VST:
            return ChunkAxis::Columns;
        case OperationType::FindMarkers:
        case OperationType::Integration:
            return ChunkAxis::Rows;
        case OperationType::Scale:
        case OperationType::ScaleSimple:
        case OperationType::FindNeighbors:
            return ChunkAxis::Rows;
        case OperationType::PCA:
            return ChunkAxis::Rows;  // row-chunked matvec
    }
    return ChunkAxis::Columns;
}

// --- Constructor ---

ChunkScheduler::ChunkScheduler(int64 cap_bytes)
    : cap_(cap_bytes > 0 ? cap_bytes : 0)
    , available_ram_(cap_bytes > 0
        ? std::min(detail::detect_free_ram(), cap_bytes)
        : detail::detect_free_ram())
    , chunk_override_(-1)
    , retry_count_(0)
    , last_bottleneck_(Bottleneck::None) {}

void ChunkScheduler::refresh_available_ram() {
    int64 free = detail::detect_free_ram();
    available_ram_ = cap_ > 0 ? std::min(free, cap_) : free;
}

int64 ChunkScheduler::worst_case_available_ram(int64 free_ram, int64 current_rss, int n_threads) {
    if (n_threads < 1) n_threads = 1;
    int64 available = free_ram - current_rss - OS_RESERVE_BYTES
                      - n_threads * PER_THREAD_RESERVE_BYTES;
    if (available < MIN_AVAILABLE_RAM_BYTES) {
        available = MIN_AVAILABLE_RAM_BYTES;
    }
    return available;
}

void ChunkScheduler::set_chunk_size(int64 rows_or_cols) {
    chunk_override_ = rows_or_cols;
}

void ChunkScheduler::clear_override() {
    chunk_override_ = -1;
    retry_count_ = 0;
}

// --- Core scheduling logic (shared implementation) ---
//
// Algorithm:
// 1. If chunk_override is set (>0), use it directly -- no auto-sizing.
// 2. Compute usable budget B = usable_ram - extra_buffers - parallel_overhead,
//    where parallel_overhead = compute_buf * (n_threads - 1).
// 3. chunk_size = floor(B / (stride * elem_size * M)), clamped to min/max bounds.
// 4. Check fits_in_memory: full matrix + multiplier fits in B.
// 5. Hard cap: if dense read buffer exceeds max_dense_bytes (128 MB default),
//    shrink chunk_size. This safety cap prevents single-chunk OOM.
//
// Column-chunked (axis=Columns):
//   stride = n_rows, chunk_size in columns.
//   Used for: Normalize, VST
//
// Row-chunked (axis=Rows):
//   stride = n_cols, chunk_size in rows.
//   Used for: Scale, PCA, FindMarkers, FindNeighbors, Integration
//
// Note: ChunkAxis::Rows means the operation processes rows, but the actual
// chunk dimension is also measured in rows (element count = stride * chunk_size).

static ChunkConfig schedule_impl(int64 R, int64 C, ChunkAxis axis,
                                  double M, int64 B, int64 n_threads,
                                  int64 extra_total, int64 chunk_override,
                                  int64 max_dense_bytes) {
    if (n_threads < 1) n_threads = 1;
    const int64 elem_size = static_cast<int64>(sizeof(double));

    ChunkConfig cfg;
    cfg.axis = axis;
    cfg.fits_in_memory = false;
    cfg.bottleneck = Bottleneck::None;
    cfg.dense_buffer_bytes = 0;
    cfg.compute_buffer_bytes = 0;

    if (chunk_override > 0) {
        cfg.chunk_size = chunk_override;
        cfg.n_chunks = (axis == ChunkAxis::Rows)
            ? (R + cfg.chunk_size - 1) / cfg.chunk_size
            : (C + cfg.chunk_size - 1) / cfg.chunk_size;
        int64 stride = (axis == ChunkAxis::Columns) ? R : C;
        cfg.dense_buffer_bytes = stride * cfg.chunk_size * elem_size;
        cfg.compute_buffer_bytes = stride * static_cast<int64>(M) * elem_size;
        cfg.memory_per_chunk = std::max(cfg.dense_buffer_bytes, cfg.compute_buffer_bytes);
        return cfg;
    }

    if (axis == ChunkAxis::Columns) {
        int64 stride = R;
        // compute_buf is for intermediates (Welford accumulators, etc.)
        int64 compute_buf = stride * static_cast<int64>(M) * elem_size;
        int64 parallel_overhead = compute_buf * (n_threads - 1);
        int64 available = B - extra_total - parallel_overhead;

        if (available <= 0) {
            cfg.chunk_size = MIN_CHUNK_COLS;
            cfg.n_chunks = (C + MIN_CHUNK_COLS - 1) / MIN_CHUNK_COLS;
            cfg.dense_buffer_bytes = stride * cfg.chunk_size * elem_size;
            cfg.compute_buffer_bytes = compute_buf;
            cfg.memory_per_chunk = std::max(cfg.dense_buffer_bytes, cfg.compute_buffer_bytes);
            return cfg;
        }

        // Compute chunk_size based on available RAM
        cfg.chunk_size = std::max(static_cast<int64>(MIN_CHUNK_COLS),
            static_cast<int64>(available / (stride * elem_size * std::max(1.0, M))));
        cfg.chunk_size = std::min(cfg.chunk_size, C);

        // fits_in_memory: full matrix fits with M multiplier
        int64 total_memory = R * C * elem_size * static_cast<int64>(M);
        if (total_memory <= B) {
            cfg.chunk_size = C;
            cfg.fits_in_memory = true;
        }

        // Hard cap: dense read buffer must not exceed MAX_DENSE_CHUNK_BYTES
        int64 dense_buf = stride * cfg.chunk_size * elem_size;
        if (dense_buf > max_dense_bytes) {
            cfg.chunk_size = max_dense_bytes / (stride * elem_size);
            cfg.chunk_size = std::max(cfg.chunk_size, static_cast<int64>(MIN_CHUNK_COLS));
            cfg.fits_in_memory = false;  // capped -> not truly in-memory
        }

        cfg.n_chunks = (C + cfg.chunk_size - 1) / cfg.chunk_size;
        cfg.dense_buffer_bytes = stride * cfg.chunk_size * elem_size;
        cfg.compute_buffer_bytes = compute_buf;
    } else {
        int64 stride = C;
        int64 compute_buf = stride * static_cast<int64>(M) * elem_size;
        int64 parallel_overhead = compute_buf * (n_threads - 1);
        int64 available = B - extra_total - parallel_overhead;

        if (available <= 0) {
            cfg.chunk_size = MIN_CHUNK_ROWS;
            cfg.n_chunks = (R + MIN_CHUNK_ROWS - 1) / MIN_CHUNK_ROWS;
            cfg.dense_buffer_bytes = stride * cfg.chunk_size * elem_size;
            cfg.compute_buffer_bytes = compute_buf;
            cfg.memory_per_chunk = std::max(cfg.dense_buffer_bytes, cfg.compute_buffer_bytes);
            return cfg;
        }

        cfg.chunk_size = std::max(static_cast<int64>(MIN_CHUNK_ROWS),
            static_cast<int64>(available / (stride * elem_size * std::max(1.0, M))));
        cfg.chunk_size = std::min(cfg.chunk_size, R);

        int64 total_memory = R * C * elem_size * static_cast<int64>(M);
        if (total_memory <= B) {
            cfg.chunk_size = R;
            cfg.fits_in_memory = true;
        }

        // Hard cap
        int64 dense_buf = stride * cfg.chunk_size * elem_size;
        if (dense_buf > max_dense_bytes) {
            cfg.chunk_size = max_dense_bytes / (stride * elem_size);
            cfg.chunk_size = std::max(cfg.chunk_size, static_cast<int64>(MIN_CHUNK_ROWS));
            cfg.fits_in_memory = false;
        }

        cfg.n_chunks = (R + cfg.chunk_size - 1) / cfg.chunk_size;
        cfg.dense_buffer_bytes = stride * cfg.chunk_size * elem_size;
        cfg.compute_buffer_bytes = compute_buf;
    }

    cfg.memory_per_chunk = std::max(cfg.dense_buffer_bytes, cfg.compute_buffer_bytes);

    return cfg;
}

// --- Max dense chunk bytes (from R option or default) ---

void set_max_dense_chunk_bytes(int64 bytes) {
    if (bytes > 0) ChunkScheduler::max_dense_chunk_bytes_ = bytes;
}

int64 get_max_dense_chunk_bytes() {
    return ChunkScheduler::max_dense_chunk_bytes_;
}

// --- schedule() implementations ---

ChunkConfig ChunkScheduler::schedule(
    const DiskMatrix& matrix,
    OperationType op_type,
    int64 n_threads,
    const std::vector<int64>& extra_buffers) {

    int64 extra_total = 0;
    for (auto eb : extra_buffers) extra_total += eb;

    ChunkAxis axis = op_axis(op_type);
    double M = op_multiplier(op_type);

    // Refresh resource snapshot and classify bottleneck
    ResourceMonitor monitor;
    auto snap = monitor.snapshot();
    last_bottleneck_ = ResourceMonitor::classify(snap);

    // Adjust thread count based on resource availability
    ThreadGovernor governor;
    int64 adjusted_threads = governor.adjust(static_cast<int>(n_threads), snap);

    // Compute usable RAM: worst-case available, bounded by usable_ram()
    int64 worst = ChunkScheduler::worst_case_available_ram(
        snap.free_ram, snap.current_rss, static_cast<int>(adjusted_threads));
    int64 B = std::min(usable_ram(), worst);

    ChunkConfig cfg = schedule_impl(matrix.n_rows(), matrix.n_cols(),
                                     axis, M, B, adjusted_threads, extra_total,
                                     chunk_override_, max_dense_chunk_bytes_);

    // Bottleneck-driven adjustments
    if (last_bottleneck_ == Bottleneck::MemoryBound ||
        last_bottleneck_ == Bottleneck::BothBound) {
        // Further shrink chunks under memory pressure
        int64 stride = (axis == ChunkAxis::Columns) ? matrix.n_rows() : matrix.n_cols();
        int64 safe_chunk = worst / (stride * static_cast<int64>(sizeof(double)));
        safe_chunk = std::max(static_cast<int64>(
            axis == ChunkAxis::Columns ? MIN_CHUNK_COLS : MIN_CHUNK_ROWS), safe_chunk);
        cfg.chunk_size = std::min(cfg.chunk_size, safe_chunk);
        cfg.n_chunks = (axis == ChunkAxis::Columns)
            ? (matrix.n_cols() + cfg.chunk_size - 1) / cfg.chunk_size
            : (matrix.n_rows() + cfg.chunk_size - 1) / cfg.chunk_size;
        cfg.dense_buffer_bytes = stride * cfg.chunk_size * static_cast<int64>(sizeof(double));
        cfg.memory_per_chunk = std::max(cfg.dense_buffer_bytes, cfg.compute_buffer_bytes);
        cfg.fits_in_memory = false;
    }

    cfg.bottleneck = last_bottleneck_;

    // Use sparse-aware estimates when available
    if (axis == ChunkAxis::Columns && cfg.chunk_size != matrix.n_cols()) {
        int64 mem = matrix.memory_per_col_chunk(matrix.n_rows());
        if (mem > 0) cfg.memory_per_chunk = std::max(cfg.memory_per_chunk, mem);
    } else if (axis == ChunkAxis::Rows && cfg.chunk_size != matrix.n_rows()) {
        int64 mem = matrix.memory_per_row_chunk(matrix.n_cols());
        if (mem > 0) cfg.memory_per_chunk = std::max(cfg.memory_per_chunk, mem);
    }

    return cfg;
}

ChunkConfig ChunkScheduler::schedule(
    int64 n_rows, int64 n_cols,
    OperationType op_type,
    int64 n_threads,
    const std::vector<int64>& extra_buffers) {

    int64 extra_total = 0;
    for (auto eb : extra_buffers) extra_total += eb;

    // Refresh resource snapshot
    ResourceMonitor monitor;
    auto snap = monitor.snapshot();
    last_bottleneck_ = ResourceMonitor::classify(snap);

    ThreadGovernor governor;
    int64 adjusted_threads = governor.adjust(static_cast<int>(n_threads), snap);

    int64 worst = ChunkScheduler::worst_case_available_ram(
        snap.free_ram, snap.current_rss, static_cast<int>(adjusted_threads));
    int64 B = std::min(usable_ram(), worst);

    ChunkConfig cfg = schedule_impl(n_rows, n_cols, op_axis(op_type),
                                     op_multiplier(op_type), B,
                                     adjusted_threads, extra_total,
                                     chunk_override_, max_dense_chunk_bytes_);

    // Bottleneck-driven adjustments
    if (last_bottleneck_ == Bottleneck::MemoryBound ||
        last_bottleneck_ == Bottleneck::BothBound) {
        ChunkAxis axis = op_axis(op_type);
        int64 stride = (axis == ChunkAxis::Columns) ? n_rows : n_cols;
        int64 safe_chunk = worst / (stride * static_cast<int64>(sizeof(double)));
        int64 min_chunk = (axis == ChunkAxis::Columns) ? MIN_CHUNK_COLS : MIN_CHUNK_ROWS;
        safe_chunk = std::max(min_chunk, safe_chunk);
        cfg.chunk_size = std::min(cfg.chunk_size, safe_chunk);
        cfg.n_chunks = (axis == ChunkAxis::Columns)
            ? (n_cols + cfg.chunk_size - 1) / cfg.chunk_size
            : (n_rows + cfg.chunk_size - 1) / cfg.chunk_size;
        cfg.dense_buffer_bytes = stride * cfg.chunk_size * static_cast<int64>(sizeof(double));
        cfg.memory_per_chunk = std::max(cfg.dense_buffer_bytes, cfg.compute_buffer_bytes);
        cfg.fits_in_memory = false;
    }

    cfg.bottleneck = last_bottleneck_;

    return cfg;
}

// 3-level OOM recovery strategy:
// When a pipeline operator catches std::bad_alloc, it calls shrink_and_retry().
//
// Level 1 (retry 1): Halve the chunk size from its current value.
// Level 2 (retry 2): Halve again.
// Level 3 (retry 3): Halve again, down to floor of 128.
// Fail (retry > 3): return false -- caller should propagate the error.
//
// Each successful shrinkage sets the halved size as a chunk_override so that
// subsequent schedule() calls within the same operation use the smaller chunks.
// The override is cleared via clear_override() when the operation completes
// (or fails out entirely).
bool ChunkScheduler::shrink_and_retry(
    const DiskMatrix& matrix, OperationType op, int64 n_threads, ChunkConfig& config) {
    return shrink_and_retry(matrix.n_rows(), matrix.n_cols(), op, n_threads, config);
}

bool ChunkScheduler::shrink_and_retry(
    int64 n_rows, int64 n_cols, OperationType, int64, ChunkConfig& config) {
    retry_count_++;
    if (retry_count_ > 3) return false;

    config.chunk_size = std::max(static_cast<int64>(128), config.chunk_size / 2);
    int64 total = (config.axis == ChunkAxis::Rows) ? n_rows : n_cols;
    config.n_chunks = (total + config.chunk_size - 1) / config.chunk_size;
    config.fits_in_memory = false;
    config.bottleneck = Bottleneck::MemoryBound;
    int64 stride = (config.axis == ChunkAxis::Columns) ? n_rows : n_cols;
    config.dense_buffer_bytes = stride * config.chunk_size * static_cast<int64>(sizeof(double));
    config.compute_buffer_bytes = config.compute_buffer_bytes / 2;
    config.memory_per_chunk = std::max(config.dense_buffer_bytes, config.compute_buffer_bytes);

    // Also set as override so that subsequent schedule() calls use the shrunk size
    set_chunk_size(config.chunk_size);

    return true;
}

} // namespace sclean
