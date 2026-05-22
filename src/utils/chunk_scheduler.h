#ifndef SCLEAN_CHUNK_SCHEDULER_H
#define SCLEAN_CHUNK_SCHEDULER_H

#include <cstdint>
#include <vector>
#include <string>
#include "matrix/disk_matrix.h"
#include "scLean_types.h"

namespace sclean {

enum class Bottleneck {
    None,           // Resources ample
    MemoryBound,    // memory_pressure > 0.7
    ComputeBound,   // cpu_pressure > 0.7 && memory_pressure < 0.5
    BothBound,      // Both constrained
};

enum class OperationType {
    Normalize,       // column-chunked, M=3
    Scale,           // row-chunked, M=6 (with regression)
    ScaleSimple,     // row-chunked, M=2 (centering only)
    VST,             // column-chunked, M=4
    PCA,             // implicit via matvec, M=2.5
    FindMarkers,     // column-chunked, M=4
    FindNeighbors,   // row-chunked (per-cell), M=3
    Integration,     // column-chunked, M=8
};

struct ChunkConfig {
    int64 chunk_size;           // rows or columns per chunk
    int64 n_chunks;             // total number of chunks
    ChunkAxis axis;             // which dimension is chunked
    int64 memory_per_chunk;     // estimated total bytes per chunk (max of dense + compute)
    int64 dense_buffer_bytes;   // actual dense read buffer: stride × chunk_size × 8
    int64 compute_buffer_bytes; // compute intermediates: stride × M × 8
    bool fits_in_memory;        // can the whole matrix be loaded at once?
    Bottleneck bottleneck;      // current resource bottleneck
};

class ChunkScheduler {
public:
    explicit ChunkScheduler(int64 available_ram_bytes = 0);

    ChunkConfig schedule(const DiskMatrix& matrix,
                         OperationType op_type,
                         int64 n_threads = 1,
                         const std::vector<int64>& extra_buffers = {});

    // Overload taking explicit dimensions (for direct HDF5 access)
    ChunkConfig schedule(int64 n_rows, int64 n_cols,
                         OperationType op_type,
                         int64 n_threads = 1,
                         const std::vector<int64>& extra_buffers = {});

    void refresh_available_ram();
    int64 available_ram() const { return available_ram_; }
    int64 user_cap() const { return cap_; }
    int64 usable_ram() const {
        return static_cast<int64>(available_ram_ * SAFETY_FACTOR);
    }

    // Bottleneck tracking (updated by schedule() on each call)
    Bottleneck current_bottleneck() const { return last_bottleneck_; }

    // Worst-case available RAM after OS + thread reserves
    static int64 worst_case_available_ram(int64 free_ram, int64 current_rss, int n_threads);

    void set_chunk_size(int64 rows_or_cols);
    void clear_override();

    // OOM recovery: halve chunk size and retry
    bool shrink_and_retry(const DiskMatrix& matrix, OperationType op_type,
                          int64 n_threads, ChunkConfig& config);

    // Overload for explicit dimensions (when no DiskMatrix available)
    bool shrink_and_retry(int64 n_rows, int64 n_cols, OperationType op_type,
                          int64 n_threads, ChunkConfig& config);

private:
    int64 cap_;               // user ceiling in bytes (0 = no cap)
    int64 available_ram_;
    int64 chunk_override_;    // -1 = auto
    int retry_count_;
    Bottleneck last_bottleneck_ = Bottleneck::None;

    static double op_multiplier(OperationType op);
    static int64 detect_system_ram();
    static int64 detect_free_ram();
    static ChunkAxis op_axis(OperationType op);
};

// Max dense chunk bytes: settable from R option
void set_max_dense_chunk_bytes(int64 bytes);
int64 get_max_dense_chunk_bytes();

} // namespace sclean

#endif // SCLEAN_CHUNK_SCHEDULER_H
