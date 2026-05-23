#ifndef SCLEAN_CHUNK_PLATFORM_H
#define SCLEAN_CHUNK_PLATFORM_H

#include "scLean_types.h"

namespace sclean {
namespace detail {

// Total physical RAM detection (cross-platform).
int64 detect_system_ram();

// Free RAM detection (cross-platform).
// Includes a floor at 20% of total RAM to avoid excessively small chunks
// when the system reports near-zero free memory (e.g., due to disk cache).
int64 detect_free_ram();

} // namespace detail
} // namespace sclean

#endif // SCLEAN_CHUNK_PLATFORM_H
