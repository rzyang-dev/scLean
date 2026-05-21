#' Performance and resource probes for runtime monitoring
#'
#' @name performance-probes
#' @aliases cpp_current_rss cpp_wall_time_ns cpp_resource_snapshot cpp_bottleneck_type
#'   cpp_set_max_dense_chunk_mb cpp_get_max_dense_chunk_mb
#' @description
#' Low-level C++ probes for measuring process memory, wall-clock time, and
#' system resource availability.
#'
#' `PerformanceSnapshot()` combines RSS and wall-time in a single R list.
#' `cpp_resource_snapshot()` returns a full resource picture including free RAM,
#' CPU load, and pressure ratios. `cpp_bottleneck_type()` classifies the current
#' resource bottleneck. `cpp_set_max_dense_chunk_mb()` and
#' `cpp_get_max_dense_chunk_mb()` control the hard cap on dense read buffers.
#'
#' @return
#' \itemize{
#'   \item `cpp_current_rss()`: process resident set size in bytes (numeric).
#'   \item `cpp_wall_time_ns()`: monotonic wall clock in nanoseconds (numeric).
#'   \item `cpp_resource_snapshot()`: list with total_ram, free_ram, current_rss,
#'     available_ram, cpu_load_1min, cpu_load_5min, physical_cores,
#'     available_cores, memory_pressure, cpu_pressure.
#'   \item `cpp_bottleneck_type()`: string "none", "memory", "compute", or "both".
#'   \item `PerformanceSnapshot()`: list with `current_rss_mb` and `wall_time_ns`.
#'   \item `cpp_set_max_dense_chunk_mb(mb)`: set the max dense buffer size in MB.
#'   \item `cpp_get_max_dense_chunk_mb()`: get the current max dense buffer size in MB.
#' }
#'
#' @examples
#' \dontrun{
#'   PerformanceSnapshot()
#'   cpp_current_rss() / 1048576  # RSS in MB
#'   cpp_resource_snapshot()
#'   cpp_bottleneck_type()
#' }
NULL
