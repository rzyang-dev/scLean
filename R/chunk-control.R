#' Set chunk size for the next operation
#'
#' @param object  A Seurat/scLean object
#' @param size  Number of rows/columns per chunk (NA to restore auto)
#' @return The object, invisibly
#' @examples
#' counts <- Matrix::rsparsematrix(100, 50, density = 0.1)
#' obj <- LoadScleanObject(counts)
#' SetChunkSize(obj, 1000)
#' SetChunkSize(obj, NA)  # restore auto
#' @seealso \code{\link{SetMaxRAM}}, \code{\link{SetThreads}}
#' @export
SetChunkSize <- function(object, size) {
  if (is.na(size)) {
    options(scLean.chunk_size = NULL)
    return(invisible(object))
  }
  size <- as.integer(size)
  if (is.na(size) || size <= 0) stop("chunk_size must be a positive integer")
  options(scLean.chunk_size = size)
  invisible(object)
}

#' Set maximum RAM scLean is allowed to use (in MB)
#'
#' @param max_ram  Memory limit in megabytes (default: 2048)
#' @return The memory limit in bytes, invisibly
#' @examples
#' SetMaxRAM(4096)
#' SetMaxRAM()  # restore default 2048 MB
#' @seealso \code{\link{SetChunkSize}}, \code{\link{MemoryReport}}
#' @export
SetMaxRAM <- function(max_ram = 2048) {
  bytes <- as.numeric(max_ram) * 1024 * 1024
  options(scLean.max_ram = bytes)
  invisible(bytes)
}

#' Report current memory and resource usage statistics
#'
#' @param object  A scLean object (optional)
#' @return List with memory and resource information
#' @examples
#' \donttest{
#' counts <- Matrix::rsparsematrix(100, 50, density = 0.1)
#' obj <- LoadScleanObject(counts)
#' MemoryReport(obj)
#' }
#' @seealso \code{\link{CheckResources}}, \code{\link{RefreshMemory}},
#'   \code{\link{PerformanceSnapshot}}
#' @export
MemoryReport <- function(object = NULL) {
  snap <- cpp_resource_snapshot()
  report <- list(
    # System resource snapshot
    total_ram_mb      = snap$total_ram / 1048576,
    free_ram_mb       = snap$free_ram / 1048576,
    current_rss_mb    = snap$current_rss / 1048576,
    available_ram_mb  = snap$available_ram / 1048576,
    memory_pressure   = snap$memory_pressure,
    # CPU
    cpu_load          = snap$cpu_load_1min,
    available_cores   = snap$available_cores,
    cpu_pressure      = snap$cpu_pressure,
    # Bottleneck
    bottleneck        = cpp_bottleneck_type(),
    # Config
    max_ram_mb        = getOption("scLean.max_ram", default = 2048),
    max_dense_chunk_mb = cpp_get_max_dense_chunk_mb(),
    chunk_size        = getOption("scLean.chunk_size", default = "auto")
  )

  if (!is.null(object)) {
    for (assay_name in names(object@assays)) {
      assay <- object@assays[[assay_name]]
      if (inherits(assay, "scLeanAssay")) {
        report$hdf5_path <- assay@hdf5_path
        if (file.exists(assay@hdf5_path)) {
          report$hdf5_file_size_mb <- file.info(assay@hdf5_path)$size / 1024^2
        }
        layers_info <- list()
        for (ln in names(assay@layers)) {
          l <- assay@layers[[ln]]
          if (is.list(l) && isTRUE(l$virtual)) {
            layers_info[[ln]] <- list(
              path = l$path,
              type = "virtual"
            )
          } else {
            layers_info[[ln]] <- list(
              dims = dim(l),
              type = class(l)[1]
            )
          }
        }
        report$layers <- layers_info
      }
    }
  }

  return(report)
}

#' Check resource availability and warn if under pressure
#'
#' Queries the system resource monitor and warns if memory or CPU
#' are under heavy pressure. Call between pipeline steps for early
#' warning of resource constraints.
#'
#' @return Invisible list with resource snapshot including memory_pressure,
#'   cpu_pressure, and bottleneck classification.
#' @examples
#' CheckResources()
#' @seealso \code{\link{MemoryReport}()}, \code{\link{RefreshMemory}()}
#' @export
CheckResources <- function() {
  snap <- cpp_resource_snapshot()
  if (snap$memory_pressure > 0.9) {
    warning(sprintf(
      "Memory pressure %.0f%% - consider reducing max_dense_chunk_mb or closing other applications",
      snap$memory_pressure * 100
    ))
  }
  if (snap$cpu_pressure > 0.9) {
    warning(sprintf(
      "CPU pressure %.0f%% - background processes may slow analysis",
      snap$cpu_pressure * 100
    ))
  }
  if (snap$memory_pressure > 0.9 && snap$cpu_pressure > 0.9) {
    warning("Both memory and CPU are under high pressure - analysis may be slow")
  }
  invisible(snap)
}

#' Trigger R-level garbage collection
#'
#' Calls R's garbage collector to release any R-side memory before
#' the next C++ operation starts. Use between pipeline steps when
#' working with large datasets.
#'
#' @param object  A scLean/Seurat object (passed through invisibly)
#' @return The object, invisibly
#' @examples
#' counts <- Matrix::rsparsematrix(100, 50, density = 0.1)
#' obj <- LoadScleanObject(counts)
#' obj <- RefreshMemory(obj)
#' @seealso \code{\link{CheckResources}()}, \code{\link{MemoryReport}()}
#' @export
RefreshMemory <- function(object = NULL) {
  gc()
  invisible(object)
}

#' Quick performance snapshot (RSS + elapsed wall time)
#'
#' Call between pipeline steps for differential timing.
#' Returns a list with current_rss_mb and wall_time_ns (monotonic).
#'
#' @return List with current_rss_mb, wall_time_ns
#' @examples
#' PerformanceSnapshot()
#' @seealso \code{\link{MemoryReport}}, \code{\link{CheckResources}}
#' @export
PerformanceSnapshot <- function() {
  list(
    current_rss_mb = cpp_current_rss() / 1048576,
    wall_time_ns   = cpp_wall_time_ns()
  )
}

#' Toggle verbose output
#'
#' @param verbose  Logical, whether to print progress messages
#' @return The verbose setting, invisibly
#' @examples
#' SetVerbose(TRUE)
#' SetVerbose(FALSE)
#' @seealso \code{\link{MemoryReport}}
#' @export
SetVerbose <- function(verbose = TRUE) {
  options(scLean.verbose = verbose)
  invisible(verbose)
}

#' Set number of threads for parallel operations
#'
#' Controls OpenMP thread count for read-intensive operations
#' (NormalizeData, ScaleData, FindVariableFeatures, FindMarkers, BuildSNN).
#' Affects per-thread HDF5 read handles; writes are always serialized.
#'
#' @param n  Number of threads (default: detected core count). Pass NULL to
#'   restore the default single-threaded mode.
#' @return The thread count, invisibly (NULL if threads were unset)
#' @examples
#' SetThreads(2)
#' SetThreads()  # restore default single-threaded mode
#' @seealso \code{\link{SetChunkSize}}, \code{\link{SetMaxRAM}}
#' @export
SetThreads <- function(n = NULL) {
  if (is.null(n)) {
    options(scLean.threads = NULL)
    cpp_set_threads(1L)
    return(invisible(NULL))
  }
  n <- as.integer(n)
  if (is.na(n) || n < 1) stop("n must be a positive integer")
  options(scLean.threads = n)
  cpp_set_threads(n)
  invisible(n)
}

# --- Internal option helpers ---

.get_chunk_override <- function() {
  getOption("scLean.chunk_size", default = NULL)
}

.get_max_ram_bytes <- function() {
  getOption("scLean.max_ram", default = 2048 * 1024 * 1024)
}

.get_verbose <- function() {
  getOption("scLean.verbose", default = TRUE)
}

.get_threads <- function() {
  getOption("scLean.threads", default = NULL)
}
