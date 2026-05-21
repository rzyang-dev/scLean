#' @importFrom Rcpp sourceCpp
#' @importFrom methods setClass setMethod new slot slot<- as is show
#' @importFrom R6 R6Class
#' @useDynLib scLean, .registration = TRUE
NULL

# Private environment to hold Seurat original S3 method references
.seurat_original <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {
  # Set default chunk size option
  if (is.null(getOption("scLean.max_ram"))) {
    # Detect physical RAM and default to 80%
    total_ram_bytes <- tryCatch({
      if (Sys.info()["sysname"] == "Darwin") {
        as.numeric(system("sysctl -n hw.memsize", intern = TRUE))
      } else if (.Platform$OS.type == "windows") {
        mem <- tryCatch(
          as.numeric(gsub("\\D", "",
            system("wmic computersystem get TotalPhysicalMemory",
                   intern = TRUE)[2])),
          error = function(e) NA)
        if (length(mem) && !is.na(mem)) mem else NA
      } else {
        as.numeric(system("awk '/MemTotal/ {print $2*1024}' /proc/meminfo",
                   intern = TRUE))
      }
    }, error = function(e) NA)

    if (!is.na(total_ram_bytes) && total_ram_bytes > 0) {
      default_ram <- max(2048 * 1024 * 1024, as.numeric(total_ram_bytes) * 0.8)
    } else {
      default_ram <- 2048 * 1024 * 1024  # fallback 2 GB
    }
    options(scLean.max_ram = default_ram)
  }

  # Suppress HDF5 error stack printing
  suppress_hdf5_errors()

  # Capture Seurat original S3 methods to avoid ::: calls
  .seurat_original$FindNeighbors.Seurat <- utils::getFromNamespace(
    "FindNeighbors.Seurat", "Seurat")
  .seurat_original$FindClusters.Seurat <- utils::getFromNamespace(
    "FindClusters.Seurat", "Seurat")
  .seurat_original$RunPCA.Seurat <- utils::getFromNamespace(
    "RunPCA.Seurat", "Seurat")
  .seurat_original$FindMarkers.Seurat <- utils::getFromNamespace(
    "FindMarkers.Seurat", "Seurat")

  invisible()
}

.onAttach <- function(libname, pkgname) {
  packageStartupMessage(
    "scLean ", utils::packageVersion("scLean"), "\n",
    "  Memory-efficient single-cell analysis\n",
    "  Default max RAM: ", .get_max_ram_bytes() / 1024^2, " MB\n",
    "  Use SetMaxRAM() to adjust."
  )
}
