#' @importFrom Rcpp sourceCpp
#' @importFrom methods setClass setMethod new slot slot<- as is show
#' @importFrom R6 R6Class
#' @useDynLib scLean, .registration = TRUE
NULL

# Private environment to hold Seurat original S3 method references
.seurat_original <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {
  # RAM is auto-managed based on system free memory.
  # No default scLean.max_ram is set; the C++ scheduler detects free RAM
  # and applies SAFETY_FACTOR + OS/thread reserves dynamically.
  # Use SetMaxRAM(N) to optionally cap the auto-detected value.

  # Suppress HDF5 error stack printing
  suppress_hdf5_errors()

  # Capture Seurat original S3 methods to avoid ::: calls
  .seurat_original$NormalizeData.Seurat <- utils::getFromNamespace(
    "NormalizeData.Seurat", "Seurat")
  .seurat_original$ScaleData.Seurat <- utils::getFromNamespace(
    "ScaleData.Seurat", "Seurat")
  .seurat_original$FindVariableFeatures.Seurat <- utils::getFromNamespace(
    "FindVariableFeatures.Seurat", "Seurat")
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
  free_mb <- tryCatch({
    snap <- cpp_resource_snapshot()
    round(snap$free_ram / 1048576)
  }, error = function(e) NA)

  msg <- paste0(
    "scLean ", utils::packageVersion("scLean"), "\n",
    "  Memory-efficient single-cell analysis\n",
    "  RAM managed automatically (free: ",
    if (is.na(free_mb)) "unknown" else paste0("~", free_mb, " MB"), ")\n"
  )

  cap <- getOption("scLean.max_ram")
  if (!is.null(cap)) {
    msg <- paste0(msg, "  User cap: ", round(cap / 1048576), " MB\n")
  }
  msg <- paste0(msg, "  Use SetMaxRAM() to set a memory ceiling.")

  packageStartupMessage(msg)
}
