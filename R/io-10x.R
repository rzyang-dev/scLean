# 10X Genomics import helpers — directory and CellRanger .h5 streaming

#' Import a 10X Genomics directory into a disk-backed scLean assay
#'
#' Detects 10X directory structure by checking for matrix.mtx or matrix.mtx.gz,
#' streams the matrix into HDF5 via \code{stream_10x_to_hdf5()}, and creates a
#' scLeanAssay from the result.
#'
#' @param tenx_dir  Path to 10X output directory
#' @param hdf5_path Path for the output HDF5 file
#' @param assay     Name for the assay
#' @return A scLeanAssay object
#' @keywords internal
import_10x_directory <- function(tenx_dir, hdf5_path, assay) {
  mtx_files <- list.files(tenx_dir, pattern = "^matrix\\.mtx(\\.gz)?$")
  if (length(mtx_files) == 0) {
    stop("No matrix.mtx(.gz) found in directory: ", tenx_dir)
  }
  stream_10x_to_hdf5(
    tenx_dir  = tenx_dir,
    hdf5_path = hdf5_path,
    assay     = assay
  )
  CreateSCleanAssayFromHDF5(hdf5_path, assay = assay)
}

#' Import CellRanger .h5 file into a disk-backed scLean assay
#'
#' Detects CellRanger v3 .h5 format (presence of a "matrix" group at root
#' level) and imports via \code{Seurat::Read10X_h5()}.  This path is
#' currently NOT streaming: the full matrix is materialised in memory before
#' writing to scLean's HDF5.  See KNOWN-ISSUES.md #4.
#'
#' @param h5_path   Path to the CellRanger .h5 file
#' @param hdf5_path Path for the output HDF5 file
#' @param assay     Name for the assay
#' @return A scLeanAssay object
#' @keywords internal
import_cellranger_h5 <- function(h5_path, hdf5_path, assay) {
  if (requireNamespace("Seurat", quietly = TRUE)) {
    counts <- Seurat::Read10X_h5(h5_path)
    CreateSCleanAssay(counts, hdf5_path = hdf5_path, assay = assay)
  } else {
    stop("CellRanger .h5 format detected but Seurat is not available")
  }
}
