# Main entry point for loading data into disk-backed scLean objects

#' Load a dataset into a disk-backed scLean object
#'
#' Supports 10X-format directories (matrix.mtx + features.tsv + barcodes.tsv),
#' HDF5 files, dgCMatrix, or existing Seurat objects.
#'
#' @param input  A 10X directory path, HDF5 file path, dgCMatrix, or Seurat object
#' @param hdf5_path  Path for the output HDF5 file (auto-generated if NULL)
#' @param assay  Name for the assay
#' @param ...  Additional arguments
#' @return A Seurat object with scLeanAssay
#' @examples
#' counts <- Matrix::rsparsematrix(200, 100, density = 0.1,
#'   dimnames = list(paste0("gene", 1:200), paste0("cell", 1:100)))
#' obj <- LoadScleanObject(counts)
#' obj
#' @seealso \code{\link{as.scLean}}, \code{\link{CreateSCleanAssay}}
#' @export
LoadScleanObject <- function(
    input,
    hdf5_path = NULL,
    assay = "RNA",
    ...
) {
  if (is.null(hdf5_path)) {
    hdf5_path <- tempfile(fileext = ".h5")
  }

  if (inherits(input, "Seurat")) {
    return(as.scLean(input, hdf5_path = hdf5_path, assay = assay, ...))
  }

  if (is.character(input) && dir.exists(input)) {
    # 10X directory — streaming import, no intermediate dgCMatrix
    assay_obj <- import_10x_directory(input, hdf5_path, assay)
  } else if (is.character(input) && grepl("\\.(h5|hdf5)$", input, ignore.case = TRUE)) {
    if (file.exists(input)) {
      groups <- list_hdf5_groups(input, "/")
      if ("matrix" %in% groups) {
        # CellRanger .h5 detection — routed to import_cellranger_h5()
        assay_obj <- import_cellranger_h5(input, hdf5_path, assay)
      } else {
        assay_obj <- CreateSCleanAssayFromHDF5(input, assay = assay)
      }
    } else {
      stop("HDF5 file not found: ", input)
    }
  } else if (inherits(input, "dgCMatrix") || is.matrix(input)) {
    assay_obj <- CreateSCleanAssay(input, hdf5_path = hdf5_path, assay = assay)
  } else {
    stop("Unsupported input type: ", class(input)[1])
  }

  # CreateSeuratObject() requires a counts matrix. We provide a 1×1 sparse
  # placeholder that satisfies the constructor arguments, then immediately
  # overwrite it with the actual scLeanAssay. This avoids materializing the
  # full expression matrix in memory. The placeholder's dimensions must match
  # the actual data dimensions so CreateSeuratObject's internal validation passes.
  placeholder <- Matrix::sparseMatrix(
    i = 1, j = 1, x = 1,
    dims = c(length(gene_names), length(cell_names)),
    dimnames = list(gene_names, cell_names)
  )
  obj <- SeuratObject::CreateSeuratObject(counts = placeholder, assay = assay)
  obj@assays[[assay]] <- assay_obj
  return(obj)
}
