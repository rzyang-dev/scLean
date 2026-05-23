#' Find variable features (disk-backed VST)
#'
#' Uses variance-stabilizing transformation to identify highly variable
#' genes. Computation is chunked across cells and streamed from HDF5.
#' When memory is constrained, automatically falls back to a sparse
#' computation path.
#'
#' @param object  Seurat object with scLeanAssay
#' @param nfeatures  Number of variable features to select (default: 2000)
#' @param ...  Additional arguments
#' @return Seurat object with variable features stored in HDF5 and
#'   registered via \code{VariableFeatures()}
#' @importFrom SeuratObject VariableFeatures
#' @examples
#' \donttest{
#' counts <- Matrix::rsparsematrix(200, 100, density = 0.15,
#'   dimnames = list(paste0("gene", 1:200), paste0("cell", 1:100)))
#' obj <- LoadScleanObject(counts)
#' obj <- NormalizeData(obj)
#' obj <- FindVariableFeatures(obj, nfeatures = 100)
#' head(VariableFeatures(obj))
#' }
#' @seealso \code{\link{NormalizeData.scLeanAssay}},
#'   \code{\link{ScaleData.scLeanAssay}}
#' @export
FindVariableFeatures.Seurat <- function(
    object,
    nfeatures = 2000,
    ...
) {
  sc_assay <- extract_sc_assay(object)
  if (!is.null(sc_assay)) {
    return(FindVariableFeatures.scLeanAssay(object, nfeatures = nfeatures, ...))
  }
  .seurat_original$FindVariableFeatures.Seurat(object, nfeatures = nfeatures, ...)
}

#' @export
FindVariableFeatures.scLeanAssay <- function(
    object,
    nfeatures = 2000,
    ...
) {
  sc_assay <- extract_sc_assay(object)

  cpp_vst(
    hdf5_path   = sc_assay@hdf5_path,
    assay_group = sc_assay@hdf5_group,
    n_top       = as.integer(nfeatures)
  )

  # Read variable features from HDF5
  vf <- read_strings_from_hdf5(sc_assay@hdf5_path, "/features/names")
  is_var <- read_hdf5_int32(sc_assay@hdf5_path,
                            paste0(sc_assay@hdf5_group, "/features/variable"))

  result <- reinstall_assay(object, sc_assay)
  if (inherits(result, "Seurat")) {
    selected <- vf[is_var == 1]
    VariableFeatures(result[[SeuratObject::DefaultAssay(result)]]) <- selected
    return(result)
  }
  VariableFeatures(result) <- vf[is_var == 1]
  return(result)
}
