#' Scale data (disk-backed)
#'
#' Computes per-gene mean and standard deviation for centering/scaling.
#' The scaled matrix is NOT materialized on disk — PCA applies
#' centering on-the-fly during Lanczos, keeping disk usage minimal.
#' Uses Welford's online algorithm for numerically stable variance
#' computation.
#'
#' @param object  Seurat object with scLeanAssay
#' @param features  Features to scale (default: all)
#' @param vars.to.regress  Variables to regress out (not yet supported)
#' @param do.scale  Whether to divide by standard deviation (default: TRUE)
#' @param do.center  Whether to subtract mean (default: TRUE)
#' @param chunk.size  Manual chunk size (auto if NULL)
#' @param in_memory  Store in memory?
#' @param ...  Additional arguments
#' @return Seurat object with scaled data (mean/sd stored in HDF5)
#' @examples
#' \donttest{
#' counts <- Matrix::rsparsematrix(100, 50, density = 0.1,
#'   dimnames = list(paste0("gene", 1:100), paste0("cell", 1:50)))
#' obj <- LoadScleanObject(counts)
#' obj <- NormalizeData(obj)
#' obj <- FindVariableFeatures(obj, nfeatures = 50)
#' obj <- ScaleData(obj)
#' }
#' @seealso \code{\link{NormalizeData.scLeanAssay}},
#'   \code{\link{RunPCA.Seurat}}
#' @export
ScaleData.scLeanAssay <- function(
    object,
    features = NULL,
    vars.to.regress = NULL,
    do.scale = TRUE,
    do.center = TRUE,
    chunk.size = NULL,
    in_memory = FALSE,
    ...
) {
  if (inherits(object, "scLeanAssay")) {
    sc_assay <- object
  } else if (inherits(object, "Seurat")) {
    assay <- SeuratObject::DefaultAssay(object)
    sc_assay <- object@assays[[assay]]
    if (!inherits(sc_assay, "scLeanAssay")) {
      return(Seurat::ScaleData(object, features = features,
        vars.to.regress = vars.to.regress,
        do.scale = do.scale, do.center = do.center, ...))
    }
  } else {
    stop("object must be a Seurat object or scLeanAssay")
  }

  chunk <- chunk.size %||% .get_chunk_override() %||% -1

  cpp_scale(
    hdf5_path       = sc_assay@hdf5_path,
    assay_group     = sc_assay@hdf5_group,
    do_scale        = do.scale,
    do_center       = do.center,
    chunk_size      = as.integer(chunk),
    in_memory       = in_memory
  )

  # scale.data is computed on-the-fly by PCA from mean/sd in HDF5.
  # Store path in misc for traceability — no materialized layer needed.
  sc_assay@misc[["scale.data.path"]] <- paste0(
    sc_assay@hdf5_group, "/features/residual_mean"
  )

  if (inherits(object, "Seurat")) {
    object@assays[[assay]] <- sc_assay
    return(object)
  }
  return(sc_assay)
}
