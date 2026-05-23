#' Normalize data (disk-backed)
#'
#' S3 method for scLeanAssay. Performs chunked normalization directly
#' from the HDF5-backed count matrix, writing normalized values to the
#' "data" layer. Supports LogNormalize, CLR, and RelativeCounts (RC).
#'
#' @param object  Seurat object containing scLeanAssay
#' @param assay  Assay name (default: current default assay)
#' @param normalization.method  "LogNormalize", "CLR", or "RC"
#' @param scale.factor  Scale factor for LogNormalize (default: 10000)
#' @param margin  1 for row, 2 for column (default: 2)
#' @param chunk.size  Manual chunk size (auto if NULL)
#' @param in_memory  If TRUE, return result in memory instead of HDF5
#' @param ...  Additional arguments
#' @return Seurat object with normalized "data" layer on disk
#' @examples
#' \donttest{
#' counts <- Matrix::rsparsematrix(100, 50, density = 0.1,
#'   dimnames = list(paste0("gene", 1:100), paste0("cell", 1:50)))
#' obj <- LoadScleanObject(counts)
#' obj <- NormalizeData(obj)
#' obj <- NormalizeData(obj, normalization.method = "CLR")
#' }
#' @seealso \code{\link{ScaleData.scLeanAssay}},
#'   \code{\link{FindVariableFeatures.scLeanAssay}}
#' @export
NormalizeData.Seurat <- function(
    object,
    assay = NULL,
    normalization.method = "LogNormalize",
    scale.factor = 10000,
    margin = 2,
    chunk.size = NULL,
    in_memory = FALSE,
    ...
) {
  sc_assay <- extract_sc_assay(object, assay)
  if (!is.null(sc_assay)) {
    return(NormalizeData.scLeanAssay(object, assay = assay,
      normalization.method = normalization.method, scale.factor = scale.factor,
      margin = margin, chunk.size = chunk.size, in_memory = in_memory, ...))
  }
  .seurat_original$NormalizeData.Seurat(object, assay = assay,
    normalization.method = normalization.method,
    scale.factor = scale.factor, margin = margin, ...)
}

#' @export
NormalizeData.scLeanAssay <- function(
    object,
    assay = NULL,
    normalization.method = "LogNormalize",
    scale.factor = 10000,
    margin = 2,
    chunk.size = NULL,
    in_memory = FALSE,
    ...
) {
  sc_assay <- extract_sc_assay(object, assay)

  method_map <- c(LogNormalize = 0L, CLR = 1L, RC = 2L)
  method_int <- method_map[normalization.method]
  if (is.na(method_int)) stop("Unknown normalization method: ", normalization.method)

  cpp_normalize(
    hdf5_path    = sc_assay@hdf5_path,
    assay_group  = sc_assay@hdf5_group,
    method       = method_int,
    scale_factor = scale.factor,
    chunk_size   = as.integer(resolve_chunk_size(chunk.size)),
    in_memory    = in_memory
  )

  data_proxy <- HDF5BackedMatrix$new(
    hdf5_path  = sc_assay@hdf5_path,
    group_path = paste0(sc_assay@hdf5_group, "/layers/data")
  )
  sc_assay@layers[["data"]] <- data_proxy
  sc_assay <- register_layer(sc_assay, "data")
  reinstall_assay(object, sc_assay, assay)
}
