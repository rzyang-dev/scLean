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
  if (inherits(object, "scLeanAssay")) {
    sc_assay <- object
  } else if (inherits(object, "Seurat")) {
    if (is.null(assay)) assay <- SeuratObject::DefaultAssay(object)
    sc_assay <- object@assays[[assay]]
    if (!inherits(sc_assay, "scLeanAssay")) {
      return(Seurat::NormalizeData(object, assay = assay,
        normalization.method = normalization.method,
        scale.factor = scale.factor, margin = margin, ...))
    }
  } else {
    stop("object must be a Seurat object or scLeanAssay")
  }

  method_map <- c(
    LogNormalize = 0L,
    CLR          = 1L,
    RC           = 2L
  )

  method_int <- method_map[normalization.method]
  if (is.na(method_int)) {
    stop("Unknown normalization method: ", normalization.method)
  }

  chunk <- chunk.size %||% .get_chunk_override() %||% -1

  cpp_normalize(
    hdf5_path    = sc_assay@hdf5_path,
    assay_group  = sc_assay@hdf5_group,
    method       = method_int,
    scale_factor = scale.factor,
    chunk_size   = as.integer(chunk),
    in_memory    = in_memory
  )

  # Update the layers in the assay
  data_proxy <- HDF5BackedMatrix$new(
    hdf5_path  = sc_assay@hdf5_path,
    group_path = paste0(sc_assay@hdf5_group, "/layers/data")
  )

  sc_assay@layers[["data"]] <- data_proxy
  # Update cell/feature maps for Seurat v5 validation
  sc_assay <- .add_layer_to_maps(sc_assay, "data")

  if (inherits(object, "Seurat")) {
    object@assays[[assay]] <- sc_assay
    return(object)
  }
  return(sc_assay)
}
