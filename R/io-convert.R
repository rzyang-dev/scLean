# Bidirectional conversion between Seurat and scLean-backed HDF5 objects

#' Convert a Seurat object to scLean (disk-backed)
#'
#' Converts one or more layers of an in-memory Seurat assay to HDF5-backed
#' storage, creating a scLeanAssay that replaces the original assay in the
#' returned Seurat object.
#'
#' @param object  A Seurat object
#' @param hdf5_path  Path for the HDF5 file
#' @param assay  Name of assay to convert (default: current default assay)
#' @param layers  Layers to store on disk (default: all)
#' @param ...  Additional arguments
#' @return A Seurat object with scLeanAssay
#' @examples
#' \donttest{
#' if (requireNamespace("Seurat", quietly = TRUE)) {
#'   counts <- Matrix::rsparsematrix(100, 50, density = 0.1,
#'     dimnames = list(paste0("gene", 1:100), paste0("cell", 1:50)))
#'   seurat_obj <- SeuratObject::CreateSeuratObject(counts)
#'   obj <- as.scLean(seurat_obj, hdf5_path = tempfile(fileext = ".h5"))
#'   obj
#' }
#' }
#' @seealso \code{\link{as.Seurat.scLean}}, \code{\link{LoadScleanObject}}
#' @export
as.scLean <- function(object, hdf5_path, assay = NULL, layers = NULL, ...) {
  if (is.null(assay)) {
    assay <- SeuratObject::DefaultAssay(object)
  }

  if (is.null(hdf5_path)) {
    hdf5_path <- tempfile(fileext = ".h5")
  }

  orig_assay <- object@assays[[assay]]

  if (is.null(layers)) {
    layers <- names(orig_assay@layers)
  }

  normalize_layer_name <- function(name) {
    if (grepl("^counts", name)) return("counts")
    if (grepl("^data", name)) return("data")
    if (grepl("^scale\\.data", name)) return("scale.data")
    return(name)
  }

  written_layers <- character()

  for (layer_name in layers) {
    layer_data <- SeuratObject::GetAssayData(object, assay = assay, layer = layer_name)
    if (is.null(layer_data)) next

    if (!inherits(layer_data, "dgCMatrix")) {
      layer_data <- as(layer_data, "dgCMatrix")
    }

    norm_name <- normalize_layer_name(layer_name)
    if (norm_name %in% written_layers) next
    written_layers <- c(written_layers, norm_name)

    write_csc_to_hdf5(
      hdf5_path  = hdf5_path,
      group_path = paste0("/assays/", assay, "/layers/", norm_name),
      data       = layer_data@x,
      indices    = layer_data@i,
      indptr     = layer_data@p,
      n_rows     = nrow(layer_data),
      n_cols     = ncol(layer_data)
    )
  }

  # Copy feature and cell names
  write_strings_to_hdf5(hdf5_path, "/features/names",
    rownames(orig_assay))
  write_strings_to_hdf5(hdf5_path, "/cells/names",
    colnames(orig_assay))

  # Create scLeanAssay with all layers
  assay_obj <- CreateSCleanAssayFromHDF5(hdf5_path, assay = assay)

  # Copy cell/feature maps and key from original assay for Seurat v5 compatibility
  if (!is.null(orig_assay@cells)) {
    assay_obj@cells <- orig_assay@cells
  }
  if (!is.null(orig_assay@features)) {
    assay_obj@features <- orig_assay@features
  }
  if (!is.null(orig_assay@key)) {
    assay_obj@key <- orig_assay@key
  }

  object@assays[[assay]] <- assay_obj
  return(object)
}

#' Convert scLean assay back to standard Seurat (in-memory)
#'
#' Loads disk-backed layers from HDF5 into memory and replaces the scLeanAssay
#' with a standard Seurat Assay5 object. Useful before saving as .rds or
#' passing to visualization functions that require in-memory data.
#'
#' @param object  A Seurat object with scLeanAssay
#' @param layers  Layers to load into memory (default: "data")
#' @param assay  Assay name (default: current default assay)
#' @param ...  Additional arguments
#' @return A standard Seurat object with in-memory Assay5
#' @examples
#' \donttest{
#' if (requireNamespace("Seurat", quietly = TRUE)) {
#'   counts <- Matrix::rsparsematrix(100, 50, density = 0.1,
#'     dimnames = list(paste0("gene", 1:100), paste0("cell", 1:50)))
#'   obj <- LoadScleanObject(counts)
#'   obj <- NormalizeData(obj)
#'   obj_mem <- as.Seurat.scLean(obj, layers = "data")
#'   obj_mem
#' }
#' }
#' @seealso \code{\link{as.scLean}}
#' @export
as.Seurat.scLean <- function(object, layers = "data", assay = NULL, ...) {
  if (is.null(assay)) {
    assay <- SeuratObject::DefaultAssay(object)
  }

  sc_assay <- object@assays[[assay]]
  if (!inherits(sc_assay, "scLeanAssay")) {
    return(object)
  }

  # Create a standard Assay5 with in-memory layers
  # Load feature/cell names from HDF5 for the placeholder matrix
  gene_names <- read_strings_from_hdf5(sc_assay@hdf5_path, "/features/names")
  cell_names <- read_strings_from_hdf5(sc_assay@hdf5_path, "/cells/names")

  new_assay <- NULL
  for (layer_name in layers) {
    proxy <- sc_assay@layers[[layer_name]]
    if (is.null(proxy)) next

    if (inherits(proxy, "HDF5BackedMatrix")) {
      mat <- proxy$as_matrix()
    } else {
      mat <- proxy
    }

    if (is.null(new_assay)) {
      new_assay <- SeuratObject::CreateAssay5Object(counts = mat)
    } else {
      new_assay <- SeuratObject::SetAssayData(new_assay, layer = layer_name, data = mat)
    }
  }

  if (is.null(new_assay)) {
    stop("No layers could be loaded from HDF5")
  }

  new_assay@key <- sc_assay@key
  new_assay@cells <- sc_assay@cells
  new_assay@features <- sc_assay@features

  object@assays[[assay]] <- new_assay
  return(object)
}
