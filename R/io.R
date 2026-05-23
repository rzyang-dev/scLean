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
    stream_10x_to_hdf5(
      tenx_dir  = input,
      hdf5_path = hdf5_path,
      assay     = assay
    )
    assay_obj <- CreateSCleanAssayFromHDF5(hdf5_path, assay = assay)
  } else if (is.character(input) && grepl("\\.(h5|hdf5)$", input, ignore.case = TRUE)) {
    if (file.exists(input)) {
      groups <- list_hdf5_groups(input, "/")
      if ("matrix" %in% groups) {
        # CellRanger .h5 detection: "matrix" group indicates CellRanger v3 format.
        # CURRENT LIMITATION: Seurat::Read10X_h5() reads the ENTIRE matrix into
        # R memory before writing to scLean's HDF5, defeating the streaming
        # advantage. A future improvement would stream directly from CellRanger
        # .h5 to scLean .h5 without the intermediate in-memory dgCMatrix.
        # See KNOWN-ISSUES.md #4.
        if (requireNamespace("Seurat", quietly = TRUE)) {
          counts <- Seurat::Read10X_h5(input)
          assay_obj <- CreateSCleanAssay(counts, hdf5_path = hdf5_path, assay = assay)
        } else {
          stop("CellRanger .h5 format detected but Seurat is not available")
        }
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

#' Create a scLeanAssay from an existing scLean HDF5 file
#'
#' Re-opens a previously created HDF5-backed assay. All layers found in the
#' HDF5 file under \code{/assays/<assay>/layers/} are wrapped as
#' HDF5BackedMatrix proxies.
#'
#' @param hdf5_path  Path to the HDF5 file
#' @param assay  Assay name (default: "RNA")
#' @return A scLeanAssay object
#' @examples
#' \donttest{
#' counts <- Matrix::rsparsematrix(100, 50, density = 0.1,
#'   dimnames = list(paste0("gene", 1:100), paste0("cell", 1:50)))
#' tmp <- tempfile(fileext = ".h5")
#' CreateSCleanAssay(counts, hdf5_path = tmp)
#' assay <- CreateSCleanAssayFromHDF5(tmp)
#' assay
#' }
#' @seealso \code{\link{CreateSCleanAssay}}, \code{\link{LoadScleanObject}}
#' @export
CreateSCleanAssayFromHDF5 <- function(hdf5_path, assay = "RNA") {
  if (!file.exists(hdf5_path)) stop("HDF5 file not found: ", hdf5_path)
  assay_obj <- new("scLeanAssay",
    hdf5_path  = hdf5_path,
    hdf5_group = paste0("/assays/", assay)
  )

  # Set the key (e.g., "rna_")
  assay_obj@key <- paste0(tolower(assay), "_")

  # List available layers in the HDF5 file
  layers <- list()
  layers_group <- paste0("/assays/", assay, "/layers")

  layer_names <- list_hdf5_groups(hdf5_path, layers_group)

  for (ln in layer_names) {
    group_path <- paste0(layers_group, "/", ln)
    layers[[ln]] <- HDF5BackedMatrix$new(
      hdf5_path  = hdf5_path,
      group_path = group_path
    )
  }

  if (length(layers) == 0) {
    stop("No layers found in HDF5 file at ", layers_group)
  }

  assay_obj@layers <- layers

  # Set up cell/feature LogMaps from stored names
  cell_names <- read_strings_from_hdf5(hdf5_path, "/cells/names")
  feature_names <- read_strings_from_hdf5(hdf5_path, "/features/names")
  assay_obj@cells <- SeuratObject::LogMap(cell_names)
  assay_obj@features <- SeuratObject::LogMap(feature_names)
  for (ln in layer_names) {
    assay_obj@cells <- .add_layer_to_logmap(assay_obj@cells, ln)
    assay_obj@features <- .add_layer_to_logmap(assay_obj@features, ln)
  }

  return(assay_obj)
}
