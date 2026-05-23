#' scLeanAssay: A disk-backed single-cell assay
#'
#' Extends SeuratObject::Assay5. All layer data is stored in HDF5 files
#' rather than in-memory R matrices.
#'
#' @slot hdf5_path  Path to the backing HDF5 file
#' @slot hdf5_group HDF5 group path within the file for this assay
#'
#' @export
setClass(
  Class = "scLeanAssay",
  contains = "Assay5",
  slots = c(
    hdf5_path  = "character",
    hdf5_group = "character"
  )
)

#' Create a new scLeanAssay
#'
#' Writes a dgCMatrix of counts to an HDF5 file and wraps it as a
#' disk-backed scLeanAssay. All subsequent operations will stream data
#' from the HDF5 file in adaptive chunks.
#'
#' @param counts  A dgCMatrix of counts (genes x cells)
#' @param hdf5_path  Path for the HDF5 file (auto-generated tempfile if NULL)
#' @param assay  Assay name (default: "RNA")
#' @param ...  Additional arguments
#' @return A scLeanAssay object
#' @examples
#' counts <- Matrix::rsparsematrix(100, 50, density = 0.1,
#'   dimnames = list(paste0("gene", 1:100), paste0("cell", 1:50)))
#' assay <- CreateSCleanAssay(counts)
#' assay
#' @seealso \code{\link{CreateSCleanAssayFromHDF5}}, \code{\link{LoadScleanObject}}
#' @export
CreateSCleanAssay <- function(
    counts,
    hdf5_path = NULL,
    assay = "RNA",
    ...
) {
  if (is.null(hdf5_path)) {
    hdf5_path <- tempfile(fileext = ".h5")
  }

  # Convert counts to dgCMatrix if needed
  if (!inherits(counts, "dgCMatrix")) {
    counts <- as(counts, "dgCMatrix")
  }

  # Write counts to HDF5 via C++ layer
  write_csc_to_hdf5(
    hdf5_path   = hdf5_path,
    group_path  = paste0("/assays/", assay, "/layers/counts"),
    data        = counts@x,
    indices     = counts@i,
    indptr      = counts@p,
    n_rows      = nrow(counts),
    n_cols      = ncol(counts)
  )

  # Write feature names (synthesize defaults if missing)
  rn <- rownames(counts)
  if (is.null(rn)) rn <- paste0("gene", seq_len(nrow(counts)))
  write_strings_to_hdf5(hdf5_path, "/features/names", rn)

  # Write cell names
  cn <- colnames(counts)
  if (is.null(cn)) cn <- paste0("cell", seq_len(ncol(counts)))
  write_strings_to_hdf5(hdf5_path, "/cells/names", cn)

  # Create the assay object
  assay_obj <- new("scLeanAssay",
    hdf5_path  = hdf5_path,
    hdf5_group = paste0("/assays/", assay)
  )

  # Set the key (e.g., "rna_")
  assay_obj@key <- paste0(tolower(assay), "_")

  # Set up inherited Assay5 slots for Seurat integration
  assay_obj@cells <- SeuratObject::LogMap(cn)
  assay_obj@features <- SeuratObject::LogMap(rn)

  # LogMap init contract:
  # Seurat v5 Assay5 requires that cell/feature LogMaps track which layers
  # each cell/feature participates in. After initial creation, we register
  # the "counts" layer. Each subsequent pipeline step that adds a new layer
  # (e.g., "data" after NormalizeData) must call .add_layer_to_maps() to
  # keep the LogMaps in sync. Failure to do so causes Seurat's internal
  # validation to reject the object.
  assay_obj@cells <- .add_layer_to_logmap(assay_obj@cells, "counts")
  assay_obj@features <- .add_layer_to_logmap(assay_obj@features, "counts")

  # Set up layers as HDF5BackedMatrix proxies
  counts_proxy <- HDF5BackedMatrix$new(
    hdf5_path  = hdf5_path,
    group_path = paste0("/assays/", assay, "/layers/counts")
  )

  assay_obj@layers <- list(counts = counts_proxy)

  return(assay_obj)
}

# Add a layer to the cell/feature LogMaps (for Seurat v5 validation)
.add_layer_to_logmap <- function(logmap, layer_name) {
  if (is.null(logmap)) return(NULL)
  if (layer_name %in% colnames(logmap@.Data)) return(logmap)
  new_data <- cbind(logmap@.Data, TRUE)
  colnames(new_data)[ncol(new_data)] <- layer_name
  logmap@.Data <- new_data
  logmap
}

# Update both cell and feature maps to include a new layer
.add_layer_to_maps <- function(assay, layer_name) {
  assay@cells <- .add_layer_to_logmap(assay@cells, layer_name)
  assay@features <- .add_layer_to_logmap(assay@features, layer_name)
  assay
}

#' Show method for scLeanAssay
#' @param object A scLeanAssay object
setMethod(
  "show",
  "scLeanAssay",
  function(object) {
    cat("scLeanAssay (disk-backed)\n")
    cat("  HDF5 file: ", object@hdf5_path, "\n")
    cat("  Group:     ", object@hdf5_group, "\n")

    layers <- names(object@layers)
    if (length(layers) > 0) {
      cat("  Layers:    ", paste(layers, collapse = ", "), "\n")
      for (ly in layers) {
        l <- object@layers[[ly]]
        if (inherits(l, "HDF5BackedMatrix")) {
          cat("    ", ly, ": ", nrow(l), " genes x ", ncol(l),
              " cells (HDF5)\n", sep = "")
        }
      }
    } else {
      cat("  (no layers)\n")
    }
  }
)
