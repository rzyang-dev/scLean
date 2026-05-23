# Re-open an existing scLean HDF5 assay

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
