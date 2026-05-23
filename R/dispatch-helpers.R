# Shared dispatch helpers for scLean pipeline functions

#' Extract scLeanAssay from Seurat object or return directly
#' @param object  A Seurat object or scLeanAssay
#' @param assay   Assay name (defaults to DefaultAssay if object is Seurat)
#' @return scLeanAssay or NULL if assay is not scLeanAssay
#' @noRd
extract_sc_assay <- function(object, assay = NULL) {
  if (inherits(object, "scLeanAssay")) {
    return(object)
  }
  if (inherits(object, "Seurat")) {
    if (is.null(assay)) assay <- SeuratObject::DefaultAssay(object)
    sc_assay <- object@assays[[assay]]
    if (inherits(sc_assay, "scLeanAssay")) return(sc_assay)
    return(NULL)
  }
  stop("object must be a Seurat object or scLeanAssay")
}

#' Resolve chunk size with fallback chain
#' @param chunk.size  User-provided chunk size or NULL
#' @return Integer chunk size or -1 (auto-detect)
#' @noRd
resolve_chunk_size <- function(chunk.size = NULL) {
  chunk.size %||% .get_chunk_override() %||% -1
}

#' Reinstall scLeanAssay back into Seurat object
#' @param object     Seurat object or scLeanAssay
#' @param sc_assay   The (possibly modified) scLeanAssay
#' @param assay      Assay name
#' @return Seurat object with assay reinstalled, or scLeanAssay unchanged
#' @noRd
reinstall_assay <- function(object, sc_assay, assay = NULL) {
  if (inherits(object, "Seurat")) {
    if (is.null(assay)) assay <- SeuratObject::DefaultAssay(object)
    object@assays[[assay]] <- sc_assay
    return(object)
  }
  return(sc_assay)
}

#' Register a new layer in cell and feature LogMaps
#' @param sc_assay   scLeanAssay
#' @param layer_name Layer name (e.g., "data")
#' @return scLeanAssay with updated LogMaps
#' @noRd
register_layer <- function(sc_assay, layer_name) {
  sc_assay@cells <- .add_layer_to_logmap(sc_assay@cells, layer_name)
  sc_assay@features <- .add_layer_to_logmap(sc_assay@features, layer_name)
  sc_assay
}
