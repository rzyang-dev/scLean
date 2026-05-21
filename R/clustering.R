#' Find clusters (disk-backed)
#'
#' Intercepts Seurat::FindClusters to route scLean-backed objects
#' to the disk-based implementation. Runs Louvain or Leiden community
#' detection on the SNN graph stored in HDF5. For graphs with more
#' than 100K cells, the implementation uses memory-aware streaming.
#'
#' @param object  A Seurat object
#' @param algorithm  Clustering algorithm: "leiden" (default) or "louvain"
#' @param resolution  Resolution parameter (default: 0.8)
#' @param n.iter  Number of iterations for Leiden refinement (default: 10)
#' @param ...  Additional arguments
#' @usage \\method{FindClusters}{Seurat}(object, algorithm = "leiden", resolution = 0.8, n.iter = 10, ...)
#' @return Seurat object with cluster assignments (Idents + seurat_clusters)
#' @examples
#' \donttest{
#' counts <- Matrix::rsparsematrix(200, 100, density = 0.15,
#'   dimnames = list(paste0("gene", 1:200), paste0("cell", 1:100)))
#' obj <- LoadScleanObject(counts)
#' obj <- NormalizeData(obj)
#' obj <- FindVariableFeatures(obj, nfeatures = 100)
#' obj <- ScaleData(obj)
#' obj <- RunPCA(obj, npcs = 10)
#' obj <- FindNeighbors(obj, dims = 1:10)
#' obj <- FindClusters(obj, resolution = 0.5, algorithm = "leiden")
#' table(SeuratObject::Idents(obj))
#' }
#' @seealso \code{\link{FindNeighbors.Seurat}}, \code{\link{FindMarkers.Seurat}}
#' @export
FindClusters.Seurat <- function(object, algorithm = "leiden", resolution = 0.8, n.iter = 10, ...) {
  assay <- SeuratObject::DefaultAssay(object)
  sc_assay <- object@assays[[assay]]
  if (inherits(sc_assay, "scLeanAssay")) {
    return(FindClusters.scLeanAssay(object, algorithm = algorithm,
      resolution = resolution, n.iter = n.iter, ...))
  }
  .seurat_original$FindClusters.Seurat(object, algorithm = algorithm,
    resolution = resolution, n.iter = n.iter, ...)
}

#' @export
FindClusters.scLeanAssay <- function(
    object,
    algorithm = "leiden",
    resolution = 0.8,
    n.iter = 10,
    ...
) {
  assay <- SeuratObject::DefaultAssay(object)
  sc_assay <- object@assays[[assay]]

  if (!inherits(sc_assay, "scLeanAssay")) {
    return(Seurat::FindClusters(object, algorithm = algorithm,
      resolution = resolution, n.iter = n.iter, ...))
  }

  cpp_find_clusters(
    hdf5_path   = sc_assay@hdf5_path,
    assay_group = sc_assay@hdf5_group,
    algorithm   = if (algorithm == "leiden") 1L else 0L,
    resolution  = resolution,
    n_iter      = as.integer(n.iter)
  )

  # Read cluster assignments
  cluster_name <- if (algorithm == "leiden") "leiden" else "louvain"
  clusters <- read_hdf5_int32(
    sc_assay@hdf5_path,
    paste0(sc_assay@hdf5_group, "/clusters/", cluster_name)
  )

  cell_names <- read_strings_from_hdf5(sc_assay@hdf5_path, "/cells/names")
  names(clusters) <- cell_names

  object@assays[[assay]] <- sc_assay
  idents <- factor(clusters)
  names(idents) <- cell_names
  SeuratObject::Idents(object) <- idents
  object$seurat_clusters <- idents
  object[[paste0(assay, "_snn_res.", resolution)]] <- clusters

  return(object)
}
