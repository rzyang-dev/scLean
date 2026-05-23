#' Find neighbors (disk-backed)
#'
#' Intercepts Seurat::FindNeighbors to route scLean-backed objects
#' to the disk-based implementation. Builds KNN and SNN graphs from
#' PCA embeddings, storing them in HDF5. Uses Annoy for large datasets,
#' falls back to brute-force with scheduler-controlled block size when
#' memory is constrained.
#'
#' @param object  A Seurat object
#' @param reduction  Reduction to use (default: "pca")
#' @param dims  Dimensions of reduction to use (default: 1:30)
#' @param k.param  Number of nearest neighbors (default: 20)
#' @param annoy.metric  Distance metric for Annoy (default: "euclidean")
#' @param n.trees  Number of trees for Annoy index (default: 50)
#' @param ...  Additional arguments
#' @usage
#' \method{FindNeighbors}{Seurat}(object, reduction = "pca", dims = NULL,
#'   k.param = 20, annoy.metric = "euclidean", n.trees = 50, ...)
#' @return Seurat object with NN and SNN graphs
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
#' }
#' @seealso \code{\link{RunPCA.Seurat}}, \code{\link{FindClusters.Seurat}}
#' @export
FindNeighbors.Seurat <- function(object, reduction = "pca", dims = NULL, k.param = 20, annoy.metric = "euclidean", n.trees = 50, ...) {
  assay <- SeuratObject::DefaultAssay(object)
  sc_assay <- object@assays[[assay]]
  if (inherits(sc_assay, "scLeanAssay")) {
    return(FindNeighbors.scLeanAssay(object, reduction = reduction,
      dims = dims, k.param = k.param, annoy.metric = annoy.metric,
      n.trees = n.trees, ...))
  }
  .seurat_original$FindNeighbors.Seurat(object, reduction = reduction,
    dims = dims, k.param = k.param, annoy.metric = annoy.metric,
    n.trees = n.trees, ...)
}

#' @export
FindNeighbors.scLeanAssay <- function(
    object,
    reduction = "pca",
    dims = NULL,
    k.param = 20,
    annoy.metric = "euclidean",
    n.trees = 50,
    ...
) {
  assay <- SeuratObject::DefaultAssay(object)
  sc_assay <- object@assays[[assay]]

  if (!inherits(sc_assay, "scLeanAssay")) {
    return(Seurat::FindNeighbors(object, reduction = reduction,
      dims = dims, k.param = k.param, annoy.metric = annoy.metric,
      n.trees = n.trees, ...))
  }

  # Get PCA embeddings
  if (!reduction %in% names(object@reductions)) {
    stop("Reduction '", reduction, "' not found. Run RunPCA first.")
  }

  # PCA embeddings are always fully loaded into R memory here.
  # Seurat's Embeddings() function requires an in-memory matrix.
  # For 500K cells × 30 PCs, this is ~120 MB — a primary RAM consumer.
  embeddings <- SeuratObject::Embeddings(object, reduction = reduction)
  npcs <- ncol(embeddings)
  if (is.null(dims)) {
    dims <- seq_len(min(npcs, 30))
  }
  emb_subset <- embeddings[, dims, drop = FALSE]

  cpp_find_neighbors(
    embeddings   = emb_subset,
    k            = as.integer(k.param),
    metric       = annoy.metric,
    n_trees      = as.integer(n.trees),
    hdf5_path    = sc_assay@hdf5_path,
    assay_group  = sc_assay@hdf5_group
  )

  # Create Neighbor graph object
  nn_path <- paste0(sc_assay@hdf5_group, "/graphs/nn")
  nn_idx <- read_hdf5_int32(sc_assay@hdf5_path, paste0(nn_path, "/indices"))
  nn_dists <- read_hdf5_float32(sc_assay@hdf5_path, paste0(nn_path, "/distances"))

  # C++ stores NN indices in column-major order (Fortran convention).
  # byrow=FALSE reads column-by-column, matching the memory layout.
  # Using byrow=TRUE would produce completely wrong neighbor assignments.
  nn_mat <- matrix(nn_idx, nrow = ncol(sc_assay), byrow = FALSE)
  colnames(nn_mat) <- paste0("NN_", seq_len(k.param))
  object[[paste0(assay, "_nn")]] <- nn_mat

  snn_path <- paste0(sc_assay@hdf5_group, "/graphs/snn")

  object@graphs[[paste0(assay, "_snn")]] <- HDF5BackedMatrix$new(
    hdf5_path  = sc_assay@hdf5_path,
    group_path = snn_path
  )

  object@assays[[assay]] <- sc_assay
  return(object)
}
