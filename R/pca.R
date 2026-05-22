#' Run PCA (disk-backed IRLBA)
#'
#' Intercepts Seurat::RunPCA to route scLean-backed objects
#' to the disk-based implementation. Uses IRLBA (implicitly restarted
#' Lanczos bidiagonalization) with on-the-fly centering from HDF5.
#'
#' @param object  A Seurat object
#' @param npcs  Number of PCs to compute (default: 50)
#' @param features  Features to use for PCA (default: VariableFeatures)
#' @param reduction.name  Name of the reduction (default: "pca")
#' @param reduction.key  Prefix for PC names (default: "PC_")
#' @param tol  Convergence tolerance (default: 1e-5)
#' @param max.iter  Maximum Lanczos iterations (default: 1000)
#' @param ...  Additional arguments passed to Seurat or scLean
#' @usage
#' \method{RunPCA}{Seurat}(object, npcs = 50, features = NULL,
#'   reduction.name = "pca", reduction.key = "PC_",
#'   tol = 1e-5, max.iter = 1000, ...)
#' @return Seurat object with PCA reduction
#' @examples
#' \donttest{
#' counts <- Matrix::rsparsematrix(200, 100, density = 0.15,
#'   dimnames = list(paste0("gene", 1:200), paste0("cell", 1:100)))
#' obj <- LoadScleanObject(counts)
#' obj <- NormalizeData(obj)
#' obj <- FindVariableFeatures(obj, nfeatures = 100)
#' obj <- ScaleData(obj)
#' obj <- RunPCA(obj, npcs = 10)
#' if (requireNamespace("Seurat", quietly = TRUE)) {
#'   Seurat::DimPlot(obj, reduction = "pca")
#' }
#' }
#' @seealso \code{\link{FindNeighbors.Seurat}}, \code{\link{FindClusters.Seurat}}
#' @export
RunPCA.Seurat <- function(object, npcs = 50, features = NULL, reduction.name = "pca", reduction.key = "PC_", tol = 1e-5, max.iter = 1000, ...) {
  assay <- SeuratObject::DefaultAssay(object)
  sc_assay <- object@assays[[assay]]
  if (inherits(sc_assay, "scLeanAssay")) {
    return(RunPCA.scLeanAssay(object, npcs = npcs, features = features,
      reduction.name = reduction.name, reduction.key = reduction.key,
      tol = tol, max.iter = max.iter, ...))
  }
  .seurat_original$RunPCA.Seurat(object, npcs = npcs, features = features,
    reduction.name = reduction.name, reduction.key = reduction.key,
    tol = tol, max.iter = max.iter, ...)
}

#' @export
RunPCA.scLeanAssay <- function(
    object,
    npcs = 50,
    features = NULL,
    reduction.name = "pca",
    reduction.key = "PC_",
    tol = 1e-5,
    max.iter = 1000,
    ...
) {
  if (inherits(object, "scLeanAssay")) {
    sc_assay <- object
    # For assay-only calls, we need cell/feature names from HDF5
  } else if (inherits(object, "Seurat")) {
    assay <- SeuratObject::DefaultAssay(object)
    sc_assay <- object@assays[[assay]]
    if (!inherits(sc_assay, "scLeanAssay")) {
      return(Seurat::RunPCA(object, npcs = npcs, features = features,
        reduction.name = reduction.name, reduction.key = reduction.key, ...))
    }
  } else {
    stop("object must be a Seurat object or scLeanAssay")
  }

  cpp_pca(
    hdf5_path       = sc_assay@hdf5_path,
    assay_group     = sc_assay@hdf5_group,
    npcs            = as.integer(npcs),
    tol             = tol,
    max_iter        = as.integer(max.iter)
  )

  # Read PCA results and add as DimReduc
  embeddings <- read_dense_matrix(
    sc_assay@hdf5_path,
    paste0(sc_assay@hdf5_group, "/reductions/pca/embeddings")
  )
  loadings <- read_dense_matrix(
    sc_assay@hdf5_path,
    paste0(sc_assay@hdf5_group, "/reductions/pca/loadings")
  )
  stdev <- read_hdf5_double(sc_assay@hdf5_path,
    paste0(sc_assay@hdf5_group, "/reductions/pca/stdev"))[seq_len(npcs)]

  colnames(embeddings) <- paste0(reduction.key, seq_len(npcs))
  colnames(loadings) <- paste0(reduction.key, seq_len(npcs))
  rownames(embeddings) <- read_strings_from_hdf5(sc_assay@hdf5_path, "/cells/names")
  rownames(loadings) <- read_strings_from_hdf5(sc_assay@hdf5_path, "/features/names")

  if (inherits(object, "Seurat")) {
    object@assays[[assay]] <- sc_assay
    object[[reduction.name]] <- SeuratObject::CreateDimReducObject(
      embeddings = embeddings,
      loadings   = loadings,
      stdev      = stdev,
      key        = reduction.key,
      assay      = assay
    )
    return(object)
  }
  return(sc_assay)
}
