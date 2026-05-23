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
RunPCA.Seurat <- function(
    object,
    npcs = 50,
    features = NULL,
    reduction.name = "pca",
    reduction.key = "PC_",
    tol = 1e-5,
    max.iter = 1000,
    ...
) {
  sc_assay <- extract_sc_assay(object)
  if (!is.null(sc_assay)) {
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
  sc_assay <- extract_sc_assay(object)

  if (is.null(features)) {
    # Feature resolution chain:
    # 1. Try VariableFeatures(object) on the Seurat object
    # 2. Fall back to VariableFeatures(sc_assay) if the first returned empty
    # 3. If both fail, feature_indices stays empty — C++ runs PCA on all genes
    if (inherits(object, "Seurat")) {
      features <- tryCatch(SeuratObject::VariableFeatures(object), error = function(e) character(0))
    }
    if (length(features) == 0) {
      features <- SeuratObject::VariableFeatures(sc_assay)
    }
  }

  feature_indices <- integer(0)
  if (length(features) > 0) {
    all_genes <- read_strings_from_hdf5(sc_assay@hdf5_path, "/features/names")
    # Index conversion: R is 1-based, C++ is 0-based.
    # which() returns 1-based indices; subtract 1L for C++ interop.
    feature_indices <- as.integer(which(all_genes %in% features) - 1L)
  }

  cpp_pca(
    hdf5_path       = sc_assay@hdf5_path,
    assay_group     = sc_assay@hdf5_group,
    npcs            = as.integer(npcs),
    tol             = tol,
    max_iter        = as.integer(max.iter),
    feature_indices = feature_indices
  )

  # Read PCA results and add as DimReduc.
  # Embeddings and loadings are always fully read into memory after computation.
  # This is because Seurat's CreateDimReducObject requires in-memory matrices.
  # For 500K cells x 30 PCs, the embedding matrix alone is ~120 MB — this is
  # the primary in-memory cost of RunPCA.
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

  result <- reinstall_assay(object, sc_assay)
  if (inherits(result, "Seurat")) {
    result[[reduction.name]] <- SeuratObject::CreateDimReducObject(
      embeddings = embeddings,
      loadings   = loadings,
      stdev      = stdev,
      key        = reduction.key,
      assay      = SeuratObject::DefaultAssay(result)
    )
    return(result)
  }
  return(result)
}
