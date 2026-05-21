#' Find markers (disk-backed)
#'
#' Intercepts Seurat::FindMarkers to route scLean-backed objects
#' to the disk-based implementation. Uses chunked gene-wise computation
#' with automatic chunk sizing via the resource scheduler. Supports
#' Wilcoxon rank-sum, t-test, and logistic regression tests.
#'
#' @param object  A Seurat object
#' @param ident.1  Identity class to define markers for
#' @param ident.2  Identity class for comparison (NULL = all others)
#' @param test.use  Statistical test: "wilcox" (default), "t", or "LR"
#' @param logfc.threshold  Minimum log2 fold change (default: 0.25)
#' @param min.pct  Minimum fraction of cells expressing the gene (default: 0.1)
#' @param ...  Additional arguments
#' @usage \\method{FindMarkers}{Seurat}(object, ident.1, ident.2 = NULL, test.use = "wilcox", logfc.threshold = 0.25, min.pct = 0.1, ...)
#' @return Data frame of marker genes
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
#' obj <- FindClusters(obj, resolution = 0.5)
#' markers <- FindMarkers(obj, ident.1 = "1", ident.2 = "2")
#' head(markers)
#' }
#' @seealso \code{\link{FindAllMarkers.scLeanAssay}}
#' @export
FindMarkers.Seurat <- function(object, ident.1, ident.2 = NULL, test.use = "wilcox", logfc.threshold = 0.25, min.pct = 0.1, ...) {
  assay <- SeuratObject::DefaultAssay(object)
  sc_assay <- object@assays[[assay]]
  if (inherits(sc_assay, "scLeanAssay")) {
    return(FindMarkers.scLeanAssay(object, ident.1 = ident.1,
      ident.2 = ident.2, test.use = test.use,
      logfc.threshold = logfc.threshold, min.pct = min.pct, ...))
  }
  .seurat_original$FindMarkers.Seurat(object, ident.1 = ident.1,
    ident.2 = ident.2, test.use = test.use,
    logfc.threshold = logfc.threshold, min.pct = min.pct, ...)
}

#' @export
FindMarkers.scLeanAssay <- function(
    object,
    ident.1,
    ident.2 = NULL,
    test.use = "wilcox",
    logfc.threshold = 0.25,
    min.pct = 0.1,
    ...
) {
  assay <- SeuratObject::DefaultAssay(object)
  sc_assay <- object@assays[[assay]]

  if (!inherits(sc_assay, "scLeanAssay")) {
    return(Seurat::FindMarkers(object, ident.1 = ident.1, ident.2 = ident.2,
      test.use = test.use, logfc.threshold = logfc.threshold,
      min.pct = min.pct, ...))
  }

  clusters <- as.integer(SeuratObject::Idents(object))
  test_map <- c(wilcox = 0L, t = 1L, LR = 2L)

  result <- cpp_find_markers(
    hdf5_path       = sc_assay@hdf5_path,
    assay_group     = sc_assay@hdf5_group,
    clusters        = clusters,
    ident_1         = as.integer(ident.1),
    ident_2         = if (is.null(ident.2)) -1L else as.integer(ident.2),
    test_type       = test_map[test.use],
    logfc_threshold = logfc.threshold,
    min_pct         = min.pct
  )

  # Convert to R data frame
  gene_names <- read_strings_from_hdf5(sc_assay@hdf5_path, "/features/names")
  # result is a list of lists from Rcpp; convert to data frame
  df <- do.call(rbind, lapply(result, as.data.frame))
  if (nrow(df) > 0) {
    df$gene <- gene_names[df$gene_idx + 1]
    df$gene_idx <- NULL
  }

  return(df)
}

#' Find all markers (disk-backed)
#'
#' Runs FindMarkers for every cluster against all other cells.
#' Returns a named list of data frames, one per cluster.
#'
#' @param object  Seurat object
#' @param test.use  Statistical test to use ("wilcox", "t", "LR")
#' @param logfc.threshold  Log2 fold change threshold
#' @param min.pct  Minimum fraction of cells expressing the gene
#' @param ...  Additional arguments passed to FindMarkers
#' @return Named list of marker data frames, one per cluster
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
#' obj <- FindClusters(obj, resolution = 0.5)
#' all_markers <- FindAllMarkers(obj)
#' names(all_markers)
#' }
#' @seealso \code{\link{FindMarkers.Seurat}}
#' @export
FindAllMarkers.scLeanAssay <- function(
    object,
    test.use = "wilcox",
    logfc.threshold = 0.25,
    min.pct = 0.1,
    ...
) {
  assay <- SeuratObject::DefaultAssay(object)
  sc_assay <- object@assays[[assay]]

  if (!inherits(sc_assay, "scLeanAssay")) {
    return(Seurat::FindAllMarkers(object, test.use = test.use,
      logfc.threshold = logfc.threshold, min.pct = min.pct, ...))
  }

  idents <- SeuratObject::Idents(object)
  all_clusters <- sort(unique(idents))

  results <- lapply(all_clusters, function(cl) {
    FindMarkers.scLeanAssay(object, ident.1 = cl,
      test.use = test.use, logfc.threshold = logfc.threshold,
      min.pct = min.pct, ...)
  })

  names(results) <- all_clusters
  return(results)
}
