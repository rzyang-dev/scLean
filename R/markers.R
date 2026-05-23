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
#' @importFrom stats setNames
#' @usage
#' \method{FindMarkers}{Seurat}(object, ident.1, ident.2 = NULL,
#'   test.use = "wilcox", logfc.threshold = 0.25, min.pct = 0.1, ...)
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
#' idents <- levels(SeuratObject::Idents(obj))
#' markers <- FindMarkers(obj, ident.1 = idents[1], ident.2 = NULL)
#' head(markers)
#' }
#' @seealso \code{\link{FindAllMarkers.scLeanAssay}}
#' @export
FindMarkers.Seurat <- function(
    object,
    ident.1,
    ident.2 = NULL,
    test.use = "wilcox",
    logfc.threshold = 0.25,
    min.pct = 0.1,
    ...
) {
  sc_assay <- extract_sc_assay(object)
  if (!is.null(sc_assay)) {
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
  sc_assay <- extract_sc_assay(object, assay)

  idents <- SeuratObject::Idents(object)
  clusters <- as.integer(idents)
  # Test-type mapping: integer codes for C++ interop.
  # Unrecognized test.use values return NA_integer_, which is passed silently
  # to C++ — validation happens in the C++ layer.
  test_map <- c(wilcox = 0L, t = 1L, LR = 2L)

  ident_map <- setNames(seq_along(levels(idents)), levels(idents))
  ident_1_int <- ident_map[as.character(ident.1)]
  if (is.na(ident_1_int)) stop("ident.1 '", ident.1, "' not found in Idents")
  # ident.2=NULL sentinel: -1L in C++ signals "test against all other clusters".
  # -1L was chosen because 0 is a valid cluster ID (C++ uses 0-based indices).
  ident_2_int <- if (is.null(ident.2)) -1L else {
    val <- ident_map[as.character(ident.2)]
    if (is.na(val)) stop("ident.2 '", ident.2, "' not found in Idents")
    val
  }

  result <- cpp_find_markers(
    hdf5_path       = sc_assay@hdf5_path,
    assay_group     = sc_assay@hdf5_group,
    clusters        = clusters,
    ident_1         = ident_1_int,
    ident_2         = ident_2_int,
    test_type       = test_map[test.use],
    logfc_threshold = logfc.threshold,
    min_pct         = min.pct
  )

  # Convert to R data frame
  gene_names <- read_strings_from_hdf5(sc_assay@hdf5_path, "/features/names")
  # result is a list of per-gene result lists from C++.
  # Each element becomes a one-row data.frame via as.data.frame();
  # rbind assembles them into a single data.frame.
  # Gene names are resolved post-hoc: gene_idx is 0-based (C++), +1 for R indexing.
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
  sc_assay <- extract_sc_assay(object, assay)

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

#' @export
#' @rdname FindAllMarkers.scLeanAssay
FindAllMarkers.Seurat <- function(
    object,
    test.use = "wilcox",
    logfc.threshold = 0.25,
    min.pct = 0.1,
    ...
) {
  sc_assay <- extract_sc_assay(object)
  if (!is.null(sc_assay)) {
    return(FindAllMarkers.scLeanAssay(object, test.use = test.use,
      logfc.threshold = logfc.threshold, min.pct = min.pct, ...))
  }
  Seurat::FindAllMarkers(object, test.use = test.use,
    logfc.threshold = logfc.threshold, min.pct = min.pct, ...)
}
