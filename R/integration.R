#' Integrate data across batches (MNN-based batch correction)
#'
#' Uses mutual nearest neighbors to correct batch effects on PCA embeddings.
#' All computation is disk-backed via HDF5.
#'
#' @param object  Seurat object with scLeanAssay
#' @param batch   Batch variable name in metadata (default: "batch")
#' @param n.ccs   Number of dimensions to use for correction (default: 30)
#' @param n.mnn   Number of mutual nearest neighbors (default: 20)
#' @param sigma   Gaussian kernel bandwidth for correction smoothing (default: 0.1)
#' @param max.iter  Maximum correction iterations (default: 2)
#' @param ...  Additional arguments (reserved)
#' @return Seurat object with "harmony" reduction containing corrected embeddings
#' @export
IntegrateData.scLean <- function(
    object,
    batch = "batch",
    n.ccs = 30,
    n.mnn = 20,
    sigma = 0.1,
    max.iter = 2,
    ...
) {
  assay <- SeuratObject::DefaultAssay(object)
  sc_assay <- object@assays[[assay]]

  if (!inherits(sc_assay, "scLeanAssay")) {
    stop("Integration requires a scLeanAssay")
  }

  # --- Input validation ---
  n.ccs <- as.integer(n.ccs)
  if (is.na(n.ccs) || n.ccs < 1) stop("n.ccs must be a positive integer")

  n.mnn <- as.integer(n.mnn)
  if (is.na(n.mnn) || n.mnn < 1) stop("n.mnn must be a positive integer")

  if (!is.numeric(sigma) || sigma <= 0) stop("sigma must be a positive number")

  max.iter <- as.integer(max.iter)
  if (is.na(max.iter) || max.iter < 1) stop("max.iter must be a positive integer")

  batch_labels <- object@meta.data[[batch]]
  if (is.null(batch_labels)) {
    stop("Batch variable '", batch, "' not found in object metadata")
  }

  batch_int <- as.integer(as.factor(batch_labels))

  cpp_integrate(
    hdf5_path    = sc_assay@hdf5_path,
    assay_group  = sc_assay@hdf5_group,
    batch_labels = batch_int,
    n_ccs        = n.ccs,
    n_mnn        = n.mnn,
    sigma        = sigma,
    max_iter     = max.iter
  )

  # Read corrected embeddings
  corr_path <- paste0(sc_assay@hdf5_group, "/reductions/harmony/embeddings")
  corrected <- read_dense_matrix(sc_assay@hdf5_path, corr_path)

  # Set dimnames required by CreateDimReducObject
  cell_names <- read_strings_from_hdf5(sc_assay@hdf5_path, "/cells/names")
  rownames(corrected) <- cell_names
  colnames(corrected) <- paste0("harmony_", seq_len(ncol(corrected)))

  object@assays[[assay]] <- sc_assay
  object[["harmony"]] <- SeuratObject::CreateDimReducObject(
    embeddings = corrected,
    key        = "HARMONY_",
    assay      = assay
  )

  return(object)
}
