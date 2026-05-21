# Re-export Seurat generics so scLean can register S3 methods for them.
# These are the core pipeline functions that scLean overrides with
# disk-backed implementations when the assay is scLeanAssay.

#' @importFrom Seurat NormalizeData
#' @export
Seurat::NormalizeData

#' @importFrom Seurat ScaleData
#' @export
Seurat::ScaleData

#' @importFrom Seurat FindVariableFeatures
#' @export
Seurat::FindVariableFeatures

#' @importFrom Seurat RunPCA
#' @export
Seurat::RunPCA

#' @importFrom Seurat FindNeighbors
#' @export
Seurat::FindNeighbors

#' @importFrom Seurat FindClusters
#' @export
Seurat::FindClusters

#' @importFrom Seurat FindMarkers
#' @export
Seurat::FindMarkers

#' @importFrom Seurat FindAllMarkers
#' @export
Seurat::FindAllMarkers

#' @importFrom SeuratObject VariableFeatures
#' @export
SeuratObject::VariableFeatures
