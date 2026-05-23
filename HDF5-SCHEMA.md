# HDF5 File Schema for scLean

This document specifies the HDF5 file layout that scLean reads and writes. Every group and dataset path listed here is part of the R-to-C++ API contract — both sides must agree on these paths for correct operation.

## Conventions

- **CSC Matrix Storage**: Sparse matrices are stored as Compressed Sparse Column (CSC) format using three datasets:
  - `data` — `float64` values (non-zero entries, length nnz)
  - `indices` — `int32` row indices (one per entry, length nnz)
  - `indptr` — `int64` column pointers (length n_cols + 1)
  - `row_ptr` (optional) — `int64` CSR row pointer cache, built lazily for O(1) row-slice queries. Not persisted.
- **Compression**: All datasets use gzip level 3.
- **Indexing**: All indices are 0-based (C++ convention). R code converts to 1-based when necessary.
- **Dimension order**: Matrices are `(n_features, n_cells)` — genes are rows, cells are columns.

## Full HDF5 Tree

```
/features/
  names               string[1D]     (n_genes,)    Gene names (row names)
  variable            int32[1D]      (n_genes,)    0/1 flag: is variable feature?
  residual_mean       float64[1D]    (n_genes,)    Per-gene mean (ScaleData output)
  residual_sd         float64[1D]    (n_genes,)    Per-gene SD (ScaleData output)
  vst_mean            float64[1D]    (n_genes,)    Per-gene VST mean (optional)
  vst_variance        float64[1D]    (n_genes,)    Per-gene VST variance (optional)

/cells/
  names               string[1D]     (n_cells,)    Cell barcodes (column names)

/assays/<assay>/layers/
  counts              CSC            (n_genes, n_cells)   Raw count matrix
  data                CSC            (n_genes, n_cells)   Normalized expression

/assays/<assay>/features/          (per-assay overrides of /features/ in some cases)
  residual_mean       float64[1D]    (n_genes,)    Per-assay scaling mean
  residual_sd         float64[1D]    (n_genes,)    Per-assay scaling SD

/assays/<assay>/reductions/
  pca/
    embeddings        float64[2D]    (n_cells, npcs)      PCA cell embeddings
    loadings          float64[2D]    (n_genes, npcs)      PCA gene loadings
    stdev             float64[1D]    (npcs,)              Singular values / sqrt(n_cells - 1)
  harmony/
    embeddings        float64[2D]    (n_cells, n_ccs)     MNN-corrected embeddings (named
                                                          "harmony" for Seurat compatibility;
                                                          actual algorithm is MNN)

/assays/<assay>/graphs/
  nn/
    indices           int32[2D]      (n_cells, k)   KNN indices (0-based)
    distances         float32[2D]    (n_cells, k)   KNN distances
  snn/                CSC            (n_cells, n_cells)  Shared nearest neighbor graph

/assays/<assay>/clusters/
  leiden              int32[1D]      (n_cells,)     Leiden cluster assignments (0-based)
  louvain             int32[1D]      (n_cells,)     Louvain cluster assignments (0-based)
```

## Writer/Reader Cross-Reference

| Path | Written by | Read by |
|------|-----------|---------|
| `/cells/names` | `CreateSCleanAssay`, `stream_10x_to_hdf5` | `CreateSCleanAssayFromHDF5`, `HDF5BackedMatrix` |
| `/features/names` | `CreateSCleanAssay`, `stream_10x_to_hdf5` | `CreateSCleanAssayFromHDF5`, `HDF5BackedMatrix`, `RunPCA.scLeanAssay` |
| `/assays/<assay>/layers/counts` | `CreateSCleanAssay`, `as.scLean` | All pipeline operators |
| `/assays/<assay>/layers/data` | `cpp_normalize` | `cpp_vst`, `cpp_scale`, `cpp_pca` |
| `/assays/<assay>/features/residual_mean` | `cpp_scale` | `cpp_pca` (on-the-fly centering) |
| `/assays/<assay>/features/residual_sd` | `cpp_scale` | `cpp_pca` (on-the-fly centering) |
| `/assays/<assay>/features/variable` | `cpp_vst` | `RunPCA.scLeanAssay` (feature subsetting) |
| `/assays/<assay>/features/vst_mean` | `cpp_vst` | (for diagnostics, not used in pipeline) |
| `/assays/<assay>/features/vst_variance` | `cpp_vst` | (for diagnostics, not used in pipeline) |
| `/assays/<assay>/reductions/pca/embeddings` | `cpp_pca` | `FindNeighbors.scLeanAssay`, `cpp_find_neighbors` |
| `/assays/<assay>/reductions/pca/loadings` | `cpp_pca` | `RunPCA.scLeanAssay` (for DimReduc object) |
| `/assays/<assay>/reductions/pca/stdev` | `cpp_pca` | `RunPCA.scLeanAssay` (for DimReduc object) |
| `/assays/<assay>/reductions/harmony/embeddings` | `cpp_integrate` | R integration code (for DimReduc object) |
| `/assays/<assay>/graphs/nn/indices` | `cpp_find_neighbors` | `FindNeighbors.scLeanAssay` |
| `/assays/<assay>/graphs/snn` | `cpp_find_neighbors` | `cpp_find_clusters` |
| `/assays/<assay>/clusters/leiden` | `cpp_find_clusters` | `FindClusters.scLeanAssay` |

## CSC Matrix Convention

Each CSC matrix is stored as a group containing exactly three datasets:

```
<group>/
  data       float64[nnz]     Non-zero values, sorted by row within each column
  indices    int32[nnz]       Row index for each non-zero entry
  indptr     int64[ncols+1]   Column offset pointer: col j spans [indptr[j], indptr[j+1])
```

- `n_rows` and `n_cols` are derived from dataset extents.
- The data type is always `float64` for values, `int32` for indices, `int64` for pointers.
- The `row_ptr` cache (CSR row pointers) may be built in memory by `HDF5CSCMatrix` for O(1) row access. It is never written back to HDF5.

## Optional vs. Required Paths

| Path | Always Present? | Notes |
|------|----------------|-------|
| `/cells/names` | Yes | Written during object creation |
| `/features/names` | Yes | Written during object creation |
| `/assays/<assay>/layers/counts` | Yes | Core assay data |
| `/assays/<assay>/layers/data` | No | Written by NormalizeData |
| `/assays/<assay>/features/residual_mean` | No | Written by ScaleData |
| `/assays/<assay>/features/variable` | No | Written by FindVariableFeatures |
| `/assays/<assay>/reductions/pca/*` | No | Written by RunPCA |
| `/assays/<assay>/reductions/harmony/*` | No | Written by IntegrateData.scLean |
| `/assays/<assay>/graphs/*` | No | Written by FindNeighbors |
| `/assays/<assay>/clusters/*` | No | Written by FindClusters |
