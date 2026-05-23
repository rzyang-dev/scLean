# Known Issues & Design Limitations

## Active Bugs

| ID | Issue | Symptoms | Workaround | Severity |
|----|-------|----------|------------|----------|
| 1 | VST returns 0 variable features on >10K cells | `FindVariableFeatures()` selects nothing or few features | Use smaller nfeatures or pre-filter genes | HIGH |
| 2 | FindAllMarkers "no slot" error on scLean objects | Error when calling `FindAllMarkers()` | Use per-cluster `FindMarkers()` in a loop | HIGH |
| 3 | as.Seurat.scLean v5 compatibility | Converted objects may be rejected by Seurat v5 internals | Explicitly pass `layers="data"` | MEDIUM |
| 4 | CellRanger .h5 loading reads entire matrix into memory | Loading from CellRanger `.h5` files loads everything into R memory before writing to HDF5, defeating streaming | Use 10X directory format instead of `.h5` | MEDIUM |
| 5 | Leiden produces excessive clusters on some datasets | Clustering with resolution=0.8 can produce 300+ clusters on 28K cells | Try lower resolution, or use Louvain algorithm instead of Leiden | MEDIUM |
| 6 | DE returns 0 significant genes on real data | `FindMarkers()` returns empty data frame on real datasets | Adjust logfc.threshold or min.pct thresholds | MEDIUM |
| 7 | MemoryReport layer clobbering on multi-assay objects | When a Seurat object has multiple scLeanAssays, `MemoryReport()` only shows layers for the last assay | Call `MemoryReport()` on individual assays | LOW |
| 8 | Performance monitoring on large datasets | `MemoryReport` and `PerformanceSnapshot` may produce incomplete data for very large HDF5 files | Use OS-level tools (e.g., `ps`) for detailed monitoring | LOW |

## Design Limitations

These are not bugs — they are intentional design decisions or known scope limitations. They are documented here because they may look like bugs to new users.

| Limitation | Detail |
|-----------|--------|
| No materialized `scale.data` layer | `ScaleData` stores only per-gene mean/SD in HDF5. There is no scaled expression matrix on disk. PCA applies centering on-the-fly during Lanczos iteration. This saves substantial disk space and I/O but means you cannot inspect scaled values directly. |
| PCA embeddings always materialized in RAM | PCA embeddings and loadings are read fully into R memory after computation (required by Seurat's `CreateDimReducObject`). For 500K cells × 30 PCs, this is ~120 MB. |
| Read-only matrix proxy | `HDF5BackedMatrix` (the `[` operator) is read-only. There is no assignment method. All writes go through C++ pipeline operators. |
| Contiguous-only subsetting | `HDF5BackedMatrix[i, j]` only supports contiguous index ranges (e.g., `1:100`, not `c(1, 5, 10)`). Non-contiguous subsets materialize the full matrix to memory via `as.matrix()`. |
| Transpose materializes to memory | `t(HDF5BackedMatrix)` reads the entire matrix into memory, transposes it, and returns an in-memory dense matrix. |
| MNN writes to "harmony" reduction | `IntegrateData.scLean()` uses the MNN (mutual nearest neighbors) algorithm but stores results under the "harmony" DimReduc key. This is for Seurat ecosystem compatibility: Seurat's `DimPlot` and `FeaturePlot` look for the "harmony" reduction. The actual Harmony algorithm is not used here. |
| `read_vector_int8` reads `int32` | The R function `read_vector_int8()` is misnamed — it actually reads 4-byte integers from HDF5. Variable-feature flags (0/1) work correctly despite the type mismatch. Renaming would break RcppExports generation. |
| `vars.to.regress` parameter accepted but unused | `ScaleData.scLeanAssay` accepts a `vars.to.regress` parameter for Seurat API compatibility, but regression against covariates is not yet implemented in the C++ operator. The parameter is silently ignored. |
| `in_memory` parameter accepted but unused | Several pipeline R functions accept an `in_memory` parameter, but the in-memory computation path is not yet implemented. All computation uses HDF5-backed streaming. |
| `write_cols`/`write_rows` throw unimplemented errors | `HDF5CSCMatrix` supports reading from HDF5 but writing new columns/rows back is not supported. Writes are done by creating new datasets via pipeline operators. |
| CellRanger .h5 loads via Seurat | The CellRanger `.h5` format is read using `Seurat::Read10X_h5()` which loads the entire matrix into R memory. A direct streaming reader from CellRanger .h5 to scLean .h5 is planned but not yet implemented. |
| Global per-session configuration | `SetChunkSize`, `SetMaxRAM`, `SetThreads`, and `SetVerbose` use R global options (`scLean.*`) shared across all scLean objects in the same R session. Per-object configuration is not supported. |
| No SCTransform support | scLean does not implement SCTransform (regularized negative binomial regression). Use `NormalizeData` (LogNormalize, CLR, or RelativeCounts) instead. |
| No Harmony support | scLean does not implement the Harmony algorithm. The "harmony" reduction contains MNN-corrected embeddings from `IntegrateData.scLean()`. |
| Windows untested | The package has not been tested on Windows. The C++ code includes Windows-compatible system calls, but full CI testing on Windows is pending. |
| Single-threaded by default | The default thread count is 1 for safety with non-threadsafe HDF5 builds. Use `SetThreads(0)` to auto-detect physical cores. |
