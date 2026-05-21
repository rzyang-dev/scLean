# scLean 0.1.0

* **Documentation overhaul**: Added `@examples`, `@return`, and `@seealso` tags to
  all 22 user-facing R functions for CRAN compliance. Fixed R6 roxygen concatenation
  bug in `HDF5BackedMatrix` documentation. Replaced `@inheritParams` with explicit
  `@param` tags in interceptor S3 methods (FindMarkers.Seurat, FindClusters.Seurat,
  FindNeighbors.Seurat, RunPCA.Seurat).
* **New vignette**: `vignette("reference", package = "scLean")` — API cheat sheet
  with Data I/O, Pipeline, Integration, and Resource Control function tables.
* **Revised vignette**: `vignette("sclean", package = "scLean")` — rewrote Quick
  Start with evaluated code chunks, synthetic data, PerformanceSnapshot timing,
  and resource monitoring demonstrations.
* **README enhancements**: Added API Overview table, conceptual benchmarks section,
  vignette cross-references, and FAQ with 5 common questions.
* Initial release.
* Memory-efficient single-cell RNA-seq analysis with HDF5-backed storage.
* Disk-backed implementations of the standard Seurat pipeline:
  NormalizeData, FindVariableFeatures, ScaleData, RunPCA, FindNeighbors,
  FindClusters, and FindMarkers.
* Supports LogNormalize, RelativeCounts, and CLR normalization methods.
* Leiden and Louvain clustering via C++ implementation.
* Wilcoxon, t-test, and logistic regression differential expression tests.
* Integration with Seurat v5 Assay5 object model (LogMap support).
* **Resource monitoring**: `ResourceMonitor` with cross-platform free RAM detection,
  CPU load sensing, memory/CPU pressure computation, and bottleneck classification.
* **Intelligent scheduler**: `ChunkScheduler` adapts chunk size to available RAM,
  enforces 128 MB dense buffer cap, routes OOM-prone operations to sparse paths,
  and provides 3-level OOM degradation (dense → sparse → shrink → retry).
* **Thread governor**: `ThreadGovernor` adjusts parallelism based on system load
  and memory pressure.
* **New R functions**: `CheckResources()`, `RefreshMemory()`, `PerformanceSnapshot()`,
  `SetThreads()`, `cpp_resource_snapshot()`, `cpp_bottleneck_type()`,
  `cpp_set_max_dense_chunk_mb()`, `cpp_get_max_dense_chunk_mb()`.
* All 4 core operators (Normalize, Scale, VST, FindMarkers) wired to ChunkScheduler
  with OOM retry logic.
* FindMarkers scheduler integration removes hardcoded 500 gene/chunk limit,
  preventing ~4 GB buffer spikes on datasets with 1M+ cells.
* FindNeighbors and FindClusters operators wired to scheduler for
  memory-aware block sizing and bottleneck-driven algorithm selection.
* HDF5 write return-value checks added across all write paths to prevent
  silent data corruption on disk-full conditions.
* VST select_top_features() guards against negative/zero nfeatures (avoiding UB).
* ScaleData guards against NaN from floating-point roundoff in Welford variance.
* `SetThreads()` now properly propagates to all operators via `get_num_threads()`.
* R input validation added for `SetChunkSize()`, `SetThreads()`, and
  `CreateSCleanAssayFromHDF5()`.
* Documentation: README updated with resource management and monitoring sections.
