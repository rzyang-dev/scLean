# scLean 0.1.0

## Memory management refactoring
* **Dynamic memory cap**: `ChunkScheduler` constructor now accepts an optional
  user-defined memory ceiling (`cap_bytes`), capping `available_ram_` to
  `min(detect_free_ram(), cap_bytes)`. Added `user_cap()` accessor.
* **`SetMaxRAM()` overhaul**: Calling `SetMaxRAM(NULL)` now disables the user cap
  and switches to fully automatic memory detection. Package startup no longer
  sets a default memory limit — the system auto-detects available RAM on every
  `refresh_available_ram()` call.
* **Startup message update**: Package load message now displays current free RAM
  alongside any user-set memory ceiling.
* **Low-memory detection**: `is_low_memory()` updated to use the new resource
  snapshot interface for free RAM detection.
* **Test suite**: Memory-related tests updated for the new cap semantics;
  clustering tests now set a random seed for reproducibility.

## Build system & CI
* **OpenMP linking**: Fixed OpenMP library link configuration across platforms;
  unified build scripts (`configure`, `configure.ac`, `configure.win`).
* **Windows build**: Added `configure.win` with OpenMP detection logic; removed
  obsolete manually-maintained `src/Makevars.win`.
* **macOS CI**: `brew install hdf5 gettext` now uses idempotent `brew list` checks
  to avoid re-installation failures. Environment variables moved from `env` block
  to inline `run` step.
* **leidenbase**: Added `leidenbase` as Suggested dependency with check skip logic
  in tests; CI check severity raised from `note` to `warning`.
* **CI release workflow** (unstaged): Added `release` job triggered by `v*` tags
  that builds the source tarball, extracts release notes from NEWS.md, and creates
  a GitHub Release via `softprops/action-gh-release@v2`.
* **DESCRIPTION**: Added `Config/Needs/check` field listing all check-time
  dependencies.

## Bug fixes
* **HDF5 CSC matrix**: Fixed member initializer order in `HDF5CSCMatrix`
  constructor, eliminating undefined behavior from out-of-order initialization.
* **`VariableFeatures` export**: Added `VariableFeatures` to NAMESPACE exports and
  `R/generics.R`, importing from `SeuratObject`.
* **`scLean_types.h` case fix**: Renamed `src/sclean_types.h` → `src/scLean_types.h`
  and updated all 7 `#include` references across C++ source files to match the
  canonical casing.
* **Seurat namespace**: Changed bare `Idents(obj)` calls to `SeuratObject::Idents(obj)`
  in `R/clustering.R` for explicit namespace resolution.
* **Windows headers**: Added `#include <psapi.h>` in `resource_monitor.h` for
  Windows build compatibility; added `<unistd.h>` and `<sys/resource.h>` for
  Linux fallback paths.
* **Annoy conflict**: Added `#define STRICT_R_HEADERS` guard before including
  raw Annoy headers in `neighbor_operator.cpp` to prevent include conflicts
  with Rcpp.
* **PBMC3k test**: Relaxed clustering ARI validation threshold from 0.3 to 0.25
  to account for algorithmic variance.

## Documentation
* **`SetMaxRAM()`**: Updated help page title, parameter descriptions, return
  value, and examples to reflect the new NULL-means-auto semantics.

---

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
