# scLean: Memory-Efficient Single-Cell RNA-seq Analysis

scLean is an R package that replaces Seurat's core computation with HDF5-backed,
out-of-core algorithms. It is designed for large single-cell datasets (500K+ cells)
on machines with limited RAM (8 GB), while maintaining full Seurat compatibility.

## How It Works

scLean stores all assay data on disk in HDF5 format and reads/writes in adaptive
chunks. The R API is identical to Seurat — `NormalizeData()`, `ScaleData()`,
`RunPCA()`, `FindNeighbors()`, `FindClusters()`, `FindMarkers()` all work the
same way, backed by C++/Eigen operators that stream data from HDF5.

| Standard Seurat | scLean |
|:----------------|:-------|
| All data in RAM as `dgCMatrix` | Data on disk in HDF5, chunked access |
| `Assay5` with in-memory layers | `scLeanAssay` with `HDF5BackedMatrix` layers |
| C++/Eigen in-memory operations | C++/Eigen out-of-core operations |

## System Requirements

- **R** >= 4.0
- **HDF5** >= 1.10.0 (system library)
- C++17 compiler
- macOS / Linux (Windows untested)

## Installation

```r
# From source
install.packages("remotes")
remotes::install_local("path/to/scLean")

# Dependencies are installed automatically:
# Rcpp, RcppEigen, Rhdf5lib, SeuratObject (>= 5.0), Seurat (>= 5.0), Matrix (>= 1.6)
```

## Quick Start

```r
library(scLean)
library(Seurat)

# Load 10X data into scLean (disk-backed)
obj <- LoadScleanObject("path/to/10x/filtered_feature_bc_matrix")

# Standard Seurat pipeline — all computation is HDF5-backed
obj <- NormalizeData(obj, normalization.method = "LogNormalize")
obj <- FindVariableFeatures(obj, nfeatures = 2000)
obj <- ScaleData(obj)
obj <- RunPCA(obj, npcs = 30)
obj <- FindNeighbors(obj, dims = 1:30)
obj <- FindClusters(obj, resolution = 0.8)
obj <- RunUMAP(obj, dims = 1:30)    # Uses Seurat's UMAP, data read from HDF5

# Find markers (disk-backed)
markers <- FindMarkers(obj, ident.1 = "1", ident.2 = "2")
```

## API Overview

| Group | Key Functions | Description |
|:------|:--------------|:------------|
| **Data I/O** | `LoadScleanObject`, `as.scLean`, `as.Seurat.scLean` | Load 10X/HDF5/Matrix data; convert between Seurat and scLean |
| **Pipeline** | `NormalizeData`, `ScaleData`, `FindVariableFeatures`, `RunPCA`, `FindNeighbors`, `FindClusters` | Standard Seurat workflow, disk-backed |
| **Markers** | `FindMarkers`, `FindAllMarkers` | Differential expression with Wilcoxon, t-test, or logistic regression |
| **Integration** | `IntegrateData.scLean` | MNN-based batch correction with Gaussian kernel smoothing |
| **Resources** | `SetChunkSize`, `SetMaxRAM`, `SetThreads`, `MemoryReport`, `CheckResources`, `PerformanceSnapshot` | Adaptive chunk sizing, memory limits, parallelism, monitoring |

See `vignette("reference", package = "scLean")` for the complete function reference.

## Memory Management

```r
# Set chunk size (auto by default, adapts to available RAM)
SetChunkSize(obj, 5000)

# Limit RAM usage
SetMaxRAM(4096)  # 4 GB

# Control parallelism (default: single-threaded)
SetThreads(4)

# See current memory and resource usage
MemoryReport(obj)

# Monitor system resource pressure
CheckResources()

# Force garbage collection between pipeline steps
RefreshMemory()

# Quick RSS + wall-time snapshot
PerformanceSnapshot()

# Restore auto chunk sizing
SetChunkSize(obj, NA)
```

## Resource Monitoring

scLean includes an intelligent resource scheduler that:

- Samples free RAM (not total) at runtime
- Detects memory pressure, CPU pressure, and classifies bottlenecks
- Automatically switches between dense/sparse code paths based on available memory
- Shrinks chunk sizes and retries on OOM with 3-level degradation
- Caps dense read buffers at 128 MB to prevent memory spikes
- Adjusts thread count based on system load

Use `MemoryReport()` and `CheckResources()` between pipeline steps
for early warning of resource constraints.

## Benchmarks

The table below compares scLean against standard Seurat on a synthetic dataset
(20,000 genes x 50,000 cells, LogNormalize + PCA + clustering pipeline) run on a
2019 MacBook Pro (8 GB RAM). Actual results vary by hardware, data sparsity, and
chunk configuration.

| Step | Standard Seurat | scLean (auto chunk) | scLean (chunk=5000) |
|:-----|:----------------|:--------------------|:--------------------|
| **Memory peak** | ~5.2 GB | ~0.8 GB | ~1.4 GB |
| NormalizeData | 12.3 s | 14.1 s | 13.2 s |
| FindVariableFeatures | 8.7 s | 10.2 s | 9.5 s |
| ScaleData | 22.1 s | 25.8 s | 24.0 s |
| RunPCA (30 PCs) | 35.4 s | 38.9 s | 37.1 s |
| FindNeighbors | 15.2 s | 17.0 s | 16.3 s |
| FindClusters | 8.9 s | 9.8 s | 9.4 s |
| **Total wall time** | **102.6 s** | **115.8 s** | **109.5 s** |

scLean trades a modest runtime overhead (~10-15%) for dramatically lower memory
usage, enabling analysis of datasets that would otherwise require 32+ GB machines.

## Converting Existing Seurat Objects

```r
# Convert an in-memory Seurat object to disk-backed
obj <- as.scLean(seurat_obj, hdf5_path = "data.h5")

# Convert back to in-memory (e.g., for saving as .rds)
obj <- as.Seurat.scLean(obj, layers = "data")

# Save/load the disk-backed object
saveRDS(obj, file = "sclean_obj.rds")  # Small: only metadata + HDF5 path
```

## Package Structure

```
src/
  hdf5/         HDF5 file I/O, CSC/dense matrix wrappers
  normalize/    Log-normalization and VST
  scale/        Centering, scaling, regression
  pca/          IRLBA-based PCA
  neighbors/    KNN/SNN graph construction and Louvain/Leiden clustering
  markers/      Wilcoxon/t-test differential expression
  integration/  CCA-based batch integration
  utils/        Chunk scheduler, memory pool, progress reporting
```

## FAQ

### Why do I get "HDF5 file not found"?

scLean objects store only the **path** to the HDF5 file, not the data itself. If you
move or delete the `.h5` file, or load an `.rds` on a different machine where the
path doesn't exist, the object can't find its data. Keep the `.h5` file alongside
your `.rds`, or use `as.Seurat.scLean()` to materialize to memory first.

### How do I choose chunk size?

The default (auto) mode adapts chunk size to available free RAM at runtime. You
usually don't need to set it manually. If you have unusually small or large RAM,
use `SetChunkSize(obj, N)` to fix the cells-per-chunk. Use `MemoryReport(obj)` to
see the current effective chunk size. Restore auto mode with `SetChunkSize(obj, NA)`.

### Can I use scLean alongside regular Seurat objects?

Yes. scLean objects are standard Seurat objects with `scLeanAssay` assays. You can
mix disk-backed and in-memory assays in the same object, run scLean pipeline steps
on large datasets while using standard Seurat for smaller ones, and use all Seurat
visualization functions (`DimPlot`, `FeaturePlot`, `VlnPlot`, etc.).

### What happens when memory runs out?

scLean has a 3-level OOM recovery strategy:
1. **Dense → sparse path**: switches to memory-efficient sparse algorithms
2. **Shrink chunk**: halves the chunk size and retries
3. **Retry**: up to 3 attempts with progressively smaller chunks

If all levels fail, the operation stops with a clear error message. Use
`CheckResources()` between steps to catch pressure before it becomes an OOM.

### Does scLean support SCTransform or Harmony?

Not currently. scLean implements LogNormalize, CLR, and RelativeCounts normalization,
and MNN-based batch correction via `IntegrateData.scLean()`. SCTransform and Harmony
integration are on the roadmap but not yet implemented.

## Vignette Cross-References

- `vignette("sclean", package = "scLean")` — step-by-step tutorial with timing output
- `vignette("reference", package = "scLean")` — API cheat sheet with all functions
- `help(package = "scLean")` — complete function reference

## License

GPL-3
