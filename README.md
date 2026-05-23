# scLean — Memory-Efficient Single-Cell Analysis for R (Discontinued)

**scLean is no longer actively developed.** Existing installations will continue to
work, but no further updates, bug fixes, or feature additions are planned. Users are
encouraged to migrate to [Scanpy](https://scanpy.readthedocs.io/) for new projects.

## Why This Project Is Being Discontinued

scLean was an attempt to build a drop-in replacement for Seurat v5 that stores
assay data on disk (HDF5) instead of in memory, targeting large single-cell datasets
on machines with limited RAM. After extensive benchmarking against real-world data
and a detailed assessment of the codebase, several fundamental issues make continued
development unsustainable:

### 1. HDF5-backed CSC access has inherent I/O bottlenecks that can't be cheaply fixed

scRNA-seq pipelines are naturally row-oriented (genes are the unit of differential
expression). Reading rows from a CSC sparse matrix on HDF5 requires a binary search
over column pointers for every non-zero entry (`src/hdf5/hdf5_csc_read.cpp:146-167`).
This adds O(nnz × log n_cols) overhead per chunk, which becomes the dominant cost
as cell count grows. Materializing CSR row pointers in memory would work, but defeats
the purpose of disk backing by doubling the memory footprint.

### 2. Performance on large-scale data does not beat established alternatives

| Data Scale | scLean | Seurat (with sketch) | Scanpy (in-memory) | Scanpy (backed) |
|-----------:|--------|----------------------|--------------------|--------------------|
| <10K cells | Fine, but overkill | Fine, few seconds | Fine, few seconds | Fine |
| 10K-30K cells | Works, ~200s pipeline | Works, similar time | Faster | Comparable |
| 100K+ cells | Very slow (hours) | Use sketch/subsample | Needs HPC | Comparable or better |
| 500K+ cells | Impractical | Not possible | Not possible | Possible with out-of-core |

The 500K+ cells on 8 GB claim in earlier versions was aspirational. While memory
usage is indeed bounded by the 128 MB chunk cap, wall-clock time degrades severely
due to I/O amplification—chunk count grows linearly with cell count and so does
the binary-search overhead. A full pipeline on 500K cells would take 8-12 hours
on consumer hardware, which is not practical for iterative analysis.

### 3. The codebase has reached a complexity level that is hard to maintain alone

Multi-language codebases (R + C++ + HDF5 C API + Eigen) with tight coupling through
S3 dispatch, Rcpp, and an HDF5 schema contract have high maintenance costs. The
current test suite is thin, CI covers only smoke tests, and real-data validation
has surfaced bugs that require deep changes to the C++ layer to fix properly.

### 4. Scanpy is the better choice for most users

Scanpy offers:
- A mature, well-maintained Python ecosystem with thousands of contributors
- Out-of-core (backed) mode via AnnData's HDF5 layer that handles large data well
- A much larger community for troubleshooting and support
- Active development with regular releases, comprehensive documentation, and tutorials
- No system dependency headaches (no need for HDF5 system libraries or C++ compilers)

For R users who prefer to stay in R, consider [BPCells](https://github.com/bnprks/BPCells),
which is more mature than scLean and addresses similar memory constraints.

## What Still Works

If you already have scLean installed and are using it on modest datasets (<30K cells),
the core pipeline (LoadScleanObject → NormalizeData → ScaleData → RunPCA →
FindNeighbors → FindClusters) functions correctly on macOS and Linux. Known issues
are documented in [KNOWN-ISSUES.md](KNOWN-ISSUES.md). No fixes are forthcoming.

## System Requirements

- R >= 4.0
- HDF5 >= 1.10.0 (system library)
- C++17 compiler
- macOS / Linux (Windows untested)

## Installation (from source, last working commit)

```r
# First-time setup
./cleanup && ./configure

# Regenerate Rcpp bindings if needed
Rscript -e 'Rcpp::compileAttributes()'

# Build and install
R CMD INSTALL .
```

## License

GPL-3
