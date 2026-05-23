#ifndef SCLEAN_NORMALIZE_INTERNAL_H
#define SCLEAN_NORMALIZE_INTERNAL_H

#include <cstdint>
#include <vector>
#include "scLean_types.h"

namespace sclean {

// Forward declarations
enum class NormalizeMethod : int;
class HDF5SparseWriter;

// ===================================================================
// normalize_compute.cpp — pure computation (no HDF5 I/O)
// ===================================================================

/// Compute per-column sums from sparse CSC data chunk.
/// @param data          non-zero values for the chunk
/// @param local_indptr  indptr for columns in this chunk (relative to chunk start)
/// @param col_count     number of columns in this chunk
/// @param out_sums      output per-column sums (size >= col_count)
void compute_column_sums(const double* data, const int64* local_indptr,
                         int64 col_count, double* out_sums);

/// Compute per-column log sums and nonzero counts from sparse CSC data chunk.
void compute_column_logsums(const double* data, const int64* local_indptr,
                            int64 col_count, double* out_log_sums,
                            int* out_nz_counts);

/// Convert accumulated log sums and nonzero counts to geometric means.
void finalize_geometric_means(const double* log_sums, const int* nz_counts,
                              int64 n_cells, double* out_geo_means);

// ===================================================================
// normalize_apply.cpp — apply normalization to data chunks
// ===================================================================

/// Apply normalization to a sparse CSC chunk, writing output via writer.
/// Handles LogNormalize, RelativeCounts, and CLR methods.
void normalize_sparse_chunk(NormalizeMethod method,
    const double* in_data, const int32* in_indices,
    const int64* in_indptr, int64 col_count,
    const double* size_factors, double scale_factor, bool do_pseudocount,
    HDF5SparseWriter* writer);

/// Apply normalization to a dense column-major chunk, writing sparse output
/// via writer (zeros are dropped).
void normalize_dense_chunk(NormalizeMethod method,
    const double* in_buf, int64 n_rows, int64 n_cols,
    const double* size_factors, double scale_factor, bool do_pseudocount,
    HDF5SparseWriter* writer);

// ===================================================================
// vst_stats.cpp — pure mean/variance computation (no HDF5 I/O)
// ===================================================================

/// Per-gene Welford accumulator for online mean/variance computation.
struct GeneWelford {
    double mean = 0.0;
    double m2 = 0.0;
    int64 count = 0;

    /// Update running stats with a single observation.
    void update(double x);

    /// Merge another GeneWelford accumulator into this one
    /// (parallel Welford merge formula).
    void merge(const GeneWelford& other);

    /// Compute unbiased sample variance from accumulated stats.
    double variance() const;
};

/// Accumulate per-gene stats from a dense column-major chunk
/// (n_genes rows x n_cols columns). Merges into existing stats.
void accumulate_dense_chunk(const double* buf, int64 n_genes, int64 n_cols,
                            std::vector<GeneWelford>& stats);

/// Accumulate per-gene stats from a sparse CSC chunk.
/// Processes only non-zero entries.
void accumulate_sparse_chunk(const double* vals, const int32* idxs,
                             const int64* indptr, int64 col_count,
                             std::vector<GeneWelford>& stats);

/// Merge thread-local stats arrays into global stats (parallel reduction).
void merge_stats_arrays(const std::vector<GeneWelford>& local,
                        std::vector<GeneWelford>& global, int64 n_genes);

/// Extract means and variances from GeneWelford arrays into output vectors.
void finalize_stats_arrays(const std::vector<GeneWelford>& stats,
                           std::vector<double>& means,
                           std::vector<double>& variances);

// ===================================================================
// vst_select.cpp — LOESS fitting and feature selection (no HDF5 I/O)
// ===================================================================

/// Bin-based LOESS approximation for mean-variance trend.
/// Takes log10(mean) and log10(variance), produces fitted log10(variance).
void fit_loess_binned(const std::vector<double>& log_means,
                      const std::vector<double>& log_variances,
                      std::vector<double>& fitted,
                      int n_bins, double span);

/// Select top N variable features by standardized variance.
/// Sets variable[i] = 1 for selected features.
void select_top_features(const std::vector<double>& vst_variances,
                         std::vector<int8_t>& variable, int n_select);

} // namespace sclean

#endif // SCLEAN_NORMALIZE_INTERNAL_H
