#ifndef SCLEAN_SCALE_INTERNAL_H
#define SCLEAN_SCALE_INTERNAL_H

#include <cstdint>
#include <vector>
#include "scLean_types.h"

namespace sclean {

// ===================================================================
// scale_compute.cpp — pure computation (no HDF5 I/O)
// ===================================================================

/// Compute mean and standard deviation for a single row of data
/// using Welford's online algorithm.
void compute_row_mean_sd(const double* row_data, int64 n_cols,
                         double& mean, double& sd);

/// Apply centering and/or scaling to non-zero entries of a sparse column.
/// Collects non-zero results into out_vals/out_idx (cleared first).
/// Returns the number of non-zero output entries.
int64 scale_sparse_column(const int32* idxs, const double* vals,
                          int64 col_start, int64 col_end,
                          const double* means, const double* sds,
                          bool do_center, bool do_scale,
                          std::vector<double>& out_vals,
                          std::vector<int32>& out_idx);

} // namespace sclean

#endif // SCLEAN_SCALE_INTERNAL_H
