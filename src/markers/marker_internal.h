#ifndef SCLEAN_MARKER_INTERNAL_H
#define SCLEAN_MARKER_INTERNAL_H

#include <cstdint>
#include <vector>
#include <string>
#include <hdf5.h>
#include "scLean_types.h"
#include "marker_operator.h"

namespace sclean {

class HDF5File;

// ============================================================
// Pure statistical test functions (no HDF5, no file I/O)
// ============================================================

// Wilcoxon rank-sum test (Mann-Whitney U, normal approximation with tie correction).
// Returns two-tailed p-value in [0, 1].
double wilcoxon_pval(const std::vector<double>& group1,
                     const std::vector<double>& group2);

// Welch's t-test with Satterthwaite degrees of freedom approximation.
// Returns two-tailed p-value in [0, 1].
double ttest_pval(const std::vector<double>& group1,
                  const std::vector<double>& group2);

// Logistic regression score test (Rao score test approximation).
// Builds binary response (1 = ident_1, 0 = ident_2 or all others),
// computes correlation-based score statistic, and derives a chi-squared(1) p-value.
double logistic_regression_pval(const std::vector<double>& expression,
                                 const std::vector<int32>& labels,
                                 int32 ident_1, int32 ident_2);

// ============================================================
// Per-gene test orchestration
// ============================================================

// Split expression by cluster, compute pct expressed, log2FC, and run
// the selected statistical test. Filters by min_pct and logfc_threshold.
MarkerResult test_gene(const std::vector<double>& expression,
                        const std::vector<int32>& clusters,
                        int32 ident_1, int32 ident_2,
                        DETest test, double logfc_threshold, double min_pct);

// ============================================================
// Chunked gene processing (HDF5-dependent)
// ============================================================

// Read expression for a single gene from the HDF5-backed CSC matrix.
std::vector<double> read_gene_expression(HDF5File* file,
                                           const std::string& data_group,
                                           int64 gene_idx, int64 n_cells);

// Read a chunk of genes, test each one, and append results.
void process_gene_chunk(HDF5File* file,
                         const std::string& data_group,
                         int64 gene_start, int64 gene_count,
                         int64 n_cells,
                         const std::vector<int32>& clusters,
                         int32 ident_1, int32 ident_2,
                         DETest test, double logfc_threshold, double min_pct,
                         std::vector<MarkerResult>& results,
                         hid_t thread_file = -1);

} // namespace sclean

#endif // SCLEAN_MARKER_INTERNAL_H
