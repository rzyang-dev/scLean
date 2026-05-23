#include "marker_internal.h"
#include "hdf5/hdf5_file.h"
#include "hdf5/hdf5_csc_matrix.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace sclean {

// ============================================================
// Gene expression reading
// ============================================================

std::vector<double> read_gene_expression(
    HDF5File* file, const std::string& data_group,
    int64 gene_idx, int64 n_cells) {

    std::vector<double> expr(n_cells, 0.0);

    // Read a single row from the CSC matrix
    auto mat = file->open_csc_matrix(data_group);
    mat->read_rows(expr.data(), gene_idx, 1, 0, n_cells);

    return expr;
}

// ============================================================
// Per-gene test orchestration
// ============================================================

MarkerResult test_gene(
    const std::vector<double>& expression,
    const std::vector<int32>& clusters,
    int32 ident_1, int32 ident_2,
    DETest test, double logfc_threshold, double min_pct) {

    int64 n = static_cast<int64>(expression.size());

    // Split into groups
    std::vector<double> group1, group2;
    bool is_vs_all = (ident_2 < 0);

    for (int64 i = 0; i < n; ++i) {
        if (clusters[i] == ident_1) {
            group1.push_back(expression[i]);
        } else if (is_vs_all || clusters[i] == ident_2) {
            group2.push_back(expression[i]);
        }
    }

    MarkerResult res;
    res.gene_idx = 0;
    res.p_val = 1.0;
    res.avg_log2FC = 0.0;
    res.pct_1 = 0.0;
    res.pct_2 = 0.0;
    res.p_val_adj = 1.0;

    if (group1.empty() || group2.empty()) return res;

    // Compute pct expressed
    double nz1 = 0, nz2 = 0;
    for (auto v : group1) if (v > 0) nz1++;
    for (auto v : group2) if (v > 0) nz2++;
    res.pct_1 = nz1 / group1.size();
    res.pct_2 = nz2 / group2.size();

    // Filter by min_pct
    if (res.pct_1 < min_pct && res.pct_2 < min_pct) return res;

    // Compute log2 fold change
    // Use pseudocount to avoid log(0)
    double mean1 = std::accumulate(group1.begin(), group1.end(), 0.0) / group1.size();
    double mean2 = std::accumulate(group2.begin(), group2.end(), 0.0) / group2.size();

    // For log2FC with single-cell data: use exp(x)-1 approach since data is log-normalized
    double fc = (mean1 + 1e-9) / (mean2 + 1e-9);
    if (fc > 0) {
        res.avg_log2FC = std::log2(fc);
    } else {
        res.avg_log2FC = 0.0;
    }

    // Filter by logfc threshold
    if (std::abs(res.avg_log2FC) < logfc_threshold) return res;

    // Statistical test
    switch (test) {
        case DETest::Wilcoxon:
            res.p_val = wilcoxon_pval(group1, group2);
            break;
        case DETest::TTest:
            res.p_val = ttest_pval(group1, group2);
            break;
        case DETest::LogisticRegression:
            res.p_val = logistic_regression_pval(expression, clusters,
                                                   ident_1, ident_2);
            break;
    }

    return res;
}

// ============================================================
// Chunked gene processing
// ============================================================

void process_gene_chunk(
    HDF5File* file, const std::string& data_group,
    int64 gene_start, int64 gene_count,
    int64 n_cells,
    const std::vector<int32>& clusters,
    int32 ident_1, int32 ident_2,
    DETest test, double logfc_threshold, double min_pct,
    std::vector<MarkerResult>& results,
    hid_t thread_file) {

    auto mat = file->open_csc_matrix(data_group, thread_file);

    // Read gene chunk (rows)
    std::vector<double> chunk(gene_count * n_cells, 0.0);
    mat->read_rows(chunk.data(), gene_start, gene_count, 0, n_cells);

    for (int64 g = 0; g < gene_count; ++g) {
        std::vector<double> expression(chunk.begin() + g * n_cells,
                                        chunk.begin() + (g + 1) * n_cells);

        auto res = test_gene(expression, clusters, ident_1, ident_2,
                             test, logfc_threshold, min_pct);
        res.gene_idx = gene_start + g;
        results.push_back(res);
    }
}

} // namespace sclean
