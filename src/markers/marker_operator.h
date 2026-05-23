#ifndef SCLEAN_MARKER_OPERATOR_H
#define SCLEAN_MARKER_OPERATOR_H

#include <cstdint>
#include <vector>
#include <string>
#include "scLean_types.h"
#include "utils/chunk_scheduler.h"

namespace sclean {

class HDF5File;

enum class DETest { Wilcoxon = 0, TTest = 1, LogisticRegression = 2 };

struct MarkerResult {
    int64 gene_idx;
    double p_val;
    double avg_log2FC;
    double pct_1;
    double pct_2;
    double p_val_adj;
};

class MarkerOperator {
public:
    MarkerOperator(DETest test = DETest::Wilcoxon,
                   double logfc_threshold = 0.25,
                   double min_pct = 0.1);

    // Find markers for one cluster vs all others
    std::vector<MarkerResult> find_markers(
        HDF5File* file,
        const std::string& data_group,
        int64 n_genes, int64 n_cells,
        const std::vector<int32>& clusters,
        int32 ident_1, int32 ident_2,
        ChunkScheduler& scheduler,
        int n_threads = 1);

    // Find markers for all clusters
    std::vector<std::vector<MarkerResult>> find_all_markers(
        HDF5File* file,
        const std::string& data_group,
        int64 n_genes, int64 n_cells,
        const std::vector<int32>& clusters,
        ChunkScheduler& scheduler,
        int n_threads = 1);

private:
    DETest test_;
    double logfc_threshold_;
    double min_pct_;

    // Benjamini-Hochberg correction
    void correct_pvalues(std::vector<MarkerResult>& results);
};

} // namespace sclean

#endif // SCLEAN_MARKER_OPERATOR_H
