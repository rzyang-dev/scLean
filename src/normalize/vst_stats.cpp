#include "normalize_internal.h"
#include <cmath>
#include <algorithm>

namespace sclean {

// ============================================================
// GeneWelford implementation
// ============================================================

void GeneWelford::update(double x) {
    count++;
    double delta = x - mean;
    mean += delta / static_cast<double>(count);
    double delta2 = x - mean;
    m2 += delta * delta2;
}

void GeneWelford::merge(const GeneWelford& other) {
    if (other.count == 0) return;
    int64 new_n = count + other.count;
    double delta = other.mean - mean;
    mean = mean + delta * static_cast<double>(other.count) /
                  static_cast<double>(new_n);
    m2 = m2 + other.m2 +
         delta * delta * static_cast<double>(count) *
         static_cast<double>(other.count) / static_cast<double>(new_n);
    count = new_n;
}

double GeneWelford::variance() const {
    return (count > 1) ? std::max(0.0, m2 / static_cast<double>(count - 1))
                       : 0.0;
}

// ============================================================
// Accumulation functions
// ============================================================

void accumulate_dense_chunk(const double* buf, int64 n_genes, int64 n_cols,
                            std::vector<GeneWelford>& stats) {
    for (int64 i = 0; i < n_genes; ++i) {
        for (int64 j = 0; j < n_cols; ++j) {
            double x = buf[i * n_cols + j];
            stats[i].update(x);
        }
    }
}

void accumulate_sparse_chunk(const double* vals, const int32* idxs,
                             const int64* indptr, int64 col_count,
                             std::vector<GeneWelford>& stats) {
    for (int64 j = 0; j < col_count; ++j) {
        int64 col_start = indptr[j] - indptr[0];
        int64 col_end = indptr[j + 1] - indptr[0];

        for (int64 k = col_start; k < col_end; ++k) {
            int64 gene = idxs[k];
            double x = vals[k];
            stats[gene].update(x);
        }
    }
}

void merge_stats_arrays(const std::vector<GeneWelford>& local,
                        std::vector<GeneWelford>& global, int64 n_genes) {
    for (int64 i = 0; i < n_genes; ++i) {
        if (local[i].count > 0) {
            global[i].merge(local[i]);
        }
    }
}

void finalize_stats_arrays(const std::vector<GeneWelford>& stats,
                           std::vector<double>& means,
                           std::vector<double>& variances) {
    int64 n = static_cast<int64>(stats.size());
    means.resize(n);
    variances.resize(n);
    for (int64 i = 0; i < n; ++i) {
        means[i] = stats[i].mean;
        variances[i] = stats[i].variance();
    }
}

} // namespace sclean
