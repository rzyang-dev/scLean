#include "normalize_internal.h"
#include <cmath>
#include <algorithm>

namespace sclean {

void compute_column_sums(const double* data, const int64* local_indptr,
                         int64 col_count, double* out_sums) {
    for (int64 j = 0; j < col_count; ++j) {
        int64 start = local_indptr[j] - local_indptr[0];
        int64 end = local_indptr[j + 1] - local_indptr[0];
        double sum = 0.0;
        for (int64 k = start; k < end; ++k) {
            sum += data[k];
        }
        out_sums[j] = sum;
    }
}

void compute_column_logsums(const double* data, const int64* local_indptr,
                            int64 col_count, double* out_log_sums,
                            int* out_nz_counts) {
    for (int64 j = 0; j < col_count; ++j) {
        int64 start = local_indptr[j] - local_indptr[0];
        int64 end = local_indptr[j + 1] - local_indptr[0];
        double log_sum = 0.0;
        int nz = 0;
        for (int64 k = start; k < end; ++k) {
            double v = data[k];
            if (v > 0) {
                log_sum += std::log(v);
                nz++;
            }
        }
        out_log_sums[j] = log_sum;
        out_nz_counts[j] = nz;
    }
}

void finalize_geometric_means(const double* log_sums, const int* nz_counts,
                              int64 n_cells, double* out_geo_means) {
    for (int64 j = 0; j < n_cells; ++j) {
        out_geo_means[j] = (nz_counts[j] > 0) ?
            std::exp(log_sums[j] / nz_counts[j]) : 1.0;
    }
}

} // namespace sclean
