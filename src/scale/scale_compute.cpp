#include "scale_internal.h"
#include <cmath>
#include <algorithm>

namespace sclean {

void compute_row_mean_sd(const double* row_data, int64 n_cols,
                         double& mean, double& sd) {
    double m = 0.0, m2 = 0.0;
    int64 count = 0;
    for (int64 j = 0; j < n_cols; ++j) {
        double x = row_data[j];
        count++;
        double delta = x - m;
        m += delta / static_cast<double>(count);
        double delta2 = x - m;
        m2 += delta * delta2;
    }
    mean = m;
    double var = (count > 1) ? std::max(0.0, m2 / static_cast<double>(count - 1))
                             : 0.0;
    sd = std::sqrt(var);
}

int64 scale_sparse_column(const int32* idxs, const double* vals,
                          int64 col_start, int64 col_end,
                          const double* means, const double* sds,
                          bool do_center, bool do_scale,
                          std::vector<double>& out_vals,
                          std::vector<int32>& out_idx) {
    out_vals.clear();
    out_idx.clear();

    for (int64 k = col_start; k < col_end; ++k) {
        int64 gene = idxs[k];
        double v = vals[k];
        double scaled = v;
        if (do_center) scaled -= means[gene];
        if (do_scale && sds[gene] > 0.0) scaled /= sds[gene];
        if (scaled != 0.0) {
            out_vals.push_back(scaled);
            out_idx.push_back(static_cast<int32>(gene));
        }
    }

    return static_cast<int64>(out_vals.size());
}

} // namespace sclean
