#include "normalize_internal.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace sclean {

void fit_loess_binned(const std::vector<double>& log_means,
                      const std::vector<double>& log_variances,
                      std::vector<double>& fitted,
                      int n_bins, double /*span*/) {

    int64 n = static_cast<int64>(log_means.size());
    fitted.resize(n);

    // Filter valid entries
    std::vector<int64> valid_idx;
    for (int64 i = 0; i < n; ++i) {
        if (std::isfinite(log_means[i]) && std::isfinite(log_variances[i])) {
            valid_idx.push_back(i);
        }
    }

    if (valid_idx.empty()) {
        std::fill(fitted.begin(), fitted.end(), 0.0);
        return;
    }

    // Sort valid entries by log_mean
    std::sort(valid_idx.begin(), valid_idx.end(),
              [&](int64 a, int64 b) { return log_means[a] < log_means[b]; });

    // Bin and compute median variance per bin
    int64 n_valid = static_cast<int64>(valid_idx.size());
    int64 per_bin = std::max(static_cast<int64>(1), n_valid / n_bins);

    std::vector<double> bin_centers, bin_variances;

    for (int64 b = 0; b < n_bins; ++b) {
        int64 start = b * per_bin;
        int64 end = std::min(start + per_bin, n_valid);
        if (start >= n_valid) break;

        double sum_mean = 0.0;
        std::vector<double> bin_vars;
        for (int64 k = start; k < end; ++k) {
            sum_mean += log_means[valid_idx[k]];
            bin_vars.push_back(log_variances[valid_idx[k]]);
        }

        // Median variance per bin
        std::sort(bin_vars.begin(), bin_vars.end());
        double med_var = bin_vars[bin_vars.size() / 2];

        bin_centers.push_back(sum_mean / static_cast<double>(end - start));
        bin_variances.push_back(med_var);
    }

    // Linear interpolation across bins
    for (int64 i = 0; i < n; ++i) {
        if (!std::isfinite(log_means[i])) {
            fitted[i] = 0.0;
            continue;
        }

        double lm = log_means[i];

        // Find surrounding bins
        if (lm <= bin_centers.front()) {
            fitted[i] = bin_variances.front();
        } else if (lm >= bin_centers.back()) {
            fitted[i] = bin_variances.back();
        } else {
            // Linear interpolation between bins
            for (size_t b = 0; b < bin_centers.size() - 1; ++b) {
                if (lm >= bin_centers[b] && lm <= bin_centers[b + 1]) {
                    double t = (lm - bin_centers[b]) /
                               (bin_centers[b + 1] - bin_centers[b]);
                    fitted[i] = bin_variances[b] * (1.0 - t) +
                                bin_variances[b + 1] * t;
                    break;
                }
            }
        }
    }
}

void select_top_features(const std::vector<double>& vst_variances,
                         std::vector<int8_t>& variable,
                         int n_select) {

    int64 n = static_cast<int64>(vst_variances.size());
    variable.assign(n, 0);

    if (n_select <= 0) return;

    if (n_select >= n) {
        std::fill(variable.begin(), variable.end(), 1);
        return;
    }

    // Partial sort to find top N
    std::vector<std::pair<double, int64>> ranked;
    ranked.reserve(n);
    for (int64 i = 0; i < n; ++i) {
        if (std::isfinite(vst_variances[i])) {
            ranked.emplace_back(vst_variances[i], i);
        }
    }

    if (ranked.empty()) return;

    int64 actual_select = std::min(static_cast<int64>(n_select),
                                    static_cast<int64>(ranked.size()));

    std::nth_element(ranked.begin(), ranked.begin() + actual_select,
                     ranked.end(), std::greater<>{});

    for (int64 k = 0; k < actual_select; ++k) {
        variable[ranked[k].second] = 1;
    }
}

} // namespace sclean
