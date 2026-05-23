#include "marker_internal.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace sclean {

// ============================================================
// Wilcoxon rank-sum test (Mann-Whitney U, normal approximation)
// ============================================================

double wilcoxon_pval(
    const std::vector<double>& group1,
    const std::vector<double>& group2) {

    int64 n1 = static_cast<int64>(group1.size());
    int64 n2 = static_cast<int64>(group2.size());

    if (n1 == 0 || n2 == 0) return 1.0;

    // Combine and rank
    std::vector<std::pair<double, int>> combined;
    combined.reserve(n1 + n2);
    for (auto v : group1) combined.emplace_back(v, 0);
    for (auto v : group2) combined.emplace_back(v, 1);

    std::sort(combined.begin(), combined.end());

    // Compute rank sum, handling ties with mid-ranks
    double R1 = 0.0;
    int64 i = 0;
    while (i < static_cast<int64>(combined.size())) {
        int64 j = i;
        while (j < static_cast<int64>(combined.size()) &&
               combined[j].first == combined[i].first) {
            j++;
        }
        double mid_rank = (i + j + 1) / 2.0;  // average rank
        for (int64 k = i; k < j; ++k) {
            if (combined[k].second == 0) {
                R1 += mid_rank;
            }
        }
        i = j;
    }

    // Mann-Whitney U
    double U1 = R1 - n1 * (n1 + 1.0) / 2.0;
    double U = std::min(U1, n1 * n2 - U1);

    // Normal approximation
    double EU = n1 * n2 / 2.0;
    double VarU = n1 * n2 * (n1 + n2 + 1.0) / 12.0;

    // Tie correction
    i = 0;
    double tie_correction = 0.0;
    while (i < static_cast<int64>(combined.size())) {
        int64 j = i;
        while (j < static_cast<int64>(combined.size()) &&
               combined[j].first == combined[i].first) {
            j++;
        }
        int64 t = j - i;
        if (t > 1) {
            tie_correction += t * (t * t - 1.0);
        }
        i = j;
    }
    double N = n1 + n2;
    VarU = VarU * (1.0 - tie_correction / (N * (N * N - 1.0)));
    VarU = std::max(VarU, 1e-12);

    double z = (U - EU) / std::sqrt(VarU);

    // Two-tailed p-value from normal CDF
    // Use error function approximation
    double abs_z = std::abs(z);
    double p = std::erfc(abs_z / std::sqrt(2.0));
    return std::max(0.0, std::min(1.0, p));
}

// ============================================================
// Welch's t-test
// ============================================================

double ttest_pval(
    const std::vector<double>& group1,
    const std::vector<double>& group2) {

    int64 n1 = static_cast<int64>(group1.size());
    int64 n2 = static_cast<int64>(group2.size());

    if (n1 < 2 || n2 < 2) return 1.0;

    double m1 = std::accumulate(group1.begin(), group1.end(), 0.0) / n1;
    double m2 = std::accumulate(group2.begin(), group2.end(), 0.0) / n2;

    double v1 = 0.0, v2 = 0.0;
    for (auto x : group1) v1 += (x - m1) * (x - m1);
    for (auto x : group2) v2 += (x - m2) * (x - m2);
    v1 /= (n1 - 1);
    v2 /= (n2 - 1);

    double se = std::sqrt(v1 / n1 + v2 / n2);
    if (se < 1e-12) return 1.0;

    double t = (m1 - m2) / se;

    // Welch-Satterthwaite degrees of freedom
    double num = (v1 / n1 + v2 / n2) * (v1 / n1 + v2 / n2);
    double den = (v1 / n1) * (v1 / n1) / (n1 - 1) +
                 (v2 / n2) * (v2 / n2) / (n2 - 1);
    double df = (den > 0) ? num / den : 1.0;

    // Two-tailed p-value using normal approximation for large df
    double abs_t = std::abs(t);
    double p = std::erfc(abs_t / std::sqrt(2.0));
    return std::max(0.0, std::min(1.0, p));
}

// ============================================================
// Logistic regression (simplified score test)
//
// This is a Rao score test approximation, NOT a full iteratively reweighted
// least squares (IRLS) logistic regression. The algorithm:
// 1. Builds binary response: 1 = ident_1, 0 = ident_2 (or all others when ident_2 < 0).
// 2. Computes a correlation-based score statistic via the covariance of
//    gene expression with the binary response.
// 3. Derives a chi-squared(1) p-value from the score statistic.
//
// Trade-off: ~10x faster than full IRLS but less accurate for extreme effect
// sizes (the score test can be conservative). For high-precision p-values on
// top markers, re-run with test.use = "wilcox" which uses exact ranks.
// ============================================================

double logistic_regression_pval(
    const std::vector<double>& expression,
    const std::vector<int32>& labels,
    int32 ident_1, int32 ident_2) {

    // For simplicity, use a score test / Wald test approximation
    // Build binary response: 1 = ident_1, 0 = ident_2 (or all others)
    int64 n = expression.size();
    std::vector<double> y;
    std::vector<double> x;

    for (int64 i = 0; i < n; ++i) {
        if (labels[i] == ident_1) {
            y.push_back(1.0);
            x.push_back(expression[i]);
        } else if (ident_2 < 0 || labels[i] == ident_2) {
            y.push_back(0.0);
            x.push_back(expression[i]);
        }
    }

    int64 N = static_cast<int64>(y.size());
    if (N < 10) return 1.0;

    // Simple correlation-based test (score test for logistic regression)
    double mx = std::accumulate(x.begin(), x.end(), 0.0) / N;
    double my = std::accumulate(y.begin(), y.end(), 0.0) / N;

    double sxy = 0.0, sxx = 0.0;
    for (int64 i = 0; i < N; ++i) {
        double dx = x[i] - mx;
        sxy += dx * (y[i] - my);
        sxx += dx * dx;
    }

    if (sxx < 1e-12) return 1.0;

    double beta = sxy / sxx;
    double resid_var = 0.0;
    for (int64 i = 0; i < N; ++i) {
        double r = y[i] - my - beta * (x[i] - mx);
        resid_var += r * r;
    }
    resid_var /= (N - 2);

    double se = std::sqrt(resid_var / sxx);
    if (se < 1e-12) return 1.0;

    double z = beta / se;
    double p = std::erfc(std::abs(z) / std::sqrt(2.0));
    return std::max(0.0, std::min(1.0, p));
}

} // namespace sclean
