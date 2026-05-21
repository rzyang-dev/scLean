#ifndef SCLEAN_PROGRESS_H
#define SCLEAN_PROGRESS_H

#include <string>
#include <functional>
#include <cstdint>
#include <Rcpp.h>
#include "../sclean_types.h"

namespace sclean {

// Simple progress reporter — calls R's message() or Rprintf
class ProgressReporter {
public:
    ProgressReporter(const std::string& label, int64 total_steps,
                     bool verbose = true)
        : label_(label), total_(total_steps), current_(0),
          verbose_(verbose), last_pct_(-1) {}

    void step(int64 n = 1) {
        current_ += n;
        if (!verbose_) return;

        int pct = static_cast<int>(100.0 * current_ / total_);
        if (pct > last_pct_ && pct % 10 == 0) {
            last_pct_ = pct;
            Rprintf("[%s] %d%% (%lld/%lld)\n", label_.c_str(),
                    pct, (long long)current_, (long long)total_);
            R_FlushConsole();
        }
    }

    void done() {
        if (verbose_) {
            Rprintf("[%s] 100%% — done\n", label_.c_str());
            R_FlushConsole();
        }
    }

    void message(const std::string& msg) {
        if (verbose_) {
            Rprintf("[%s] %s\n", label_.c_str(), msg.c_str());
            R_FlushConsole();
        }
    }

    void increment(int64 n) { step(n); }
    void set_total(int64 n) { total_ = n; }

    static bool is_verbose() {
        try {
            Rcpp::Environment base = Rcpp::Environment::base_namespace();
            Rcpp::Function getOpt = base["getOption"];
            Rcpp::RObject result = getOpt("scLean.verbose");
            if (result.isNULL() || result.sexp_type() == NILSXP) return true;
            if (!Rf_isLogical(result)) return true;
            Rcpp::LogicalVector v(result);
            if (v.size() > 0) return v[0];
        } catch (...) {}
        return true;  // verbose by default
    }

private:
    std::string label_;
    int64 total_;
    int64 current_;
    bool verbose_;
    int last_pct_;
};

} // namespace sclean

#endif // SCLEAN_PROGRESS_H
