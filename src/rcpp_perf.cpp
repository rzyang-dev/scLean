#include <Rcpp.h>
#include <RcppEigen.h>
#include <string>
#include <vector>
#include <unordered_set>
#include <hdf5.h>

#include "utils/memory_probe.h"
#include "utils/parallel.h"
#include "utils/resource_monitor.h"

using namespace Rcpp;
using namespace sclean;

// ============================================================
// Performance probes
// ============================================================

// [[Rcpp::export]]
double cpp_current_rss() {
    return static_cast<double>(sclean::current_rss_bytes());
}

// [[Rcpp::export]]
double cpp_wall_time_ns() {
    return static_cast<double>(sclean::wall_time_ns());
}

// [[Rcpp::export]]
void suppress_hdf5_errors() {
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
}

// [[Rcpp::export]]
void cpp_set_threads(int n) {
    set_num_threads(n);
}

// [[Rcpp::export]]
int cpp_get_threads() {
    return static_cast<int>(sclean::get_num_threads());
}

// [[Rcpp::export]]
Rcpp::List cpp_resource_snapshot() {
    ResourceMonitor monitor;
    auto snap = monitor.snapshot();
    return Rcpp::List::create(
        Rcpp::Named("total_ram")        = static_cast<double>(snap.total_ram),
        Rcpp::Named("free_ram")         = static_cast<double>(snap.free_ram),
        Rcpp::Named("current_rss")      = static_cast<double>(snap.current_rss),
        Rcpp::Named("available_ram")    = static_cast<double>(snap.available_ram),
        Rcpp::Named("cpu_load_1min")    = snap.cpu_load_1min,
        Rcpp::Named("cpu_load_5min")    = snap.cpu_load_5min,
        Rcpp::Named("physical_cores")   = snap.physical_cores,
        Rcpp::Named("available_cores")  = snap.available_cores,
        Rcpp::Named("memory_pressure")  = snap.memory_pressure,
        Rcpp::Named("cpu_pressure")     = snap.cpu_pressure
    );
}

// [[Rcpp::export]]
std::string cpp_bottleneck_type() {
    ResourceMonitor monitor;
    auto snap = monitor.snapshot();
    Bottleneck b = ResourceMonitor::classify(snap);
    switch (b) {
        case Bottleneck::None:         return "none";
        case Bottleneck::MemoryBound:  return "memory";
        case Bottleneck::ComputeBound: return "compute";
        case Bottleneck::BothBound:    return "both";
    }
    return "none";
}

// [[Rcpp::export]]
void cpp_set_max_dense_chunk_mb(double mb) {
    set_max_dense_chunk_bytes(static_cast<int64>(mb * 1024.0 * 1024.0));
}

// [[Rcpp::export]]
double cpp_get_max_dense_chunk_mb() {
    return static_cast<double>(get_max_dense_chunk_bytes()) / (1024.0 * 1024.0);
}
