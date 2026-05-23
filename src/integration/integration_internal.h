#ifndef SCLEAN_INTEGRATION_INTERNAL_H
#define SCLEAN_INTEGRATION_INTERNAL_H

#include <cstdint>
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include "../scLean_types.h"

namespace sclean {

// ============================================================
// Pure functions for MNN-based batch correction.
// These operate on in-memory Eigen matrices and have no side
// effects (no HDF5 I/O, no console output, no global state).
// ============================================================

// Find mutual nearest neighbors between two batches.
// Uses brute-force O(n1 * n2 * d) search with a max-heap for
// top-k selection. Returns (ref_idx, query_idx) pairs.
std::vector<std::pair<int, int>> find_mnn_pairs(
    const Eigen::MatrixXd& batch1_emb,
    const Eigen::MatrixXd& batch2_emb,
    int k);

// Compute raw correction vectors from MNN pairs.
// For each query cell, averages the difference vectors
// (ref_cell - query_cell) across all its MNN pairs.
// Returns an (n_query, d) matrix where each row is the
// average correction for that query cell.
Eigen::MatrixXd compute_raw_correction(
    const Eigen::MatrixXd& ref_emb,
    const Eigen::MatrixXd& query_emb,
    const std::vector<std::pair<int, int>>& mnn_pairs);

// Gaussian kernel smoothing (full-matrix path).
// Smooths raw correction vectors using a Gaussian kernel
// over the query embedding space. Bandwidth is data-driven:
// sigma * ||first_row|| .
Eigen::MatrixXd smooth_correction_gaussian(
    const Eigen::MatrixXd& query_emb,
    const Eigen::MatrixXd& raw_correction,
    double sigma);

// Chunked Gaussian kernel smoothing for large datasets.
// Processes target cells in blocks to control peak memory.
// Pre-extracts source embeddings and runs the kernel in a
// double loop, optionally parallelised via OpenMP.
Eigen::MatrixXd smooth_correction_gaussian_chunked(
    const Eigen::MatrixXd& query_emb,
    const Eigen::MatrixXd& raw_correction,
    double sigma,
    int64 chunk_size,
    int n_threads);

} // namespace sclean

#endif // SCLEAN_INTEGRATION_INTERNAL_H
