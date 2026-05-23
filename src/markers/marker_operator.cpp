#include "marker_operator.h"
#include "marker_internal.h"
#include "hdf5/hdf5_file.h"
#include "utils/progress.h"
#include "utils/resource_monitor.h"
#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sclean {

MarkerOperator::MarkerOperator(DETest test, double logfc_threshold, double min_pct)
    : test_(test), logfc_threshold_(logfc_threshold), min_pct_(min_pct) {}

// ============================================================
// Benjamini-Hochberg correction
// ============================================================

void MarkerOperator::correct_pvalues(std::vector<MarkerResult>& results) {
    int n = static_cast<int>(results.size());
    if (n <= 1) {
        for (auto& r : results) r.p_val_adj = r.p_val;
        return;
    }

    // Sort by p-value
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return results[a].p_val < results[b].p_val;
    });

    // BH correction
    for (int rank = 0; rank < n; ++rank) {
        int idx = order[rank];
        double bh = results[idx].p_val * n / (rank + 1.0);
        results[idx].p_val_adj = std::min(1.0, bh);
    }

    // Ensure monotonicity
    for (int rank = n - 2; rank >= 0; --rank) {
        int idx = order[rank];
        int next_idx = order[rank + 1];
        results[idx].p_val_adj = std::min(results[idx].p_val_adj,
                                           results[next_idx].p_val_adj);
    }
}

// ============================================================
// Public methods
// ============================================================

std::vector<MarkerResult> MarkerOperator::find_markers(
    HDF5File* file, const std::string& data_group,
    int64 n_genes, int64 n_cells,
    const std::vector<int32>& clusters,
    int32 ident_1, int32 ident_2,
    ChunkScheduler& scheduler, int n_threads) {

    std::vector<MarkerResult> results;

    scheduler.refresh_available_ram();
    auto cfg = scheduler.schedule(n_genes, n_cells, OperationType::FindMarkers, n_threads);
    int64 chunk_size = cfg.chunk_size;
    int64 n_chunks = (n_genes + chunk_size - 1) / chunk_size;
    ProgressReporter progress("FindMarkers", n_chunks,
                              ProgressReporter::is_verbose());

    bool oom_occurred = false;
    #pragma omp parallel num_threads(n_threads) shared(oom_occurred)
    {
        hid_t t_fid = (n_threads > 1) ? file->open_thread_handle(FileMode::ReadOnly) : -1;
        std::vector<MarkerResult> local_results;
        #pragma omp for schedule(dynamic)
        for (int64 g = 0; g < n_genes; g += chunk_size) {
            if (oom_occurred) continue;
            int64 gc = std::min(chunk_size, n_genes - g);
            try {
                process_gene_chunk(file, data_group, g, gc, n_cells,
                                    clusters, ident_1, ident_2,
                                    test_, logfc_threshold_, min_pct_,
                                    local_results, t_fid);
            } catch (const std::bad_alloc&) {
                #pragma omp critical
                { oom_occurred = true; }
                continue;
            }
            #pragma omp critical
            progress.step();
        }

        #pragma omp critical
        {
            results.insert(results.end(), local_results.begin(),
                           local_results.end());
        }
    }

    if (oom_occurred) {
        auto snap = ResourceMonitor().snapshot();
        if (!scheduler.shrink_and_retry(n_genes, n_cells,
                OperationType::FindMarkers, n_threads, cfg)) {
            REprintf("[scLean] FindMarkers: FATAL after OOM "
                     "(free: %lld MB, RSS: %lld MB)\n",
                     (long long)(snap.free_ram >> 20),
                     (long long)(snap.current_rss >> 20));
            throw std::bad_alloc();
        }
        REprintf("[scLean] FindMarkers: OOM, shrinking chunk to %lld, retrying\n",
                 (long long)cfg.chunk_size);
        // Retry with shrunk chunk
        return find_markers(file, data_group, n_genes, n_cells,
                            clusters, ident_1, ident_2, scheduler, n_threads);
    }

    // Apply multiple testing correction
    correct_pvalues(results);

    // Sort by p-value
    std::sort(results.begin(), results.end(), [](const MarkerResult& a,
                                                   const MarkerResult& b) {
        return a.p_val_adj < b.p_val_adj;
    });

    progress.done();
    return results;
}

std::vector<std::vector<MarkerResult>> MarkerOperator::find_all_markers(
    HDF5File* file, const std::string& data_group,
    int64 n_genes, int64 n_cells,
    const std::vector<int32>& clusters,
    ChunkScheduler& scheduler, int n_threads) {

    // Get unique clusters
    std::vector<int32> unique_clusters = clusters;
    std::sort(unique_clusters.begin(), unique_clusters.end());
    unique_clusters.erase(std::unique(unique_clusters.begin(),
                                       unique_clusters.end()),
                          unique_clusters.end());

    std::vector<std::vector<MarkerResult>> all_results;

    ProgressReporter progress("FindAllMarkers", unique_clusters.size(),
                              ProgressReporter::is_verbose());
    for (int32 cl : unique_clusters) {
        progress.message("Cluster " + std::to_string(cl) + "...");
        auto markers = find_markers(file, data_group, n_genes, n_cells,
                                     clusters, cl, -1, scheduler, n_threads);
        all_results.push_back(markers);
        progress.step();
    }
    progress.done();

    return all_results;
}

} // namespace sclean
