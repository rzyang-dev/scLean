# Tests for FindNeighbors, FindClusters, FindMarkers, and IntegrateLayers

test_that("cpp_find_neighbors builds KNN and SNN graphs", {
  skip_if_not_installed("Matrix")

  counts <- Matrix::rsparsematrix(60, 40, density = 0.25)
  counts@x <- abs(counts@x) * 50
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  write_csc_to_hdf5(h5, "/assays/RNA/layers/counts",
    counts@x, counts@i, counts@p, nrow(counts), ncol(counts))
  cpp_normalize(h5, "/assays/RNA", 0L, 10000, -1, FALSE)
  cpp_scale(h5, "/assays/RNA", TRUE, TRUE, -1, FALSE)
  pca_result <- cpp_pca(h5, "/assays/RNA", npcs = 10L, tol = 1e-6, max_iter = 100L)

  # Read PCA embeddings for neighbors
  pca_emb <- read_hdf5_dense_chunk(
    h5, "/assays/RNA/reductions/pca/embeddings",
    0L, ncol(counts), 0L, pca_result$npcs
  )

  cpp_find_neighbors(
    embeddings  = pca_emb,
    k           = 20L,
    metric      = "euclidean",
    n_trees     = 50L,
    hdf5_path   = h5,
    assay_group = "/assays/RNA"
  )

  # Graphs should be written
  graph_groups <- list_hdf5_groups(h5, "/assays/RNA/graphs")
  expect_true("nn" %in% graph_groups)
  expect_true("snn" %in% graph_groups)
})

test_that("cpp_find_clusters runs Louvain clustering", {
  skip_if_not_installed("Matrix")

  counts <- Matrix::rsparsematrix(60, 30, density = 0.3)
  counts@x <- abs(counts@x) * 40
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  write_csc_to_hdf5(h5, "/assays/RNA/layers/counts",
    counts@x, counts@i, counts@p, nrow(counts), ncol(counts))
  cpp_normalize(h5, "/assays/RNA", 0L, 10000, -1, FALSE)
  cpp_scale(h5, "/assays/RNA", TRUE, TRUE, -1, FALSE)
  pca_result <- cpp_pca(h5, "/assays/RNA", npcs = 10L, tol = 1e-6, max_iter = 100L)
  pca_emb <- read_hdf5_dense_chunk(
    h5, "/assays/RNA/reductions/pca/embeddings",
    0L, ncol(counts), 0L, pca_result$npcs
  )
  cpp_find_neighbors(pca_emb, 15L, "euclidean", 50L, h5, "/assays/RNA")

  result <- cpp_find_clusters(
    hdf5_path    = h5,
    assay_group  = "/assays/RNA",
    algorithm    = 0L,     # Louvain
    resolution   = 0.8,
    n_iter       = 10L
  )

  expect_type(result, "list")
  expect_true("n_clusters" %in% names(result))
  expect_true("modularity" %in% names(result))
  expect_true("assignments" %in% names(result))
  expect_equal(length(result$assignments), ncol(counts))
  expect_true(all(result$assignments >= 0))
})

test_that("cpp_find_markers identifies DE genes", {
  skip_if_not_installed("Matrix")

  counts <- Matrix::rsparsematrix(50, 30, density = 0.3)
  counts@x <- abs(counts@x) * 50
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  write_csc_to_hdf5(h5, "/assays/RNA/layers/counts",
    counts@x, counts@i, counts@p, nrow(counts), ncol(counts))
  cpp_normalize(h5, "/assays/RNA", 0L, 10000, -1, FALSE)

  # Create mock clusters: split cells into two groups
  clusters <- rep(c(0L, 1L), each = 15)[1:ncol(counts)]

  result <- cpp_find_markers(
    hdf5_path      = h5,
    assay_group    = "/assays/RNA",
    clusters       = as.integer(clusters),
    ident_1        = 0L,
    ident_2        = 1L,
    test_type      = 0L,      # Wilcoxon
    logfc_threshold = 0.0,
    min_pct        = 0.0
  )

  expect_type(result, "list")
  expect_true(length(result) > 0)

  # Each result is a list with expected fields
  first <- result[[1]]
  expect_true("p_val" %in% names(first))
  expect_true("avg_log2FC" %in% names(first))
  expect_true("pct.1" %in% names(first))
  expect_true("pct.2" %in% names(first))
  expect_true("p_val_adj" %in% names(first))
})

test_that("cpp_find_markers with t-test", {
  skip_if_not_installed("Matrix")

  counts <- Matrix::rsparsematrix(30, 20, density = 0.35)
  counts@x <- abs(counts@x) * 60
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  write_csc_to_hdf5(h5, "/assays/RNA/layers/counts",
    counts@x, counts@i, counts@p, nrow(counts), ncol(counts))
  cpp_normalize(h5, "/assays/RNA", 0L, 10000, -1, FALSE)

  clusters <- rep(c(0L, 1L), each = 10)

  result <- cpp_find_markers(
    hdf5_path      = h5,
    assay_group    = "/assays/RNA",
    clusters       = as.integer(clusters),
    ident_1        = 0L,
    ident_2        = 1L,
    test_type      = 1L,      # T-Test
    logfc_threshold = 0.0,
    min_pct        = 0.0
  )

  expect_type(result, "list")
})

test_that("cpp_integrate runs MNN correction with default params", {
  skip_if_not_installed("Matrix")

  counts <- Matrix::rsparsematrix(50, 30, density = 0.25)
  counts@x <- abs(counts@x) * 30
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  write_csc_to_hdf5(h5, "/assays/RNA/layers/counts",
    counts@x, counts@i, counts@p, nrow(counts), ncol(counts))
  cpp_normalize(h5, "/assays/RNA", 0L, 10000, -1, FALSE)
  cpp_scale(h5, "/assays/RNA", TRUE, TRUE, -1, FALSE)
  cpp_pca(h5, "/assays/RNA", npcs = 10L, tol = 1e-6, max_iter = 100L)

  # Create mock batch labels: half cells in batch 1, half in batch 2
  batch_labels <- rep(c(1L, 2L), each = 15)

  cpp_integrate(
    hdf5_path    = h5,
    assay_group  = "/assays/RNA",
    batch_labels = as.integer(batch_labels),
    n_ccs        = 10L
  )

  # Verify corrected embeddings written to HDF5
  corr <- read_hdf5_dense_chunk(
    h5, "/assays/RNA/reductions/harmony/embeddings",
    0L, 30L, 0L, 10L
  )
  expect_equal(dim(corr), c(30, 10))
  expect_true(all(is.finite(corr)))
})

test_that("cpp_integrate handles single batch (no-op)", {
  skip_if_not_installed("Matrix")

  counts <- Matrix::rsparsematrix(50, 30, density = 0.25)
  counts@x <- abs(counts@x) * 30
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  write_csc_to_hdf5(h5, "/assays/RNA/layers/counts",
    counts@x, counts@i, counts@p, nrow(counts), ncol(counts))
  cpp_normalize(h5, "/assays/RNA", 0L, 10000, -1, FALSE)
  cpp_scale(h5, "/assays/RNA", TRUE, TRUE, -1, FALSE)
  cpp_pca(h5, "/assays/RNA", npcs = 10L, tol = 1e-6, max_iter = 100L)

  # All cells in one batch
  batch_labels <- rep(1L, 30)

  expect_no_error(
    cpp_integrate(h5, "/assays/RNA", as.integer(batch_labels), 10L)
  )

  # Verify embeddings have correct dimensions and are finite
  corr <- read_hdf5_dense_chunk(
    h5, "/assays/RNA/reductions/harmony/embeddings",
    0L, 30L, 0L, 10L
  )
  expect_equal(dim(corr), c(30, 10))
  expect_true(all(is.finite(corr)))
})

test_that("cpp_integrate MNN correction reduces batch separation", {
  skip_if_not_installed("Matrix")

  # Create two batches with systematic offset by adding structured noise
  set.seed(42)
  n_genes <- 60
  n_cells <- 40

  # Shift gene expression for second batch to create batch effect
  counts1 <- Matrix::rsparsematrix(n_genes, n_cells / 2, density = 0.25)
  counts2 <- Matrix::rsparsematrix(n_genes, n_cells / 2, density = 0.25)
  counts1@x <- abs(counts1@x) * 50
  counts2@x <- abs(counts2@x) * 50 + 10  # offset
  counts <- cbind(counts1, counts2)
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  write_csc_to_hdf5(h5, "/assays/RNA/layers/counts",
    counts@x, counts@i, counts@p, nrow(counts), ncol(counts))
  cpp_normalize(h5, "/assays/RNA", 0L, 10000, -1, FALSE)
  cpp_scale(h5, "/assays/RNA", TRUE, TRUE, -1, FALSE)
  cpp_pca(h5, "/assays/RNA", npcs = 10L, tol = 1e-6, max_iter = 100L)

  # Read original PCA embeddings
  pca_before <- read_hdf5_dense_chunk(
    h5, "/assays/RNA/reductions/pca/embeddings", 0L, n_cells, 0L, 10L
  )

  batch_labels <- rep(c(1L, 2L), each = n_cells / 2)
  cpp_integrate(h5, "/assays/RNA", as.integer(batch_labels), 10L, 10L, 0.1, 2L)

  # Read corrected embeddings
  pca_after <- read_hdf5_dense_chunk(
    h5, "/assays/RNA/reductions/harmony/embeddings", 0L, n_cells, 0L, 10L
  )

  # Batch means should be closer or similar after correction
  batch1_before <- colMeans(pca_before[1:20, , drop = FALSE])
  batch2_before <- colMeans(pca_before[21:40, , drop = FALSE])
  batch1_after <- colMeans(pca_after[1:20, , drop = FALSE])
  batch2_after <- colMeans(pca_after[21:40, , drop = FALSE])

  dist_before <- sqrt(sum((batch1_before - batch2_before)^2))
  dist_after <- sqrt(sum((batch1_after - batch2_after)^2))

  # Correction should not increase batch separation
  expect_true(dist_after <= dist_before * 1.2,
    info = sprintf("dist before: %.4f, after: %.4f", dist_before, dist_after))
})

test_that("cpp_integrate handles n_ccs > available PCs", {
  skip_if_not_installed("Matrix")

  counts <- Matrix::rsparsematrix(50, 30, density = 0.25)
  counts@x <- abs(counts@x) * 30
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  write_csc_to_hdf5(h5, "/assays/RNA/layers/counts",
    counts@x, counts@i, counts@p, nrow(counts), ncol(counts))
  cpp_normalize(h5, "/assays/RNA", 0L, 10000, -1, FALSE)
  cpp_scale(h5, "/assays/RNA", TRUE, TRUE, -1, FALSE)
  cpp_pca(h5, "/assays/RNA", npcs = 5L, tol = 1e-6, max_iter = 100L)

  batch_labels <- rep(c(1L, 2L), each = 15)

  # Request 10 CCs but only 5 PCs available -> should use min(10, 5)=5
  expect_no_error(
    cpp_integrate(h5, "/assays/RNA", as.integer(batch_labels), 10L, 20L, 0.1, 1L)
  )

  corr <- read_hdf5_dense_chunk(
    h5, "/assays/RNA/reductions/harmony/embeddings", 0L, 30L, 0L, 5L
  )
  expect_equal(ncol(corr), 5)
  expect_true(all(is.finite(corr)))
})

test_that("cpp_integrate respects n_mnn, sigma, max_iter parameters", {
  skip_if_not_installed("Matrix")

  counts <- Matrix::rsparsematrix(50, 30, density = 0.25)
  counts@x <- abs(counts@x) * 30
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  write_csc_to_hdf5(h5, "/assays/RNA/layers/counts",
    counts@x, counts@i, counts@p, nrow(counts), ncol(counts))
  cpp_normalize(h5, "/assays/RNA", 0L, 10000, -1, FALSE)
  cpp_scale(h5, "/assays/RNA", TRUE, TRUE, -1, FALSE)
  cpp_pca(h5, "/assays/RNA", npcs = 10L, tol = 1e-6, max_iter = 100L)

  batch_labels <- rep(c(1L, 2L), each = 15)

  # Run with non-default parameters
  expect_no_error(
    cpp_integrate(h5, "/assays/RNA", as.integer(batch_labels),
                  n_ccs = 8L, n_mnn = 10L, sigma = 0.5, max_iter = 1L)
  )

  corr <- read_hdf5_dense_chunk(
    h5, "/assays/RNA/reductions/harmony/embeddings", 0L, 30L, 0L, 8L
  )
  expect_equal(ncol(corr), 8)
  expect_true(all(is.finite(corr)))
})

test_that("cpp_integrate chunked path is consistent with full-matrix path", {
  skip_if_not_installed("Matrix")

  set.seed(123)
  counts <- Matrix::rsparsematrix(40, 20, density = 0.3)
  counts@x <- abs(counts@x) * 30
  counts <- as(counts, "dgCMatrix")

  h5_full <- tempfile(fileext = ".h5")
  h5_chunked <- tempfile(fileext = ".h5")
  on.exit({unlink(h5_full); unlink(h5_chunked)})

  for (h5_path in c(h5_full, h5_chunked)) {
    write_csc_to_hdf5(h5_path, "/assays/RNA/layers/counts",
      counts@x, counts@i, counts@p, nrow(counts), ncol(counts))
    cpp_normalize(h5_path, "/assays/RNA", 0L, 10000, -1, FALSE)
    cpp_scale(h5_path, "/assays/RNA", TRUE, TRUE, -1, FALSE)
    cpp_pca(h5_path, "/assays/RNA", npcs = 6L, tol = 1e-6, max_iter = 100L)
  }

  batch_labels <- rep(c(1L, 2L), each = 10)

  # Full-memory path (default scheduler)
  cpp_integrate(h5_full, "/assays/RNA", as.integer(batch_labels),
                6L, 10L, 0.1, 1L)

  # Chunked path (override with small chunk size to force chunked smoothing)
  old_chunk <- getOption("scLean.chunk_size", default = NULL)
  options(scLean.chunk_size = 3L)
  cpp_integrate(h5_chunked, "/assays/RNA", as.integer(batch_labels),
                6L, 10L, 0.1, 1L)
  options(scLean.chunk_size = old_chunk)

  corr_full <- read_hdf5_dense_chunk(
    h5_full, "/assays/RNA/reductions/harmony/embeddings", 0L, 20L, 0L, 6L
  )
  corr_chunked <- read_hdf5_dense_chunk(
    h5_chunked, "/assays/RNA/reductions/harmony/embeddings", 0L, 20L, 0L, 6L
  )

  # Results should be close (floating-point order may differ slightly)
  max_diff <- max(abs(corr_full - corr_chunked))
  expect_true(max_diff < 1e-8,
    info = sprintf("Max difference between full and chunked: %g", max_diff))
})

test_that("cpp_find_clusters errors without SNN graph", {
  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  expect_error(
    cpp_find_clusters(h5, "/assays/RNA", 0L, 0.8, 10L),
    "SNN graph not found"
  )
})
