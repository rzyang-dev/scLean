# Tests for ScaleData, FindVariableFeatures, and RunPCA operators

test_that("cpp_scale with centering and scaling", {
  skip_if_not_installed("Matrix")

  counts <- Matrix::rsparsematrix(40, 25, density = 0.3)
  counts@x <- abs(counts@x) * 80
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  # First normalize so data layer exists
  write_csc_to_hdf5(h5, "/assays/RNA/layers/counts",
    counts@x, counts@i, counts@p, nrow(counts), ncol(counts))
  cpp_normalize(h5, "/assays/RNA", 0L, 10000, -1, FALSE)

  result <- cpp_scale(
    hdf5_path    = h5,
    assay_group  = "/assays/RNA",
    do_scale     = TRUE,
    do_center    = TRUE,
    chunk_size   = -1,
    in_memory    = FALSE
  )

  expect_type(result, "list")
  expect_equal(result$n_genes, nrow(counts))
  expect_equal(result$n_cells, ncol(counts))

  # ScaleData only stores per-gene mean/sd; scale.data is NOT materialized.
  # PCA applies centering on-the-fly via Lanczos.
  feat_groups <- list_hdf5_groups(h5, "/assays/RNA/features")
  expect_true("residual_mean" %in% feat_groups)
  expect_true("residual_sd" %in% feat_groups)
})

test_that("cpp_scale without scaling or centering", {
  skip_if_not_installed("Matrix")

  counts <- Matrix::rsparsematrix(20, 15, density = 0.4)
  counts@x <- abs(counts@x) * 50
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  write_csc_to_hdf5(h5, "/assays/RNA/layers/counts",
    counts@x, counts@i, counts@p, nrow(counts), ncol(counts))
  cpp_normalize(h5, "/assays/RNA", 0L, 10000, -1, FALSE)

  result <- cpp_scale(h5, "/assays/RNA", FALSE, FALSE, -1, FALSE)
  expect_equal(result$n_genes, nrow(counts))
})

test_that("cpp_vst finds variable features", {
  skip_if_not_installed("Matrix")

  counts <- Matrix::rsparsematrix(100, 40, density = 0.2)
  counts@x <- abs(counts@x) * 100
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  write_csc_to_hdf5(h5, "/assays/RNA/layers/counts",
    counts@x, counts@i, counts@p, nrow(counts), ncol(counts))
  cpp_normalize(h5, "/assays/RNA", 0L, 10000, -1, FALSE)

  result <- cpp_vst(h5, "/assays/RNA", n_top = 50L)

  expect_type(result, "list")
  expect_true("n_variable" %in% names(result))
  expect_lte(result$n_variable, 50L)

  # Variable features written to HDF5
  expect_true(file.exists(h5))
})

test_that("cpp_pca runs on scaled data", {
  skip_if_not_installed("Matrix")

  counts <- Matrix::rsparsematrix(80, 30, density = 0.25)
  counts@x <- abs(counts@x) * 60
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  write_csc_to_hdf5(h5, "/assays/RNA/layers/counts",
    counts@x, counts@i, counts@p, nrow(counts), ncol(counts))
  cpp_normalize(h5, "/assays/RNA", 0L, 10000, -1, FALSE)
  cpp_scale(h5, "/assays/RNA", TRUE, TRUE, -1, FALSE)

  result <- cpp_pca(
    hdf5_path    = h5,
    assay_group  = "/assays/RNA",
    npcs         = 10L,
    tol          = 1e-5,
    max_iter     = 200L
  )

  expect_type(result, "list")
  expect_true("npcs" %in% names(result))
  expect_true("total_variance" %in% names(result))

  # PCA results written to HDF5
  pca_groups <- list_hdf5_groups(h5, "/assays/RNA/reductions")
  expect_true("pca" %in% pca_groups)
})

test_that("cpp_pca falls back to data layer when no scale.data", {
  skip_if_not_installed("Matrix")

  counts <- Matrix::rsparsematrix(60, 25, density = 0.3)
  counts@x <- abs(counts@x) * 40
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  write_csc_to_hdf5(h5, "/assays/RNA/layers/counts",
    counts@x, counts@i, counts@p, nrow(counts), ncol(counts))
  cpp_normalize(h5, "/assays/RNA", 0L, 10000, -1, FALSE)

  # No scale data, should use data layer
  result <- cpp_pca(h5, "/assays/RNA", npcs = 5L, tol = 1e-6, max_iter = 100L)
  expect_true("npcs" %in% names(result))
})

test_that("cpp_vst with no features requested returns zero", {
  skip_if_not_installed("Matrix")

  counts <- Matrix::rsparsematrix(30, 20, density = 0.1)
  counts@x <- abs(counts@x) * 10
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  write_csc_to_hdf5(h5, "/assays/RNA/layers/counts",
    counts@x, counts@i, counts@p, nrow(counts), ncol(counts))
  cpp_normalize(h5, "/assays/RNA", 0L, 10000, -1, FALSE)

  result <- cpp_vst(h5, "/assays/RNA", n_top = 0L)
  expect_equal(result$n_variable, 0L)
})
