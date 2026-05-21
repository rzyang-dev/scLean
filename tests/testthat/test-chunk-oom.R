# Tests for OOM recovery and edge-case robustness

test_that("VST select_top_features handles zero nfeatures", {
  skip_if_not_installed("Matrix")

  counts <- Matrix::rsparsematrix(100, 50, density = 0.2)
  counts@x <- abs(counts@x) * 10
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  write_csc_to_hdf5(h5, "/assays/RNA/layers/counts",
    counts@x, counts@i, counts@p, nrow(counts), ncol(counts))
  cpp_normalize(h5, "/assays/RNA", 0L, 10000, -1, FALSE)

  # n_top = 0 should return no variable features, not crash
  result <- cpp_vst(h5, "/assays/RNA", n_top = 0L)
  expect_equal(result$n_variable, 0)
})

test_that("VST select_top_features handles negative nfeatures", {
  skip_if_not_installed("Matrix")

  counts <- Matrix::rsparsematrix(60, 30, density = 0.2)
  counts@x <- abs(counts@x) * 10
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  write_csc_to_hdf5(h5, "/assays/RNA/layers/counts",
    counts@x, counts@i, counts@p, nrow(counts), ncol(counts))
  cpp_normalize(h5, "/assays/RNA", 0L, 10000, -1, FALSE)

  # Negative n_top should not crash (UB guard)
  result <- cpp_vst(h5, "/assays/RNA", n_top = -1L)
  expect_equal(result$n_variable, 0)
})

test_that("ScaleData handles small matrices without NaN in SDs", {
  skip_if_not_installed("Matrix")

  counts <- Matrix::rsparsematrix(40, 25, density = 0.3)
  counts@x <- abs(counts@x) * 50
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  write_csc_to_hdf5(h5, "/assays/RNA/layers/counts",
    counts@x, counts@i, counts@p, nrow(counts), ncol(counts))
  cpp_normalize(h5, "/assays/RNA", 0L, 10000, -1, FALSE)

  result <- cpp_scale(h5, "/assays/RNA", TRUE, TRUE, -1L, FALSE)

  # SD should be finite or 0, never NaN
  expect_true(all(is.finite(result$gene_sds) | result$gene_sds == 0))
  expect_true(all(result$gene_sds >= 0))
})
