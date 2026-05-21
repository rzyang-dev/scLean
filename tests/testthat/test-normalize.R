# Tests for NormalizeData operator

test_that("cpp_normalize LogNormalize produces expected output", {
  skip_if_not_installed("Matrix")

  counts <- Matrix::rsparsematrix(50, 30, density = 0.3)
  counts@x <- abs(counts@x) * 100
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  write_csc_to_hdf5(h5, "/assays/RNA/layers/counts",
    counts@x, counts@i, counts@p, nrow(counts), ncol(counts))

  result <- cpp_normalize(
    hdf5_path    = h5,
    assay_group  = "/assays/RNA",
    method       = 0L,
    scale_factor = 10000,
    chunk_size   = -1,
    in_memory    = FALSE
  )

  expect_type(result, "list")
  expect_true("n_cells" %in% names(result))
  expect_true("n_genes" %in% names(result))
  expect_equal(result$n_cells, ncol(counts))
  expect_equal(result$n_genes, nrow(counts))
})

test_that("cpp_normalize RelativeCounts produces output", {
  skip_if_not_installed("Matrix")

  counts <- Matrix::rsparsematrix(30, 20, density = 0.4)
  counts@x <- abs(counts@x) * 50
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  write_csc_to_hdf5(h5, "/assays/RNA/layers/counts",
    counts@x, counts@i, counts@p, nrow(counts), ncol(counts))

  result <- cpp_normalize(
    hdf5_path    = h5,
    assay_group  = "/assays/RNA",
    method       = 2L,
    scale_factor = 10000,
    chunk_size   = -1,
    in_memory    = FALSE
  )

  expect_equal(result$n_cells, ncol(counts))
  expect_true(file.exists(h5))
})

test_that("cpp_normalize CLR works", {
  skip_if_not_installed("Matrix")

  counts <- Matrix::rsparsematrix(25, 15, density = 0.35)
  counts@x <- abs(counts@x) * 30
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  write_csc_to_hdf5(h5, "/assays/RNA/layers/counts",
    counts@x, counts@i, counts@p, nrow(counts), ncol(counts))

  result <- cpp_normalize(
    hdf5_path    = h5,
    assay_group  = "/assays/RNA",
    method       = 1L,
    scale_factor = 1,
    chunk_size   = -1,
    in_memory    = FALSE
  )

  expect_equal(result$n_cells, ncol(counts))
})

test_that("cpp_normalize with empty input handles gracefully", {
  skip_if_not_installed("Matrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  empty <- Matrix::sparseMatrix(i = integer(0), j = integer(0), x = double(0),
                                dims = c(10, 5))
  empty <- as(empty, "dgCMatrix")

  write_csc_to_hdf5(h5, "/assays/RNA/layers/counts",
    empty@x, empty@i, empty@p, nrow(empty), ncol(empty))

  result <- cpp_normalize(
    hdf5_path    = h5,
    assay_group  = "/assays/RNA",
    method       = 0L,
    scale_factor = 10000,
    chunk_size   = -1,
    in_memory    = FALSE
  )

  expect_equal(result$n_cells, ncol(empty))
})
