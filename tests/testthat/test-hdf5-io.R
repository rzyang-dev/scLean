# Tests for HDF5 I/O operations

test_that("CSC matrix write and read round-trip", {
  skip_if_not_installed("Matrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  m <- Matrix::rsparsematrix(100, 50, density = 0.1)
  m <- as(m, "dgCMatrix")

  write_csc_to_hdf5(
    hdf5_path  = h5,
    group_path = "/test/matrix",
    data       = m@x,
    indices    = m@i,
    indptr     = m@p,
    n_rows     = nrow(m),
    n_cols     = ncol(m)
  )

  shape <- read_hdf5_shape(h5, "/test/matrix")
  expect_equal(shape, c(nrow(m), ncol(m)))

  m2 <- read_hdf5_as_dgCMatrix(h5, "/test/matrix")
  expect_equal(m2$Dim, dim(m))
  expect_equal(m2$x, m@x, tolerance = 1e-10)
  expect_equal(m2$i, m@i)
  expect_equal(m2$p, m@p)

  chunk <- read_hdf5_dense_chunk(h5, "/test/matrix", 0L, 10L, 0L, 5L)
  expect_equal(dim(chunk), c(10, 5))
  expected_chunk <- as.matrix(m[1:10, 1:5])
  expect_equal(chunk, expected_chunk, tolerance = 1e-10)
})

test_that("Dense chunk read handles edge cases", {
  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  m <- Matrix::rsparsematrix(50, 30, density = 0.05)
  m <- as(m, "dgCMatrix")

  write_csc_to_hdf5(h5, "/test/edge", m@x, m@i, m@p, nrow(m), ncol(m))

  s <- read_hdf5_dense_chunk(h5, "/test/edge", 0L, 1L, 0L, 1L)
  expect_equal(dim(s), c(1, 1))

  last <- read_hdf5_dense_chunk(h5, "/test/edge", nrow(m) - 1L, 1L,
                                ncol(m) - 1L, 1L)
  expect_equal(dim(last), c(1, 1))
})

test_that("String vector write and read round-trip", {
  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  str_vec <- paste0("gene_", seq_len(200))
  write_strings_to_hdf5(h5, "/features/names", str_vec)

  str_back <- read_strings_from_hdf5(h5, "/features/names")
  expect_equal(str_back, str_vec)
})

test_that("List HDF5 groups", {
  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  m <- Matrix::rsparsematrix(10, 10, density = 0.5)
  m <- as(m, "dgCMatrix")

  write_csc_to_hdf5(h5, "/layers/counts", m@x, m@i, m@p, 10L, 10L)
  write_csc_to_hdf5(h5, "/layers/data", m@x, m@i, m@p, 10L, 10L)

  groups <- list_hdf5_groups(h5, "/layers")
  expect_true("counts" %in% groups)
  expect_true("data" %in% groups)
})

test_that("Matvec multiplication", {
  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  m <- Matrix::rsparsematrix(20, 15, density = 0.3)
  m <- as(m, "dgCMatrix")
  write_csc_to_hdf5(h5, "/test/matvec", m@x, m@i, m@p, nrow(m), ncol(m))

  x <- runif(15)
  y_sclean <- hdf5_matvec(h5, "/test/matvec", x)
  y_expected <- as.vector(m %*% x)

  expect_equal(y_sclean, y_expected, tolerance = 1e-10)
})
