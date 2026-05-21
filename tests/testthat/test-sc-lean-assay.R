# Tests for scLeanAssay class and HDF5BackedMatrix

test_that("CreateSCleanAssay builds valid object", {
  skip_if_not_installed("Matrix")

  counts <- Matrix::rsparsematrix(50, 20, density = 0.2)
  rownames(counts) <- paste0("gene", 1:50)
  colnames(counts) <- paste0("cell", 1:20)
  counts <- as(counts, "dgCMatrix")

  assay <- CreateSCleanAssay(counts, assay = "RNA")

  expect_s4_class(assay, "scLeanAssay")
  expect_true(inherits(assay, "Assay5"))
  expect_true(file.exists(assay@hdf5_path))
  expect_equal(assay@hdf5_group, "/assays/RNA")

  expect_true("counts" %in% names(assay@layers))
  proxy <- assay@layers[["counts"]]
  expect_true(inherits(proxy, "HDF5BackedMatrix"))

  expect_equal(nrow(proxy), 50)
  expect_equal(ncol(proxy), 20)

  unlink(assay@hdf5_path)
})

test_that("HDF5BackedMatrix dim and accessors", {
  skip_if_not_installed("Matrix")

  counts <- Matrix::rsparsematrix(30, 10, density = 0.15)
  rownames(counts) <- paste0("g", 1:30)
  colnames(counts) <- paste0("c", 1:10)
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  write_csc_to_hdf5(h5, "/mat", counts@x, counts@i, counts@p, 30L, 10L)
  write_strings_to_hdf5(h5, "/features/names", rownames(counts))
  write_strings_to_hdf5(h5, "/cells/names", colnames(counts))

  hbm <- HDF5BackedMatrix$new(h5, "/mat")

  expect_equal(dim(hbm), c(30, 10))
  expect_equal(nrow(hbm), 30)
  expect_equal(ncol(hbm), 10)
  expect_equal(dimnames(hbm), list(rownames(counts), colnames(counts)))

  sub <- hbm$get(1:5, 1:3)
  expect_equal(dim(sub), c(5, 3))
  expected <- as.matrix(counts[1:5, 1:3])
  expect_equal(unname(sub), unname(expected), tolerance = 1e-10)

  full <- hbm$as_matrix()
  expect_s4_class(full, "dgCMatrix")
  expect_equal(dim(full), dim(counts))
})

test_that("HDF5BackedMatrix S3 methods work", {
  skip_if_not_installed("Matrix")

  counts <- Matrix::rsparsematrix(20, 8, density = 0.2)
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(h5))

  write_csc_to_hdf5(h5, "/s3mat", counts@x, counts@i, counts@p, 20L, 8L)

  hbm <- HDF5BackedMatrix$new(h5, "/s3mat")

  expect_equal(dim(hbm), c(20, 8))
  expect_equal(as.vector(hbm[1:3, 1:2]), as.vector(as.matrix(counts[1:3, 1:2])))

  m <- as.matrix(hbm)
  expect_s4_class(m, "dgCMatrix")
})

test_that("CreateSCleanAssay with random counts", {
  skip_if_not_installed("Matrix")

  set.seed(123)
  counts <- Matrix::rsparsematrix(100, 50, density = 0.15)
  counts@x <- pmax(counts@x, 0)
  rownames(counts) <- paste0("gene", 1:100)
  colnames(counts) <- paste0("cell", 1:50)
  counts <- as(counts, "dgCMatrix")

  assay <- CreateSCleanAssay(counts, assay = "RNA")

  proxy <- assay@layers[["counts"]]
  expect_equal(nrow(proxy), 100)
  expect_equal(ncol(proxy), 50)

  # Read back and verify
  m <- proxy$as_matrix()
  expect_equal(dim(m), c(100, 50))

  unlink(assay@hdf5_path)
})
