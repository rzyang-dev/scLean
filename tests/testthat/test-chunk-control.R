# Tests for chunk control and memory management

test_that("SetChunkSize sets and clears chunk size", {
  # Set a chunk size
  SetChunkSize(NULL, 1000L)
  expect_equal(getOption("scLean.chunk_size"), 1000L)

  # Clear it (NA restores auto)
  SetChunkSize(NULL, NA)
  expect_null(getOption("scLean.chunk_size"))
})

test_that("SetMaxRAM sets memory limit", {
  SetMaxRAM(1024)
  expect_equal(getOption("scLean.max_ram"), 1024L * 1024 * 1024)

  # Restore default
  SetMaxRAM(2048)
  expect_equal(getOption("scLean.max_ram"), 2048L * 1024 * 1024)
})

test_that("SetVerbose toggles verbose output", {
  SetVerbose(TRUE)
  expect_true(getOption("scLean.verbose"))

  SetVerbose(FALSE)
  expect_false(getOption("scLean.verbose"))

  # Restore default
  SetVerbose(TRUE)
})

test_that("MemoryReport returns expected structure", {
  report <- MemoryReport()
  expect_type(report, "list")
  expect_true("max_ram_mb" %in% names(report))
  expect_true("chunk_size" %in% names(report))
  expect_equal(report$max_ram_mb, getOption("scLean.max_ram", default = 2048))
})

test_that("MemoryReport with Seurat object includes file info", {
  skip_if_not_installed("Matrix")
  skip_if_not_installed("SeuratObject")

  counts <- Matrix::rsparsematrix(20, 10, density = 0.1)
  counts <- as(counts, "dgCMatrix")

  h5 <- tempfile(fileext = ".h5")
  write_csc_to_hdf5(h5, "/assays/RNA/layers/counts",
    counts@x, counts@i, counts@p, 20L, 10L)
  write_strings_to_hdf5(h5, "/features/names", paste0("g", 1:20))
  write_strings_to_hdf5(h5, "/cells/names", paste0("c", 1:10))

  assay <- CreateSCleanAssayFromHDF5(h5, assay = "RNA")
  obj <- SeuratObject::CreateSeuratObject(counts = counts, assay = "RNA")
  obj@assays[["RNA"]] <- assay
  report <- MemoryReport(obj)

  expect_equal(report$hdf5_path, h5)
  expect_true("hdf5_file_size_mb" %in% names(report))

  unlink(h5)
})

test_that("Internal option helpers return correct defaults", {
  expect_null(scLean:::.get_chunk_override())
  expect_gt(scLean:::.get_max_ram_bytes(), 0)
  expect_true(scLean:::.get_verbose())
})
