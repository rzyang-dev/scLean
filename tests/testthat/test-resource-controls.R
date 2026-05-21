test_that("CheckResources returns invisible snapshot", {
  expect_silent(snap <- CheckResources())
  expect_true(is.list(snap))
  expect_true("memory_pressure" %in% names(snap))
  expect_true("cpu_pressure" %in% names(snap))
})

test_that("RefreshMemory triggers gc", {
  expect_silent(RefreshMemory())
})

test_that("PerformanceSnapshot has expected structure", {
  ps <- PerformanceSnapshot()
  expect_true(is.list(ps))
  expect_true("current_rss_mb" %in% names(ps))
  expect_true("wall_time_ns" %in% names(ps))
})

test_that("SetChunkSize validates input", {
  skip_on_cran()
  data("pbmc_small", package = "SeuratObject")
  expect_error(SetChunkSize(pbmc_small, 0),
               "chunk_size must be a positive integer")
  expect_error(SetChunkSize(pbmc_small, -5),
               "chunk_size must be a positive integer")
  expect_silent(SetChunkSize(pbmc_small, 100))
  expect_silent(SetChunkSize(pbmc_small, NA))
})

test_that("SetThreads validates input", {
  expect_error(SetThreads(0), "n must be a positive integer")
  expect_error(SetThreads(-1), "n must be a positive integer")
  expect_silent(SetThreads(2))
  expect_silent(SetThreads(NULL))
})
