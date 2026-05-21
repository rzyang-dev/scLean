test_that("cpp_resource_snapshot returns valid fields", {
  snap <- cpp_resource_snapshot()
  expect_true(snap$total_ram > 0)
  expect_true(snap$free_ram > 0)
  expect_true(snap$current_rss > 0)
  expect_true(snap$memory_pressure >= 0)
  expect_true(snap$cpu_pressure >= 0)
  expect_true(snap$physical_cores >= 1)
})

test_that("cpp_bottleneck_type returns valid string", {
  bt <- cpp_bottleneck_type()
  expect_true(bt %in% c("none", "memory", "compute", "both"))
})

test_that("cpp_current_rss and cpp_wall_time_ns return positive values", {
  expect_gt(cpp_current_rss(), 0)
  t0 <- cpp_wall_time_ns()
  Sys.sleep(0.01)
  t1 <- cpp_wall_time_ns()
  expect_gt(t1, t0)
})

test_that("cpp_set/get_max_dense_chunk_mb round-trip", {
  orig <- cpp_get_max_dense_chunk_mb()
  cpp_set_max_dense_chunk_mb(64)
  expect_equal(cpp_get_max_dense_chunk_mb(), 64)
  cpp_set_max_dense_chunk_mb(orig)
})
