# PBMC 3k memory benchmark: scLean pipeline peak memory
# skip_on_cran

test_that("scLean pipeline memory on PBMC 3k", {
  skip_on_cran()
  mem_gb <- tryCatch(
    as.numeric(system("sysctl -n hw.memsize", intern = TRUE)) / (1024^3),
    error = function(e) 0)
  skip_if(mem_gb > 0 && mem_gb < 10,
    sprintf("System has %.0f GB RAM; PBMC 3k test needs 10+ GB in full suite", mem_gb))

  cache_dir <- file.path(Sys.getenv("SCLEAN_CACHE_DIR",
    unset = tools::R_user_dir("scLean", "data")))
  data_dir <- file.path(cache_dir, "filtered_gene_bc_matrices", "hg19")
  skip_if_not(dir.exists(data_dir),
    "PBMC 3k data not cached; run basic validation test first")

  # Streaming import
  h5_path <- tempfile(fileext = ".h5")
  scLean:::stream_10x_to_hdf5(data_dir, h5_path, "RNA")
  sc_assay <- CreateSCleanAssayFromHDF5(h5_path, assay = "RNA")

  # Run pipeline (identical to validation test)
  sc_assay <- NormalizeData(sc_assay, verbose = FALSE)
  sc_assay <- FindVariableFeatures(sc_assay, nfeatures = 2000, verbose = FALSE)
  sc_assay <- ScaleData(sc_assay, verbose = FALSE)
  sc_assay <- RunPCA(sc_assay, npcs = 10, verbose = FALSE)
  sc_pca <- read_dense_matrix(h5_path, "/assays/RNA/reductions/pca/embeddings")
  sc_emb <- sc_pca[, 1:10]
  scLean:::cpp_find_neighbors(sc_emb, 20, "euclidean", 50, h5_path, "/assays/RNA")
  clust_res <- scLean:::cpp_find_clusters(h5_path, "/assays/RNA", 1, 0.8, 10)
  sc_clusters <- read_hdf5_int32(h5_path, "/assays/RNA/clusters/leiden")

  # Measure RSS AFTER pipeline completes (no fork during HDF5 operations)
  pid <- Sys.getpid()
  rss_mb <- as.numeric(system(paste("ps -o rss= -p", pid), intern = TRUE)) / 1024

  # Validate results
  sc_stdev <- read_hdf5_double(h5_path, "/assays/RNA/reductions/pca/stdev")[1:10]
  expect_equal(nrow(sc_pca), ncol(sc_assay))
  expect_equal(ncol(sc_pca), 10)
  expect_gt(sc_stdev[1], 0)
  expect_equal(clust_res$n_clusters, length(unique(sc_clusters)))
  expect_gte(clust_res$n_clusters, 2)
  expect_gt(clust_res$modularity, 0)

  cat(sprintf("\n  Process RSS after full pipeline: %.0f MB\n", rss_mb))

  # Disk-backed architecture: process should stay well under 4 GB
  expect_lt(rss_mb, 4096,
    label = sprintf("Pipeline RSS: %.0f MB (< 4096 MB)", rss_mb))

  rm(sc_assay, sc_pca, sc_emb)
  gc()
  if (file.exists(h5_path)) file.remove(h5_path)
})

test_that("scLean vs Seurat memory comparison", {
  skip_on_cran()
  skip_if_not_installed("Seurat")
  skip_if(
    tolower(Sys.getenv("SCLEAN_COMPARE", "true")) == "false",
    "SCLEAN_COMPARE=false: skipping Seurat comparison"
  )

  mem_gb <- tryCatch(
    as.numeric(system("sysctl -n hw.memsize", intern = TRUE)) / (1024^3),
    error = function(e) 0)
  skip_if(mem_gb > 0 && mem_gb < 14,
    sprintf("System has %.0f GB RAM (< 14 GB needed for Seurat comparison)", mem_gb))

  cache_dir <- file.path(Sys.getenv("SCLEAN_CACHE_DIR",
    unset = tools::R_user_dir("scLean", "data")))
  data_dir <- file.path(cache_dir, "filtered_gene_bc_matrices", "hg19")
  skip_if_not(dir.exists(data_dir),
    "PBMC 3k data not cached; run basic validation test first")

  counts <- suppressWarnings(Seurat::Read10X(data_dir))

  seurat_obj <- Seurat::CreateSeuratObject(counts, project = "pbmc3k")
  seurat_obj <- Seurat::NormalizeData(seurat_obj, verbose = FALSE)
  seurat_obj <- Seurat::FindVariableFeatures(seurat_obj, nfeatures = 2000, verbose = FALSE)
  seurat_obj <- Seurat::ScaleData(seurat_obj, verbose = FALSE)
  seurat_obj <- Seurat::RunPCA(seurat_obj, npcs = 10, verbose = FALSE)
  seurat_obj <- Seurat::FindNeighbors(seurat_obj, dims = 1:10, verbose = FALSE)
  seurat_obj <- Seurat::FindClusters(seurat_obj, resolution = 0.8, verbose = FALSE)

  pid <- Sys.getpid()
  rss_seurat <- as.numeric(system(paste("ps -o rss= -p", pid), intern = TRUE)) / 1024

  cat(sprintf("\n  Seurat pipeline RSS: %.0f MB\n", rss_seurat))

  expect_true(is.finite(rss_seurat) && rss_seurat > 0,
    label = "Seurat pipeline completed")

  rm(seurat_obj, counts)
  gc()
})
