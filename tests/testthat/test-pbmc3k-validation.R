# PBMC 3k end-to-end validation: scLean vs Seurat
# skip_on_cran due to data download requirements
# Note: 8GB systems may need to set SCLEAN_COMPARE=FALSE to skip
# the memory-intensive Seurat side-by-side comparison.

sys_ram_gb <- function() {
  tryCatch(scLean:::cpp_resource_snapshot()$total_ram / (1024^3),
           error = function(e) 0)
}

test_that("PBMC 3k pipeline runs end-to-end", {
  skip_on_cran()
  skip_if_not_installed("Seurat")
  # Require ~6GB free RAM for Lanczos + dense moment buffers
  mem_gb <- sys_ram_gb()
  skip_if(mem_gb > 0 && mem_gb < 10,
    sprintf("System has %.0f GB RAM; PBMC 3k test needs 10+ GB in full suite", mem_gb))

  # ---- Download/cache PBMC 3k ----
  cache_dir <- file.path(Sys.getenv("SCLEAN_CACHE_DIR",
    unset = tools::R_user_dir("scLean", "data")))
  dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
  tmp_tar <- file.path(cache_dir, "pbmc3k.tar.gz")
  data_dir <- file.path(cache_dir, "filtered_gene_bc_matrices", "hg19")

  if (!dir.exists(data_dir)) {
    skip_if_offline()
    options(timeout = 300)
    download.file(
      "https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz",
      tmp_tar, mode = "wb")
    untar(tmp_tar, exdir = cache_dir)
  }

  # ---- scLean pipeline (streaming import) ----
  h5_path <- tempfile(fileext = ".h5")
  scLean:::stream_10x_to_hdf5(data_dir, h5_path, "RNA")
  sc_assay <- CreateSCleanAssayFromHDF5(h5_path, assay = "RNA")

  sc_assay <- NormalizeData(sc_assay, verbose = FALSE)
  sc_assay <- FindVariableFeatures(sc_assay, nfeatures = 2000, verbose = FALSE)
  sc_assay <- ScaleData(sc_assay, verbose = FALSE)
  sc_assay <- RunPCA(sc_assay, npcs = 10, verbose = FALSE)

  # Read PCA results
  sc_pca <- read_dense_matrix(h5_path, "/assays/RNA/reductions/pca/embeddings")
  sc_stdev <- read_hdf5_double(h5_path, "/assays/RNA/reductions/pca/stdev")[1:10]

  # Read variable features
  var_int <- read_hdf5_int32(h5_path, "/assays/RNA/features/variable")
  sc_vf <- read_strings_from_hdf5(h5_path, "/features/names")[var_int == 1]

  # ---- Validation 1: PCA dimensions correct ----
  expect_equal(nrow(sc_pca), ncol(sc_assay),
    label = "PCA embeddings have correct number of cells")
  expect_equal(ncol(sc_pca), 10,
    label = "PCA embeddings have correct number of PCs")

  sc_loadings <- read_dense_matrix(h5_path, "/assays/RNA/reductions/pca/loadings")
  expect_equal(nrow(sc_loadings), nrow(sc_assay),
    label = "PCA loadings have correct number of genes")
  expect_equal(ncol(sc_loadings), 10,
    label = "PCA loadings have correct number of PCs")

  # PCA stdev must be non-zero and finite
  expect_true(all(is.finite(sc_stdev) & sc_stdev >= 0),
    label = "PCA stdev values are non-negative and finite")
  expect_gt(sc_stdev[1], 0,
    label = "First PC has positive stdev")

  # ---- Validation 2: Variable features ----
  expect_equal(length(sc_vf), 2000,
    label = "Variable features count equals nfeatures")
  expect_true(all(sc_vf %in% read_strings_from_hdf5(h5_path, "/features/names")),
    label = "All variable features are valid gene names")

  # ---- Validation 3: Neighbors + Clustering ----
  sc_emb <- sc_pca[, 1:10]
  scLean:::cpp_find_neighbors(sc_emb, 20, "euclidean", 50, h5_path, "/assays/RNA")
  clust_res <- scLean:::cpp_find_clusters(h5_path, "/assays/RNA", 1, 0.8, 10)
  sc_clusters <- read_hdf5_int32(h5_path, "/assays/RNA/clusters/leiden")

  expect_equal(clust_res$n_clusters, length(unique(sc_clusters)),
    label = "Cluster count matches")
  expect_gte(clust_res$n_clusters, 2,
    label = "At least 2 clusters found")

  # ---- Validation 4: Modularity is non-trivial ----
  expect_gt(clust_res$modularity, 0,
    label = "Clustering modularity is positive")

  # ---- Cleanup ----
  if (file.exists(h5_path)) file.remove(h5_path)
})

test_that("PBMC 3k Seurat comparison", {
  skip_on_cran()
  skip_if_not_installed("Seurat")
  skip_if_not_installed("leidenbase")
  skip_if(
    tolower(Sys.getenv("SCLEAN_COMPARE", "true")) == "false",
    "SCLEAN_COMPARE=false: skipping Seurat comparison"
  )

  # ---- Download/cache PBMC 3k ----
  cache_dir <- file.path(Sys.getenv("SCLEAN_CACHE_DIR",
    unset = tools::R_user_dir("scLean", "data")))
  data_dir <- file.path(cache_dir, "filtered_gene_bc_matrices", "hg19")
  skip_if_not(dir.exists(data_dir),
    "PBMC 3k data not cached; run basic validation test first")

  counts <- suppressWarnings(Seurat::Read10X(data_dir))

  # ---- Run scLean first (streaming import, disk-backed, lower memory) ----
  h5_path <- tempfile(fileext = ".h5")
  scLean:::stream_10x_to_hdf5(data_dir, h5_path, "RNA")
  sc_assay <- CreateSCleanAssayFromHDF5(h5_path, assay = "RNA")

  sc_assay <- NormalizeData(sc_assay, verbose = FALSE)
  sc_assay <- FindVariableFeatures(sc_assay, nfeatures = 2000, verbose = FALSE)
  var_int <- read_hdf5_int32(h5_path, "/assays/RNA/features/variable")
  sc_vf <- read_strings_from_hdf5(h5_path, "/features/names")[var_int == 1]
  sc_assay <- ScaleData(sc_assay, verbose = FALSE)
  sc_assay <- RunPCA(sc_assay, npcs = 10, verbose = FALSE)
  sc_pca <- read_dense_matrix(h5_path, "/assays/RNA/reductions/pca/embeddings")
  sc_stdev <- read_hdf5_double(h5_path, "/assays/RNA/reductions/pca/stdev")[1:10]
  sc_emb <- sc_pca[, 1:10]
  scLean:::cpp_find_neighbors(sc_emb, 20, "euclidean", 50, h5_path, "/assays/RNA")
  clust_res <- scLean:::cpp_find_clusters(h5_path, "/assays/RNA", 1, 0.8, 10)
  sc_clusters <- read_hdf5_int32(h5_path, "/assays/RNA/clusters/leiden")

  # Free scLean memory before Seurat
  rm(sc_assay, sc_pca, sc_emb, var_int)
  gc()
  # Note: h5_path file stays on disk; sc_clusters, sc_vf, sc_stdev kept

  # ---- Run Seurat pipeline (in-memory, swap after gc) ----
  seurat_obj <- Seurat::CreateSeuratObject(counts, project = "pbmc3k")
  rm(counts)
  gc()

  seurat_obj <- Seurat::NormalizeData(seurat_obj, verbose = FALSE)
  seurat_obj <- Seurat::FindVariableFeatures(seurat_obj, nfeatures = 2000, verbose = FALSE)
  seurat_vf <- Seurat::VariableFeatures(seurat_obj)
  seurat_obj <- Seurat::ScaleData(seurat_obj, verbose = FALSE)
  seurat_obj <- Seurat::RunPCA(seurat_obj, npcs = 10, verbose = FALSE)
  seurat_stdev <- seurat_obj@reductions$pca@stdev[1:10]
  seurat_obj <- Seurat::FindNeighbors(seurat_obj, dims = 1:10, verbose = FALSE)
  seurat_obj <- Seurat::FindClusters(seurat_obj, resolution = 0.8,
    random.seed = 1, verbose = FALSE)
  seurat_clusters <- as.integer(Seurat::Idents(seurat_obj))

  # ---- Comparison 1: Variable features overlap ----
  overlap <- length(intersect(sc_vf, seurat_vf))
  jaccard <- overlap / length(union(sc_vf, seurat_vf))
  expect_gt(jaccard, 0.02,
    label = sprintf("Variable features Jaccard: %.3f (>0.02)", jaccard))

  # ---- Comparison 2: scLean PCA captures non-trivial variance ----
  sc_var_ratio <- sum(sc_stdev^2) / sum(seurat_stdev^2)
  expect_gt(sc_var_ratio, 0.1,
    label = "scLean PCA captures non-trivial variance")

  # ---- Comparison 3: Clustering ARI ----
  if (requireNamespace("mclust", quietly = TRUE)) {
    ari <- mclust::adjustedRandIndex(seurat_clusters, sc_clusters)
    expect_gt(ari, 0.25,
      label = sprintf("Clustering ARI: %.3f (>0.25)", ari))
  }

  # ---- Cleanup ----
  rm(seurat_obj)
  gc()
  if (file.exists(h5_path)) file.remove(h5_path)
})
