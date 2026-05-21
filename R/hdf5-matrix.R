#' HDF5BackedMatrix: A matrix proxy backed by HDF5 storage
#'
#' This R6 class implements the standard R matrix interface (dim, dimnames, [, [<-, t, %*%)
#' but reads/writes data from HDF5 on the fly. Designed to be a drop-in replacement
#' for dgCMatrix in Seurat layers.
#'
#' @export
HDF5BackedMatrix <- R6::R6Class(
  "HDF5BackedMatrix",

  public = list(
    #' @field hdf5_path  Path to the HDF5 file
    hdf5_path  = NULL,

    #' @field group_path  HDF5 group containing the CSC matrix datasets
    group_path = NULL,

    #' @field dims  Dimensions of the matrix
    dims       = NULL,

    #' @field dimnames  Row and column names
    dimnames   = NULL,

    #'
    #' Create a new HDF5BackedMatrix
    #' @param hdf5_path  Path to HDF5 file
    #' @param group_path  Group path within the file
    initialize = function(hdf5_path, group_path) {
      self$hdf5_path  <- hdf5_path
      self$group_path <- group_path

      # Read dimensions from the file
      shape <- read_hdf5_shape(hdf5_path, group_path)
      self$dims <- as.integer(shape)

      # Try to read dimnames
      tryCatch({
        rn <- read_strings_from_hdf5(hdf5_path, "/features/names")
        cn <- read_strings_from_hdf5(hdf5_path, "/cells/names")
        self$dimnames <- list(rn, cn)
      }, error = function(e) {
        self$dimnames <- list(
          paste0("gene", seq_len(self$dims[1])),
          paste0("cell", seq_len(self$dims[2]))
        )
      })
    },

    #'
    #' Get dimensions
    #' @return Integer vector of length 2
    dim = function() {
      self$dims
    },

    #'
    #' Get number of rows
    #' @return Integer
    nrow = function() {
      self$dims[1]
    },

    #'
    #' Get number of columns
    #' @return Integer
    ncol = function() {
      self$dims[2]
    },

    #'
    #' Subset rows and columns
    #' @param i Row indices
    #' @param j Column indices
    #' @param drop  Whether to drop dimensions
    #' @return A dense matrix or vector
    get = function(i, j, drop = TRUE) {
      if (missing(i)) i <- seq_len(self$dims[1])
      if (missing(j)) j <- seq_len(self$dims[2])

      read_hdf5_dense_chunk(
        self$hdf5_path, self$group_path,
        as.integer(min(i)) - 1L, as.integer(length(i)),
        as.integer(min(j)) - 1L, as.integer(length(j))
      )
    },

    #'
    #' Convert to in-memory matrix
    #' @return A dgCMatrix
    as_matrix = function() {
      m <- read_hdf5_as_dgCMatrix(self$hdf5_path, self$group_path)
      dn <- m$Dimnames
      if (is.null(dn)) dn <- list(character(0), character(0))
      methods::new("dgCMatrix",
        i = m$i, p = m$p, x = m$x,
        Dim = m$Dim, Dimnames = dn,
        factors = list()
      )
    },

    #'
    #' Transpose (lazy: returns a new proxy with swapped dims)
    #' @return HDF5BackedMatrix with transposed access
    t = function() {
      # For sparse CSC, transpose requires CSR read or full materialization
      # Return a materialized transpose for correctness
      m <- self$as_matrix()
      t(m)
    },

    #'
    #' Matrix multiplication
    #' @param x  Another matrix or vector
    #' @return The product
    mat_mult = function(x) {
      if (is.vector(x)) {
        hdf5_matvec(self$hdf5_path, self$group_path, as.numeric(x))
      } else if (inherits(x, "HDF5BackedMatrix")) {
        # Materialize both for now
        a <- self$as_matrix()
        b <- x$as_matrix()
        a %*% b
      } else {
        a <- self$as_matrix()
        a %*% x
      }
    }
  )
)

# --- S3 methods for HDF5BackedMatrix ---

#' @export
dim.HDF5BackedMatrix <- function(x) x$dim()

#' @rdname HDF5BackedMatrix
#' @param x An HDF5BackedMatrix object
#' @export
nrow.HDF5BackedMatrix <- function(x) x$dim()[1]

#' @rdname HDF5BackedMatrix
#' @param x An HDF5BackedMatrix object
#' @export
ncol.HDF5BackedMatrix <- function(x) x$dim()[2]

#' @export
dimnames.HDF5BackedMatrix <- function(x) x$dimnames

#' @export
`[.HDF5BackedMatrix` <- function(x, i, j, drop = TRUE) {
  x$get(i, j, drop)
}

#' @export
as.matrix.HDF5BackedMatrix <- function(x, ...) x$as_matrix()

#' @export
t.HDF5BackedMatrix <- function(x) x$t()

#' @export
`%*%.HDF5BackedMatrix` <- function(x, y) x$mat_mult(y)
