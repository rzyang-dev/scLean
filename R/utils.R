# Internal utility functions for scLean

#' Null-coalescing infix operator
#' @param x  Left value
#' @param y  Right fallback value
#' @return x if not NULL, otherwise y
#' @noRd
`%||%` <- function(x, y) if (is.null(x)) y else x

#' Register S3 methods in the correct namespace
#' @param generic  Generic function name (e.g., "Seurat::NormalizeData")
#' @param class    S3 class name
#' @param env      Environment to register in
#' @noRd
s3_register <- function(generic, class, env = parent.frame()) {
  parts <- strsplit(generic, "::")[[1]]
  if (length(parts) == 1) {
    pkg <- NULL
    generic <- parts
  } else {
    pkg <- parts[1]
    generic <- parts[2]
  }

  if (!is.null(pkg)) {
    # Register in the package's namespace
    ns <- asNamespace(pkg)
    method_name <- paste0(generic, ".", class)
    if (!exists(method_name, envir = ns, inherits = FALSE)) {
      # Register the method
      registerS3method(generic, class, get(method_name, envir = env), ns)
    }
  }
}

#' Check if a variable is a numeric-like factor
#' @param x  A vector
#' @return logical
#' @noRd
is_numeric_like <- function(x) {
  if (is.numeric(x)) return(TRUE)
  if (is.factor(x)) {
    return(suppressWarnings(!any(is.na(as.numeric(levels(x))))))
  }
  return(FALSE)
}

#' Check if running on limited memory hardware
#' @noRd
is_low_memory <- function() {
  ram <- .get_max_ram_bytes()
  return(ram <= 4 * 1024^3)  # 4GB or less
}
