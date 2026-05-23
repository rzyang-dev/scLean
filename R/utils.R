# Internal utility functions for scLean

#' Null-coalescing infix operator
#' @param x  Left value
#' @param y  Right fallback value
#' @return x if not NULL, otherwise y
#' @noRd
`%||%` <- function(x, y) if (is.null(x)) y else x
