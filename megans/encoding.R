encode.onehot <- function(data, cat_vars, ...) {
  new_data <- data[, !names(data) %in% cat_vars, drop = F]
  cat_data <- data[, names(data) %in% cat_vars, drop = F]
  binary_col_indices <- list()
  binary_col_names <- list()

  for (col in cat_vars) {
    unique_categories <- unique(na.omit(cat_data[[col]]))
    new_cols <- vector()
    for (category in unique_categories) {
      new_col_name <- paste0(col, "_", category)
      new_data[[new_col_name]] <- ifelse(cat_data[[col]] == category, 1, 0)
      new_cols <- c(new_cols, new_col_name)
    }
    binary_col_indices[[col]] <- which(names(new_data) %in% new_cols)
    binary_col_names[[col]] <- names(new_data)[binary_col_indices[[col]]]
  }
  return(list(data = new_data, binary_indices = binary_col_indices,
              new_col_names = binary_col_names))
}

decode.onehot <- function(data, encode_obj) {
  binary_indices <- encode_obj$binary_indices
  original_data <- data[, -unlist(binary_indices)]
  for (var_name in names(binary_indices)) {
    indices <- binary_indices[[var_name]]
    code <- apply(data[, indices, drop = FALSE], 1, function(i){
      k <- which.max(i)
      k
    })
    original_data[[var_name]] <- 
      sub(paste0("^", var_name, "_"), "", names(data[, indices, drop = FALSE]))[code]
  }
  return(original_data)
}
