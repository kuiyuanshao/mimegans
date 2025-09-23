encode.onehot <- function(data, cat_vars, ori_cat_vars, phase1_vars, phase2_vars, ...) {
  new_data <- data[, !names(data) %in% cat_vars, drop = F]
  cat_data <- data[, names(data) %in% cat_vars, drop = F]
  binary_col_indices <- list()
  binary_col_names <- list()
  
  cat_phase2_idx <- which(phase2_vars %in% ori_cat_vars)
  for (col in cat_vars) {
    if (col %in% phase2_vars[cat_phase2_idx]){
      partner <- phase1_vars[match(col, phase2_vars)]
      unique_categories <- na.omit(unique(cat_data[[partner]]))
    }else{
      unique_categories <- na.omit(unique(cat_data[[col]]))
    }
    new_cols <- vector()
    for (category in sort(unique_categories)) {
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

decode.onehot <- function(data, encode_obj, ...) {
  matches <- lapply(encode_obj$new_col_names, function(x) any(x %in% names(data)))
  matched_names <- names(encode_obj$new_col_names)[unlist(matches)]
  
  binary_indices <- encode_obj$binary_indices[matched_names]
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
