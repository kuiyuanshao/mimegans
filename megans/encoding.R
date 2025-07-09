encode.token <- function(data, cat_vars, phase1_vars, ...) {
  new_data <- data[, !names(data) %in% cat_vars, drop = F]
  cat_data <- data[, names(data) %in% cat_vars, drop = F]
  binary_col_indices <- list()
  binary_col_names <- list()
  n_unique <- list()
  for (col in cat_vars) {
    if (col %in% phase1_vars){
      new_data[[col]] <- as.numeric(as.factor(cat_data[[col]]))
      binary_col_indices[[col]] <- which(names(new_data) %in% col)
      binary_col_names[[col]] <- names(new_data)[binary_col_indices[[col]]]
      n_unique[[col]] <- length(unique(new_data[[col]]))
      next
    }
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
              new_col_names = binary_col_names, n_unique = n_unique))
}

decode.token <- function(data, encode_obj, ...) {
  binary_indices <- encode_obj$binary_indices
  original_data <- data[, -unlist(binary_indices)]
  for (var_name in names(binary_indices)) {
    indices <- binary_indices[[var_name]]
    if (length(encode_obj$new_col_names[[var_name]]) == 1){
      original_data[[var_name]] <- data[, indices]
      next
    }
    code <- apply(data[, indices, drop = FALSE], 1, function(i){
      k <- which.max(i)
      k
    })
    original_data[[var_name]] <- 
      sub(paste0("^", var_name, "_"), "", names(data[, indices, drop = FALSE]))[code]
  }
  return(original_data)
}

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

decode.onehot <- function(data, encode_obj, ...) {
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
