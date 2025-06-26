normalize.mode <- function(data, num_vars, ...) {
  # count_modes <- function(x, adjust = 1.5, tol = 0, 
  #                         min_dist = 0, prop_drop = 0.10, 
  #                         ...) {
  #   d <- density(x, adjust = adjust, ...)
  #   y <- d$y;  xs <- d$x
  #   
  #   idx <- which(diff(sign(diff(y))) == -2) + 1
  #   
  #   if (tol > 0)
  #     idx <- idx[y[idx] > tol]
  #   
  #   if (min_dist > 0 && length(idx) > 1) {
  #     keep <- c(TRUE, diff(xs[idx]) >= min_dist)
  #     idx  <- idx[keep]
  #   }
  #   if (length(idx) > 1) {
  #     peak_h <- y[idx]
  #     
  #     left  <- c(1, idx[-length(idx)])
  #     right <- c(idx[-1], length(y))
  #     valley <- mapply(function(l, r) min(y[l:r]), left, right)
  #     
  #     prominence <- peak_h - valley
  #     keep <- prominence >= prop_drop * max(y) 
  #     idx <- idx[keep]
  #   }
  #   length(idx)
  # }
  count_modes <- function(x,
                          adjust   = 1,
                          tol      = 0,
                          min_dist = 0,
                          ...) {
    d <- density(x, adjust = adjust, ...)
    y  <- d$y
    idx <- which(diff(sign(diff(y))) == -2) + 1
    if (tol > 0)
      idx <- idx[y[idx] > tol]
    if (min_dist > 0 && length(idx) > 1) {
      keep <- c(TRUE, diff(d$x[idx]) >= min_dist)
      idx  <- idx[keep]
    }
    length(idx)
  }
  if (!require(mclust, quietly = TRUE)) {
    install.packages("mclust")
    library(mclust)
  }
  data_norm <- data
  mode_params <- list()
  
  for (col in num_vars) {
    curr_col <- data[[col]]
    curr_col_obs <- curr_col[!is.na(curr_col)]
    if (length(unique(curr_col)) == 1){
      mc <- Mclust(curr_col_obs, G = 1)
    }else{
      mc <- Mclust(curr_col_obs, 
                   G = min(c(5, count_modes(curr_col_obs, ...))))
    }
    pred <- predict(mc, newdata = curr_col_obs)
    mode_labels <- as.numeric(as.factor(pred$classification))
    mode_means <- numeric(length(unique(mode_labels)))
    mode_sds <- numeric(length(unique(mode_labels)))
    
    curr_col_norm <- rep(NA, length(curr_col_obs))
    for (mode in sort(unique(mode_labels))) {
      idx <- which(mode_labels == mode)
      mode_means[mode] <- mean(curr_col_obs[idx]) + 1e-6
      mode_sds[mode] <- sd(curr_col_obs[idx]) + 1e-6
      if (is.na(mode_sds[mode])){
        curr_col_norm[idx] <- (curr_col_obs[idx] - mode_means[mode])
      }else{
        curr_col_norm[idx] <- (curr_col_obs[idx] - mode_means[mode]) / (mode_sds[mode])
      }
    }
    mode_labels_curr_col <- rep(NA, length(curr_col))
    mode_labels_curr_col[!is.na(curr_col)] <- mode_labels
    curr_col[!is.na(curr_col)] <- curr_col_norm
    data_norm[[col]] <- curr_col
    if (length(unique(mode_labels)) > 1){
      data_norm[[paste0(col, "_mode")]] <- mode_labels_curr_col
    }
    mode_params[[col]] <- list(mode_means = mode_means, mode_sds = mode_sds)
  }
  return(list(data = data_norm, mode_params = mode_params))
}

denormalize.mode <- function(data, num_vars, norm_obj){
  mode_params <- norm_obj$mode_params
  for (col in num_vars){
    curr_col <- data[[col]]
    curr_labels <- data[[paste0(col, "_mode")]]
    curr_transform <- rep(NA, length(curr_col))
    
    mode_means <- mode_params[[col]][["mode_means"]]
    mode_sds <- mode_params[[col]][["mode_sds"]]
    for (mode in 1:length(unique(mode_means))){
      idx <- which(curr_labels == mode)
      if (is.na(mode_sds[as.integer(mode)])){
        curr_transform[idx] <- curr_col[idx] + mode_means[as.integer(mode)]
      }else if (length(mode_means) == 1){
        curr_transform <- curr_col * mode_sds + mode_means
      }else{
        curr_transform[idx] <- curr_col[idx] * mode_sds[as.integer(mode)] + mode_means[as.integer(mode)] 
      }
    }
    data[[col]] <- curr_transform
  }
  data_denorm <- data[, !grepl("_mode$", names(data))]
  return (list(data = data_denorm, data_mode = data))
}

normalize.minmax <- function(data, num_vars, ...){
  maxs <- apply(data[, num_vars, drop = F], 2, max, na.rm = T)
  mins <- apply(data[, num_vars, drop = F], 2, min, na.rm = T)
  data_norm <- as.data.frame(do.call(cbind, lapply(names(data), function(i){
    if (i %in% num_vars){
      return ((data[, i] - mins[i] + 1e-6) / (maxs[i] - mins[i] + 1e-6))
    }else{
      return (data[, i])
    }
  })))
  names(data_norm) <- names(data)
  return (list(data = data_norm,
               maxs = maxs,
               mins = mins))
}

denormalize.minmax <- function(data, num_vars, norm_obj){
  maxs <- norm_obj$maxs
  mins <- norm_obj$mins
  data_denorm <- as.data.frame(do.call(cbind, lapply(names(data), function(i){
    if (i %in% num_vars){
      return (data[, i] * (maxs[i] - mins[i] + 1e-6) + (mins[i] - 1e-6))
    }else{
      return (ifelse(data[, i] >= 0.5, 1, 0))
    }
  })))
  names(data_denorm) <- names(data)
  return (list(data = data_denorm))
}


normalize.zscore <- function(data, num_vars, ...){
  means <- apply(data[, num_vars, drop = F], 2, mean, na.rm = T)
  sds <- apply(data[, num_vars, drop = F], 2, sd, na.rm = T)
  data_norm <- as.data.frame(do.call(cbind, lapply(names(data), function(i){
    if (i %in% num_vars){
      normalized <- (data[, i] - means[i] + 1e-6) / ((sds[i] + 1e-6))
      return (normalized)
    }else{
      return (data[, i])
    }
  })))
  names(data_norm) <- names(data)
  return (list(data = data_norm,
               means = means,
               sds = sds))
}

denormalize.zscore <- function(data, num_vars, norm_obj){
  means <- norm_obj$means
  sds <- norm_obj$sds
  data_denorm <- as.data.frame(do.call(cbind, lapply(names(data), function(i){
    if (i %in% num_vars){
      return (data[, i] * ((sds[i] + 1e-6)) + (means[i] - 1e-6))
    }else{
      return (ifelse(data[, i] >= 0.5, 1, 0))
    }
  })))
  names(data_denorm) <- names(data)
  return (list(data = data_denorm))
}
