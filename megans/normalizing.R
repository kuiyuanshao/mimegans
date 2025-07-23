normalize.mode <- function(data, num_vars, cond_vars, phase2_vars) {
  if (!require(mclust, quietly = TRUE)) {
    install.packages("mclust")
    library(mclust)
  }
  data_norm <- data
  mode_params <- list()
  for (col in num_vars) {
    curr_col <- data[[col]]
    curr_col_obs <- curr_col[!is.na(curr_col)]
    if (length(unique(curr_col_obs)) == 1 | col %in% cond_vars) {
      mc <- mclust::Mclust(curr_col_obs, G = 1, verbose = F)
    } else {
      mc <- mclust::Mclust(curr_col_obs, G = 1:9, verbose = F, modelNames = "V")
    }
    pred <- predict(mc, newdata = curr_col_obs)
    mode_labels <- as.numeric(as.factor(pred$classification))
    # mode_means <- c()
    # mode_sds <- c()
    mode_means <- mc$parameters$mean + 1e-6
    mode_sds <- sqrt(mc$parameters$variance$sigmasq) + 1e-6
    # 
    # if (length(mode_sds) != length(mode_means)){
    #   mode_sds <- rep(mode_sds, length(mode_means))
    # }
    curr_col_norm <- rep(NA, length(curr_col_obs))
    for (mode in sort(unique(mode_labels))) {
      mode <- as.numeric(mode)
      idx <- which(mode_labels == mode)
      # mode_means <- c(mode_means, mean(curr_col_obs[idx])) + 1e-6
      # mode_sds <- c(mode_sds, sd(curr_col_obs[idx]))
      if (is.na(mode_sds[mode]) | mode_sds[mode] == 0){
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
  num_vars <- num_vars[num_vars %in% names(data)]
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


normalize.zscore <- function(data, num_vars, phase1_vars, phase2_vars){
  means <- apply(data[, num_vars, drop = F], 2, mean, na.rm = T)
  sds <- apply(data[, num_vars, drop = F], 2, sd, na.rm = T)
  names(means) <- names(sds) <- num_vars 
  
  means[phase2_vars] <- means[phase1_vars]
  sds [phase2_vars] <- sds [phase1_vars] 
  
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
    }
  })))
  names(data_denorm) <- names(data)
  return (list(data = data_denorm))
}
