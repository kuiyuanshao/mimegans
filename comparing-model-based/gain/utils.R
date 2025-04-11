

normalize <- function(data, numCol, parameters = NULL){
  norm_data <- data
  
  if (is.null(parameters)){
    min_val <- max_val <- rep(0, ncol(data))
    for (i in 1:ncol(data)){
      if (numCol[i] == 0){
        next
      }
      min_val[i] <- min(norm_data[, i], na.rm = T)
      norm_data[, i] <- norm_data[, i] - min(norm_data[, i], na.rm = T)
      max_val[i] <- max(norm_data[, i], na.rm = T)
      norm_data[, i] <- norm_data[, i] / (max(norm_data[, i], na.rm = T) + 1e-6)
    }
    norm_parameters <- list(min_val = min_val,
                            max_val = max_val)
  }else{
    min_val <- parameters$min_val
    max_val <- parameters$max_val
    
    for (i in 1:ncol(data)){
      norm_data[, i] <- norm_data[, i] - min_val[i]
      norm_data[, i] <- norm_data[, i] / (max_val[i] + 1e-6)
    }
    norm_parameters <- parameters
  }
  return (list(norm_data = norm_data, norm_parameters = norm_parameters))
}

renormalize <- function(norm_data, norm_parameters, numCol){
  
  min_val <- norm_parameters$min_val
  max_val <- norm_parameters$max_val
  
  renorm_data <- norm_data
  for (i in 1:ncol(norm_data)){
    if (numCol[i] == 0){
      next
    }
    renorm_data[, i] <- renorm_data[, i] * (max_val[i] + 1e-6)
    renorm_data[, i] <- renorm_data[, i] + min_val[i]
  }
  
  return (renorm_data)
  
}


new_batch <- function(norm_data, data_mask, nRow, batch_size, device = "cpu"){
  # complete_idx <- sample(which(misRow == F))
  # miss_idx <- sample(which(misRow == T))
  # 
  # batch_complete_idx <- complete_idx[1:round(0.8 * batch_size)]
  # batch_miss_idx <- miss_idx[1:(batch_size - round(0.8 * batch_size))]
  # 
  # batch_idx <- c(batch_complete_idx, batch_miss_idx)
  rows <- sample(nRow)
  inds <- rows[1:batch_size]
  norm_curr_batch <- norm_data[inds, ]
  mask_curr_batch <- data_mask[inds, ]
  
  return (list(norm_curr_batch, mask_curr_batch))
}









