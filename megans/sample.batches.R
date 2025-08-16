samplebatches <- function(data_original, tensor_list, 
                          phase1_rows, phase2_rows, phase2_vars,
                          cat_vars, batch_size, at_least_p = 0.5, 
                          weights){
  weights <- as.vector(weights)
  phase1_bins <- cat_vars[!(cat_vars %in% phase2_vars)] 
  phase1_bins <- if (length(phase1_bins) > 0) {
    phase1_bins[sapply(phase1_bins, function(col) {
      length(unique(data_original[phase1_rows, col])) > 1
    })]
  } else {
    character(0)
  }
  
  for (i in 1:20){
    curr_var <- sample(phase1_bins, 1)
    p1 <- data_original[phase1_rows, curr_var]
    p2 <- data_original[phase2_rows, curr_var]
    if (length(unique(p1)) >= 2 & length(unique(p2)) >= 2){
      break
    }
  }
  
  alloc_even <- function(total, levl) {
    k <- length(levl)
    if (k == 0L) return(integer(0))
    base <- total %/% k
    rem  <- total - base * k
    counts <- rep.int(base, k)
    if (rem > 0) counts[seq_len(rem)] <- counts[seq_len(rem)] + 1
    names(counts) <- levl
    counts
  }
  phase1_total <- alloc_even(floor(batch_size * (1 - at_least_p)), unique(p1))
  phase2_total <- alloc_even(ceiling(batch_size * at_least_p), unique(p2))
  values <- data_original[, curr_var] # All values for the current sampled onehot encoded variable
  samp_idx <- c()
  w1 <- weights[phase1_rows]
  w2 <- weights[phase2_rows]
  new_weights <- c()

  for (i in unique(p1)){
    curr_p1rows <- phase1_rows[p1 == i]
    sampled <- sample(1:length(curr_p1rows), phase1_total[[i]],
                      replace = phase1_total[[i]] > sum(p1 == i))
    samp_idx <- c(samp_idx, curr_p1rows[sampled])
    new_weights <- c(new_weights, (w1[p1 == i])[sampled])
  }
  for (i in unique(p2)){
    curr_p2rows <- phase2_rows[p2 == i]
    sampled <- sample(1:length(curr_p2rows), phase2_total[[i]],
                      replace = phase2_total[[i]] > sum(p2 == i))
    samp_idx <- c(samp_idx, curr_p2rows[sampled])
    new_weights <- c(new_weights, (w2[p2 == i])[sampled])
  }
  batches <- lapply(tensor_list, function(tensor) tensor[samp_idx, , drop = FALSE])
  batches[[length(batches) + 1]] <- torch_tensor(as.matrix(new_weights), device = tensor_list[[1]]$device)
  batches[[length(batches) + 1]] <- samp_idx
  return(batches)
}

# samplebatches <- function(data_original, data_training, tensor_list, 
#                           phase1_rows, phase2_rows, phase2_vars,
#                           new_col_names, batch_size, at_least_p = 0.5, 
#                           weights){
#   weights <- as.vector(weights)
#   cat_names <- unlist(new_col_names)
#   phase1_bins <- cat_names[!(cat_names %in% phase2_vars)] 
#   phase1_bins <- if (length(phase1_bins) > 0) {
#     phase1_bins[sapply(phase1_bins, function(col) {
#       length(unique(data_training[phase1_rows, col])) > 1
#     })]
#   } else {
#     character(0)
#   }
#   # 50vs50 in phase-1 rows and phase-2 rows.
#   phase1_total <- c(floor(batch_size * (1 - at_least_p) / 2), ceiling(batch_size * (1 - at_least_p) / 2))
#   phase2_total <- c(floor(batch_size * at_least_p / 2), ceiling(batch_size * at_least_p / 2))
#   
#   for (i in 1:20){
#     curr_var <- sample(phase1_bins, 1)
#     p1 <- data_training[phase1_rows, curr_var]
#     p2 <- data_training[phase2_rows, curr_var]
#     if (length(unique(p1)) >= 2 & length(unique(p2)) >= 2){
#       break
#     }
#   }
#   values <- data_training[, curr_var] # All values for the current sampled onehot encoded variable
#   samp_idx <- c()
#   w1 <- weights[phase1_rows]
#   w2 <- weights[phase2_rows]
#   new_weights <- c()
#   m <- 1
#   for (i in unique(values)){
#     curr_p1rows <- phase1_rows[p1 == i]
#     sampled <- sample(1:length(curr_p1rows), phase1_total[m],
#                       replace = phase1_total[m] > sum(p1 == i))# , 
#     # prob = w1[p1 == i] / sum(w1[p1 == i]))
#     samp_idx <- c(samp_idx, curr_p1rows[sampled])
#     new_weights <- c(new_weights, (w1[p1 == i])[sampled])
#     
#     curr_p2rows <- phase2_rows[p2 == i]
#     sampled <- sample(1:length(curr_p2rows), phase2_total[m],
#                       replace = phase2_total[m] > sum(p2 == i))# , 
#     # prob = w2[p2 == i] / sum(w2[p2 == i]))
#     samp_idx <- c(samp_idx, curr_p2rows[sampled])
#     new_weights <- c(new_weights, (w2[p2 == i])[sampled])
#     m <- m + 1
#   }
#   batches <- lapply(tensor_list, function(tensor) tensor[samp_idx, , drop = FALSE])
#   batches[[length(batches) + 1]] <- torch_tensor(as.matrix(new_weights), device = tensor_list[[1]]$device)
#   batches[[length(batches) + 1]] <- samp_idx
#   return(batches)
# }
# 
