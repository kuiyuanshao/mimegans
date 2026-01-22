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

sampleFun <- function(indices, elements, total, pi){
  remaining_pool <- indices
  total_deficit <- 0

  samp_idx <- NULL
  samp_pi <- NULL
  
  for (i in unique(elements)){
    curr_rows <- indices[elements == i]
    sampled <- sample(1:length(curr_rows), total[[i]],
                      replace = total[[i]] > length(curr_rows))
    samp_idx <- c(samp_idx, curr_rows[sampled])
    samp_pi <- c(samp_pi, pi[sampled])
  }
  return (list(samp_idx, samp_pi))
}

sampleBatch <- function(data_original, tensor_list, phase1_bins,
                        phase1_rows, phase2_rows, 
                        batch_size, at_least_p = 0.5, 
                        weights, net){
  curr_var <- sample(phase1_bins, 1)
  if (net == "G"){
    p1 <- data_original[phase1_rows, curr_var]
    p2 <- data_original[phase2_rows, curr_var]   
    
    phase1_total <- alloc_even(round(batch_size * (1 - at_least_p)), unique(p1))
    phase2_total <- alloc_even(round(batch_size * at_least_p), unique(p2))
    samp_res_p1 <- sampleFun(phase1_rows, p1, phase1_total, 1 - 1 / (weights[phase1_rows]))
    samp_res_p2 <- sampleFun(phase2_rows, p2, phase2_total, 1 / (weights[phase1_rows]))
    
    samp_idx <- c(samp_res_p1[[1]], samp_res_p2[[1]])
    samp_pi <- c(samp_res_p1[[2]], samp_res_p2[[2]])
  }else{
    p2 <- data_original[phase2_rows, curr_var]
    phase2_total <- alloc_even(batch_size, unique(p2))
    samp_res <- sampleFun(phase2_rows, p2, phase2_total, 1 / (weights[phase1_rows]))
    samp_idx <- samp_res[[1]]
    samp_pi <- samp_res[[2]]
  }

  batches <- lapply(tensor_list, function(tensor) tensor[samp_idx, ])
  batches[[length(batches) + 1]] <- torch_tensor(1 / samp_pi, device = tensor_list[[1]]$device)
  batches[[length(batches) + 1]] <- samp_idx
  batches[[length(batches) + 1]] <- curr_var
  return (batches)
}

# samplebatches <- function(data_original, data_training, tensor_list, 
#                           phase1_rows, phase2_rows, phase2_vars,
#                           design_col_names, batch_size, at_least_p = 0.5, 
#                           weights){
#   weights <- as.vector(weights)
#   cat_names <- unlist(design_col_names)
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
#   design_weights <- c()
#   m <- 1
#   for (i in unique(values)){
#     curr_p1rows <- phase1_rows[p1 == i]
#     sampled <- sample(1:length(curr_p1rows), phase1_total[m],
#                       replace = phase1_total[m] > sum(p1 == i))# , 
#     # prob = w1[p1 == i] / sum(w1[p1 == i]))
#     samp_idx <- c(samp_idx, curr_p1rows[sampled])
#     design_weights <- c(design_weights, (w1[p1 == i])[sampled])
#     
#     curr_p2rows <- phase2_rows[p2 == i]
#     sampled <- sample(1:length(curr_p2rows), phase2_total[m],
#                       replace = phase2_total[m] > sum(p2 == i))# , 
#     # prob = w2[p2 == i] / sum(w2[p2 == i]))
#     samp_idx <- c(samp_idx, curr_p2rows[sampled])
#     design_weights <- c(design_weights, (w2[p2 == i])[sampled])
#     m <- m + 1
#   }
#   batches <- lapply(tensor_list, function(tensor) tensor[samp_idx, , drop = FALSE])
#   batches[[length(batches) + 1]] <- torch_tensor(as.matrix(design_weights), device = tensor_list[[1]]$device)
#   batches[[length(batches) + 1]] <- samp_idx
#   return(batches)
# }
# 
