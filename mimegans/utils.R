pacman::p_load(torch)

reconLoss <- function(fake, true, fake_proj, true_proj, I, params, 
                      num_inds, cat_inds, 
                      ce_groups_p2, ce_groups_mode, total_cat_count){
  
  use_mm <- (length(num_inds) > 0L) && (params$alpha != 0)
  use_ce <- (length(cat_inds) > 0L) && (params$beta != 0)
  
  if (!use_mm && !use_ce)
    return (torch_tensor(0, device = fake$device))
  
  # Calculate count on GPU to avoid CPU sync
  n_I <- I$sum()
  
  mm <- if (use_mm) {
    # Division by (n + eps) handles the 0-row case purely on GPU.
    mse_sum <- nnf_mse_loss(fake[I, num_inds], true[I, num_inds], reduction = "sum")
    params$alpha * (mse_sum / (n_I + 1e-8))
  } else NULL
  
  ce <- if (use_ce) {
    params$beta *
      ceLoss(fake, true, fake_proj, true_proj, I, n_I, params, ce_groups_p2, ce_groups_mode, total_cat_count)
  } else NULL
  
  if (is.null(mm)) return(ce)
  if (is.null(ce)) return(mm)
  
  mm$add_(ce)
  return(mm)
}

infoLoss <- function(fake, true){
  f_flat <- fake$view(c(fake$size(1), -1))
  t_flat <- true$view(c(true$size(1), -1))
  
  return (torch_norm(torch_mean(f_flat, dim = 1) - torch_mean(t_flat, dim = 1), 2) +
            torch_norm(torch_std(f_flat, dim = 1) - torch_std(t_flat, dim = 1), 2))
}

# Batched Cross Entropy Loss
ceLoss <- function(fake, true, fake_proj, A, I, n_I, params, ce_groups_p2, ce_groups_mode, total_cat_count){
  loss_sum <- torch_tensor(0, device = fake$device)
  
  notI <- I$logical_not()
  n_notI <- notI$sum()
  
  # Helper: Compute loss for a group (Batch, K, L)
  # Slicing with empty mask returns empty tensor. CrossEntropy(sum) on empty returns 0.
  calc_batch_ce <- function(input, target_onehot, group, col_indices, mask_idx){
    inp_sub <- input[mask_idx, col_indices]
    tgt_sub <- target_onehot[mask_idx, col_indices]
    
    n_curr <- inp_sub$size(1)
    # Reshape: (N, K, L) -> (N, L, K) for CrossEntropy
    inp_view <- inp_sub$view(c(n_curr, group$k, group$L))$permute(c(1, 3, 2))
    
    # Calculate target indices (Argmax) on the fly
    tgt_idx <- torch_argmax(tgt_sub$view(c(n_curr, group$k, group$L)), dim = 3)
    
    return(nnf_cross_entropy(inp_view, tgt_idx, reduction = "sum"))
  }
  
  for (group in ce_groups_p2){
    if (params$cat == "projp1" | params$cat == "projp2"){
      idx_A <- if(!is.null(group$p1_idx)) group$p1_idx else group$p2_idx
      
      l1 <- calc_batch_ce(fake_proj, A, group, idx_A, notI)
      l2 <- calc_batch_ce(fake, true, group, group$p2_idx, I)
      
      # Tensor division
      term <- (params$proj_weight * l1 + l2) / (n_notI + n_I + 1e-8)
      loss_sum$add_(term)
    } else {
      l2 <- calc_batch_ce(fake, true, group, group$p2_idx, I)
      loss_sum$add_(l2 / (n_I + 1e-8))
    }
  }
  
  for (group in ce_groups_mode){
    l_mode <- calc_batch_ce(fake, true, group, group$idx, I)
    loss_sum$add_(l_mode / (n_I + 1e-8))
  }
  
  return (loss_sum / total_cat_count)
}

# Batched Gumbel Softmax
activationFun <- function(fake, cat_groups, all_nums, params, gen = F){
  hard_flag <- if (gen) TRUE else isTRUE(params$hard)
  
  for (group in cat_groups){
    subset <- fake[, group$indices]
    subset_view <- subset$view(c(fake$size(1), group$k, group$L))
    act <- nnf_gumbel_softmax(subset_view, tau = params$tau, hard = hard_flag, dim = 3)
    fake[, group$indices] <- act$view(c(fake$size(1), -1))
  }
  return (fake)
}

# Batched Projection
projCat <- function(fake, proj_groups){
  fake_result <- fake$clone()
  for (group in proj_groups) {
    subset <- fake[, group$indices]
    subset_view <- subset$view(c(fake$size(1), group$k, group$L))
    
    prob <- nnf_softmax(subset_view, dim = 3)
    # Batched matrix multiplication: (N, K, 1, L) x (K, L, L)
    proj <- torch_matmul(prob$unsqueeze(3), group$matrices)$squeeze(3)
    
    logits_obs <- torch_log(proj$clamp(1e-8, 1 - 1e-8))
    fake_result[, group$indices] <- logits_obs$view(c(fake$size(1), -1))
  }
  return (fake_result)
}

gradientPenalty <- function(D, real_samples, fake_samples, params, device, ones_buf = NULL) {
  batch_size <- real_samples$size(1)
  alp <- torch_rand(batch_size, 1, device = device)
  interpolates <- (alp * real_samples + (1 - alp) * fake_samples)$requires_grad_(TRUE)
  
  d_interpolates <- D(interpolates)
  
  fake <- if (!is.null(ones_buf) && ones_buf$size(1) == d_interpolates$size(1)) {
    ones_buf
  } else {
    torch_ones(d_interpolates$size(), device = device)
  }
  fake <- fake$detach() # Ensure no graph attachment
  
  gradients <- torch::autograd_grad(
    outputs = d_interpolates,
    inputs = interpolates,
    grad_outputs = fake,
    create_graph = TRUE,
    retain_graph = TRUE
  )[[1]]
  
  if (params$pac > 1){
    gradients <- gradients$reshape(c(-1, params$pac * interpolates$size(2)))
  }
  gradient_penalty <- torch_mean((torch_norm(gradients, p = 2, dim = 2) - 1) ^ 2)
  return (gradient_penalty)
}

lossCalc <- function(gen, true, info, inds){
  rmse_num <- 0; mis_cat <- 0
  num_vars <- info$phase2_vars[info$phase2_vars %in% info$num_vars]
  cat_vars <- info$phase2_vars[info$phase2_vars %in% info$cat_vars]
  
  if (length(num_vars) > 0) {
    diff_sq <- (gen$gsample[[1]][inds, num_vars, drop = FALSE] - true[inds, num_vars, drop = FALSE])^2
    rmse_num <- sum(sqrt(colMeans(diff_sq, na.rm = TRUE)))
  }
  
  n_inds <- sum(inds)
  if (length(cat_vars) > 0 && n_inds > 0) {
    for (i in cat_vars){
      mis_num <- sum(gen$gsample[[1]][[i]][inds] != true[[i]][inds])
      mis_cat <- mis_cat + mis_num / n_inds
    }
  }
  return (c(rmse_num, mis_cat))
}