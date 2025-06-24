g_loss <- function(y_fake, fake, true, encode_result, vars, params, num_inds, cat_inds, ...){
  ## ── always compute the GAN term ───────────────────────────────
  gan_term <- params$gamma * -torch_mean(y_fake)
  
  ## ── optionally compute MSE ────────────────────────────────────
  mse_term <- torch_tensor(0, dtype = gan_term$dtype, device = gan_term$device)
  if (length(num_inds) > 0){
    if (params$alpha != 0) {
      mse_term <- params$alpha *
        nnf_mse_loss(
          fake[, num_inds, drop = FALSE],
          true[, num_inds, drop = FALSE]
        )
    }
  }
  
  ## ── optionally compute cross-entropy ──────────────────────────
  ce_term <- torch_tensor(0, dtype = gan_term$dtype, device = gan_term$device)
  if (length(cat_inds) > 0){
    if (params$beta != 0) {
      ce_term <- params$beta *
        cross_entropy_loss(fake, true, encode_result, vars)
    }
  }
  ## ── aggregate and return ─────────────────────────────────────
  total_loss <- gan_term + mse_term + ce_term
  return(total_loss)
}


d_loss <- function(dnet, true, fake, params, device){
  y_fake <- dnet(fake)
  y_true <- dnet(true)
  gradient_penalty<- gradient_penalty(dnet, true, fake, pac = params$pac, device = device)
  d_loss <- -(torch_mean(y_true) - torch_mean(y_fake)) + params$lambda * gradient_penalty
  return (d_loss)
}

cross_entropy_loss <- function(fake, true, encode_result, vars){
  cats <- encode_result$binary_indices[which(sapply(encode_result$new_col_names, function(col_names) {
    any(col_names %in% vars)
  }))]
  loss <- list()
  i <- 1
  for (cat in cats){
    loss[[i]] <- nnf_cross_entropy(fake[, cat, drop = F], 
                                   torch_argmax(true[, cat, drop = F], dim = 2), 
                                   reduction = "mean")
    i <- i + 1
  }
  loss_t <- torch_stack(loss, dim = 1)$sum()
  return (loss_t)
}

activation_fun <- function(fake, encode_result, vars, tau = 1, hard = F, gen = F){
  cats <- encode_result$binary_indices[which(sapply(encode_result$new_col_names, function(col_names) {
    any(col_names %in% vars)
  }))]
  if (!gen){
    for (cat in cats){
      fake[, cat] <- nnf_gumbel_softmax(fake[, cat], tau = tau, hard = hard)
    }
  }else{
    for (cat in cats){
      p <- nnf_softmax(fake[, cat], dim = 2)
      idx <- torch_multinomial(p, 1)
      onehot <- nnf_one_hot(idx, num_classes = length(cat))$squeeze(2)$float()
      fake[, cat] <- onehot
    }
  }
  return (fake)
}

