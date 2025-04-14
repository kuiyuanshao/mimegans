g_loss.gan <- function(y_fake, params, ...){
  return (params$gamma * -torch_mean(y_fake))
}

g_loss.recon <- function(y_fake, params, true, fake, 
                         num_inds, cat_inds, data_encode, phase2_vars){
  mse <- if (length(num_inds) > 0) nnf_mse_loss(true[, num_inds, drop = F], fake[, num_inds, drop = F]) else 0
  cross_entropy <- if (length(cat_inds) > 0) cross_entropy_loss(fake, true, data_encode, phase2_vars) else 0
  
  g_loss <- params$gamma * -torch_mean(y_fake) + params$alpha * mse + params$beta * cross_entropy
  
  return (g_loss)
}

d_loss.pacwgan_gp <- function(dnet, true, fake, params, device){
  y_fake <- dnet(fake)
  y_true <- dnet(true)
  gradient_penalty<- gradient_penalty(dnet, true, fake, pac = params$pac, device = device)
  d_loss <- -(torch_mean(y_true) - torch_mean(y_fake)) + params$lambda * gradient_penalty
  return (d_loss)
}

cross_entropy_loss <- function(fake, true, encode_result, phase2_vars){
  cat_phase2 <- encode_result$binary_indices[which(sapply(encode_result$new_col_names, function(col_names) {
    any(col_names %in% phase2_vars)
  }))]
  loss <- list()
  i <- 1
  for (cat in cat_phase2){
    loss[[i]] <- nnf_cross_entropy(fake[, cat], 
                                   torch_argmax(true[, cat], dim = 2), 
                                   reduction = "mean")
    i <- i + 1
  }
  loss_t <- torch_stack(loss, dim = 1)$sum()
  return (loss_t)
}

activation_fun <- function(fake, encode_result, vars, tau = 0.2, hard = F){
  cats <- encode_result$binary_indices[which(sapply(encode_result$new_col_names, function(col_names) {
    any(col_names %in% vars)
  }))]
  for (cat in cats){
    fake[, cat] <- nnf_gumbel_softmax(fake[, cat], tau = tau, hard = hard)
  }
  
  return (fake)
}

