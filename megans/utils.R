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
  nums <- (1:(dim(fake)[2]))[!(1:(dim(fake)[2]) %in% unlist(cats))]
  if (!gen){
    for (cat in cats){
      fake[, cat] <- nnf_gumbel_softmax(fake[, cat], tau = tau, hard = hard)
    }
  }else{
    for (cat in cats){
      if (length(cat) == 1){
        next
      }
      p <- nnf_softmax(fake[, cat], dim = 2)
      idx <- torch_multinomial(p, 1)
      onehot <- nnf_one_hot(idx, num_classes = length(cat))$squeeze(2)$float()
      fake[, cat] <- onehot
    }
  }
  #fake[, nums] <- (fake[, nums] - fake[, nums]$mean()) / fake[, nums]$std()
  return (fake)
}

gradient_penalty <- function(D, real_samples, fake_samples, pac, device) {
  alp <- torch_rand(c(ceiling(real_samples$size(1) / pac), 1, 1))$to(device = device)
  pac <- torch_tensor(as.integer(pac), device = device)
  size <- torch_tensor(real_samples$size(2), device = device)
  
  alp <- alp$repeat_interleave(pac, dim = 2)$repeat_interleave(size, dim = 3)
  alp <- alp$reshape(c(-1, real_samples$size(2)))
  
  interpolates <- (alp * real_samples + (1 - alp) * fake_samples)$requires_grad_(TRUE)
  d_interpolates <- D(interpolates)
  
  fake <- torch_ones(d_interpolates$size(), device = device)
  fake$requires_grad <- FALSE
  
  gradients <- torch::autograd_grad(
    outputs = d_interpolates,
    inputs = interpolates,
    grad_outputs = fake,
    create_graph = TRUE,
    retain_graph = TRUE
  )[[1]]
  
  # Reshape gradients to group the pac samples together
  if (pac$item() > 1){
    gradients <- gradients$reshape(c(-1, pac$item() * size$item()))
  }
  gradient_penalty <- torch_mean((torch_norm(gradients, p = 2, dim = 2) - 1) ^ 2)
  
  return (gradient_penalty)
}

