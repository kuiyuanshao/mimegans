g_loss <- function(y_fake, params, ...){
  return (params$gamma * -torch_mean(y_fake))
}

d_loss <- function(dnet, true, fake, params, device){
  y_fake <- dnet(fake)
  y_true <- dnet(true)
  gradient_penalty<- gradient_penalty(dnet, true, fake, pac = params$pac, device = device)
  d_loss <- -(torch_mean(y_true) - torch_mean(y_fake)) + params$lambda * gradient_penalty
  return (d_loss)
}

cross_entropy_loss <- function(fake, true, encode_result, vars, size){
  cats <- encode_result$binary_indices[which(sapply(encode_result$new_col_names, function(col_names) {
    any(col_names %in% vars)
  }))]
  loss <- list()
  i <- 1
  for (cat in cats){
    loss[[i]] <- nnf_cross_entropy(fake[, cat - size, drop = F], 
                                   torch_argmax(true[, cat - size, drop = F], dim = 2), 
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

