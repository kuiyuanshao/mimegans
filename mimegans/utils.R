reconLoss <- function(fake, true, fake_proj, true_proj, C, I, params, num_inds, cat_inds, cats_p1, cats_p2, cats_mode){
  use_mm <- (length(num_inds) > 0L) && (params$alpha != 0)
  use_ce <- (length(cat_inds) > 0L) && (params$beta  != 0)
  
  if (!use_mm && !use_ce)
    return (torch_tensor(0, device = fake$device))
  
  if (use_mm){
    conditions <- torch_cat(list(true_proj, C), dim = 2)
    n <- true[I, ]$size(1)
    fake_X <- fake - fake$mean(dim = 1, keepdim = TRUE)
    true_X <- true - true$mean(dim = 1, keepdim = TRUE)
    conditions <- conditions - conditions$mean(dim = 1, keepdim = TRUE)
    covXfC <- fake_X$t()$matmul(conditions) / (n - 1)  
    covXtC <- true_X$t()$matmul(conditions) / (n - 1)  
    mm <- params$alpha * nnf_mse_loss(covXfC, covXtC)
  }else{
    mm <- NULL
  }
  ce <- if (use_ce) {
    params$beta *
      ceLoss(fake, true, fake_proj, true_proj, I, params, cats_p1, cats_p2, cats_mode)
  } else NULL
  
  if (is.null(mm)) return(ce)
  if (is.null(ce)) return(mm)
  mm + ce
}

infoLoss <- function(fake, true){
  return (torch_norm(torch_mean(fake$view(c(fake$size(1), -1)), dim = 1) - 
                       torch_mean(true$view(c(fake$size(1), -1)), dim = 1), 2) +
            torch_norm(torch_std(fake$view(c(fake$size(1), -1)), dim = 1) - 
                         torch_std(true$view(c(fake$size(1), -1)), dim = 1), 2))
}

ceLoss <- function(fake, true, fake_proj, A, I, params, cats_p1, cats_p2, cats_mode){
  loss <- torch_tensor(0, device = fake$device)
  notI <- I$logical_not()
  for (i in 1:length(cats_p2)){
    cat_p2 <- cats_p2[[i]]
    if (params$cat == "projp1" | params$cat == "projp2"){
      cat_p1 <- cats_p1[[i]]
      ce.1 <- nnf_cross_entropy(fake_proj[notI, cat_p2, drop = F], 
                                torch_argmax(A[notI, cat_p1, drop = F], dim = 2), 
                                reduction = "none")
      ce.2 <- nnf_cross_entropy(fake[I, cat_p2, drop = F], 
                                torch_argmax(true[I, cat_p2, drop = F], dim = 2), 
                                reduction = "none")
      ce <- torch_cat(list(ce.1, ce.2), dim = 1)$mean()
      loss <- loss + ce
    } else{
      ce <- nnf_cross_entropy(fake[I, cat_p2, drop = F], 
                              torch_argmax(true[I, cat_p2, drop = F], dim = 2), 
                              reduction = "mean")
      loss <- loss + ce
    }
  }
  for (catmode in cats_mode){
    ce <- nnf_cross_entropy(fake[I, catmode, drop = F],
                            torch_argmax(true[I, catmode, drop = F], dim = 2),
                            reduction = "mean")
    loss <- loss + ce
  }
  loss_t <- loss / (length(cats_p2) + length(cats_mode))
  
  return (loss_t)
}

activationFun <- function(fake, cats_mode, cats_p2, params, gen = F){
  cats <- c(cats_mode, cats_p2)
  for (cat in cats){
    if (gen){
      p <- nnf_gumbel_softmax(fake[, cat, drop = F], tau = params$tau, hard = T)
      fake[, cat] <- p
    }else{
      fake[, cat] <- nnf_gumbel_softmax(fake[, cat, drop = F], tau = params$tau, hard = params$hard)
    }
  }
  return (fake)
}

projCat <- function(fake, CM_tensors, cats){
  fake_result <- fake$clone()
  for (c in seq_along(cats)) {
    cat_idx <- cats[[c]]   
    cm <- CM_tensors[[c]] 
    prob <- nnf_softmax(fake[, cat_idx, drop = F], dim = 2)
    proj <- prob$matmul(cm) 
    logits_obs <- torch_log(proj$clamp(1e-8, 1 - 1e-8))
    fake_result[, cat_idx] <- logits_obs
  }
  return (fake_result)
}

gradientPenalty <- function(D, real_samples, fake_samples, params, device) {
  alp <- torch_rand(real_samples$size(1), 1, device = device)
  interpolates <- (alp * real_samples + (1 - alp) * fake_samples)$requires_grad_(TRUE)
  if (params$type_d == "infomlp" | params$type_d == "sninfomlp"){
    d_interpolates <- D(interpolates)[[1]]
  }else{
    d_interpolates <- D(interpolates)
  }
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
  if (params$pac > 1){
    gradients <- gradients$reshape(c(-1, params$pac * interpolates$size(2)))
  }
  gradient_penalty <- torch_mean((torch_norm(gradients, p = 2, dim = 2) - 1) ^ 2)
  return (gradient_penalty)
}
