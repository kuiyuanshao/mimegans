reconLoss <- function(fake, true, fake_proj, true_proj, I, params, num_inds, cat_inds, cats, cats_mode){
  mm_term <- torch_tensor(0, dtype = fake$dtype, device = fake$device)
  if (length(num_inds) > 0){
    if (params$alpha != 0){
      mm_term <- params$alpha * 
        nnf_mse_loss(fake[I, num_inds, drop = F],
                     true[I, num_inds, drop = F], reduction = "mean")
    }
  }
  ce_term <- torch_tensor(0, dtype = fake$dtype, device = fake$device)
  if (length(cat_inds) > 0){
    if (params$beta != 0) {
      ce_term <- params$beta *
        ceLoss(fake, true, fake_proj, true_proj, I, params, cats, cats_mode)
    }
  }
  return (mm_term + ce_term)
}

ceLoss <- function(fake, true, fake_proj, true_proj, I, params, cats, cats_mode){
  loss <- torch_tensor(0, device = fake$device)
  notI <- I$logical_not()
  for (cat in cats){
    if (params$cat_proj){
      ce.1 <- - 0.2 * (true_proj[notI, cat, drop = F] * 
                 (fake_proj[notI, cat, drop = F]$clamp_min(1e-8))$log())$sum(dim = 2)
      ce.2 <- nnf_cross_entropy(fake[I, cat, drop = F], 
                                torch_argmax(true[I, cat, drop = F], dim = 2), 
                                reduction = "none")
      ce <- torch_cat(list(ce.1, ce.2), dim = 1)$mean()
      loss <- loss + ce
      
    }else{
      ce <- nnf_cross_entropy(fake[I, cat, drop = F], 
                              torch_argmax(true[I, cat, drop = F], dim = 2), 
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
  loss_t <- loss / (length(cats)+ length(cats_mode))
  return (loss_t)
}

activationFun <- function(fake, nums, cats, tau = 0.2, hard = F, gen = F){
  for (cat in cats){
    if (gen){
      p <- nnf_softmax(fake[, cat, drop = F], dim = 2)
      idx <- torch_multinomial(p, 1)
      onehot <- nnf_one_hot(idx, num_classes = length(cat))$squeeze(2)$float()
      fake[, cat] <- onehot
    }else{
      fake[, cat] <- nnf_gumbel_softmax(fake[, cat, drop = F], tau = tau, hard = hard)
    }
  }
  return (fake)
}

projP1 <- function(fake, X, A, I, CM_tensors, cats, phase1_cats_inds, phase2_cats_inds, eps = 1e-6){
  fake_result <- fake$clone()
  X_result <- X$clone()
  
  A_cat <- A[, phase1_cats_inds]
  notI <- I$logical_not()
  
  for (c in 1:length(cats)){
    cat <- cats[[c]]
    cm <- CM_tensors[[names(cats)[c]]]
    prob <- fake[notI, cat, drop = F]
    prob_proj <- prob$matmul(cm)
    tmp <- fake[notI, , drop = F]
    tmp[, cat] <- prob_proj
    fake_result[notI, ] <- tmp
  }
  for (k in 1:length(phase2_cats_inds)){
    X_result[notI, phase2_cats_inds[k]] <- A_cat[notI, k]
  }
  return (list(fake_result, X_result))
}

gradientPenalty <- function(D, real_samples, fake_samples, params, device) {
  alp <- torch_rand(real_samples$size(1), 1, device = device)
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
  if (params$pac > 1){
    gradients <- gradients$reshape(c(-1, params$pac * interpolates$size(2)))
  }
  gradient_penalty <- torch_mean((torch_norm(gradients, p = 2, dim = 2) - 1) ^ 2)
  return (gradient_penalty)
}
