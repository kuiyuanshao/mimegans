reconLoss <- function(fake, true, I, W, params, num_inds, cat_inds, cats, cats_mode){
  mm_term <- torch_tensor(0, dtype = fake$dtype, device = fake$device)
  if (length(num_inds) > 0){
    if (params$alpha != 0){
      mm_term <- params$alpha * 
        nnf_mse_loss(fake[I, , drop = F]$index_select(2, num_inds),
                     true[I, , drop = F]$index_select(2, num_inds), reduction = "mean")
    }
  }
  ce_term <- torch_tensor(0, dtype = fake$dtype, device = fake$device)
  if (length(cat_inds) > 0){
    if (params$beta != 0) {
      ce_term <- params$beta *
        ceLoss(fake, true, I, W, params, cats, cats_mode)
    }
  }
  return (mm_term + ce_term)
}

ceLoss <- function(fake, true, I, W, params, cats, cats_mode){
  loss <- list()
  i <- 1
  W_I <- W[I]
  W_I <- (W_I / torch_sum(W_I))$reshape(c(W_I$shape[1], 1))
  for (cat in cats){
    if (params$cat_proj){
      loss[[i]] <- nnf_cross_entropy(fake[, cat, drop = F], 
                                     torch_argmax(true[, cat, drop = F], dim = 2), 
                                     reduction = "mean")
    }else{
      ce <- nnf_cross_entropy(fake[I, cat, drop = F], 
                              torch_argmax(true[I, cat, drop = F], dim = 2), 
                              reduction = "none")
      loss[[i]] <- torch_sum(W_I * ce$reshape(c(W_I$shape[1], 1)))
    }
    i <- i + 1
  }
  for (catmode in cats_mode){
    ce <- nnf_cross_entropy(fake[I, catmode, drop = F], 
                            torch_argmax(true[I, catmode, drop = F], dim = 2), 
                            reduction = "none")
    loss[[i]] <- torch_sum(W_I * ce$reshape(c(W_I$shape[1], 1)))
    i <- i + 1
  }
  
  loss_t <- torch_stack(loss, dim = 1)$sum() / (length(cats) + length(cats_mode))
  return (loss_t)
}

activationFun <- function(fake, nums, cats, tau = 1, hard = F, gen = F){
  if (!gen){
    for (cat in cats){
      fake[, cat] <- nnf_gumbel_softmax(fake[, cat, drop = F], tau = tau, hard = hard)
    }
  }else{
    for (cat in cats){
      if (length(cat) == 1){
        next
      }
      p <- nnf_softmax(fake[, cat, drop = F], dim = 2)
      idx <- torch_multinomial(p, 1)
      onehot <- nnf_one_hot(idx, num_classes = length(cat))$squeeze(2)$float()
      fake[, cat] <- onehot
    }
  }
  return (fake)
}

projP1 <- function(fake, X, A, I, CM_tensors, cats, phase1_cats_inds, phase2_cats_inds, eps = 1e-6){
  A_cat <- A[, phase1_cats_inds]
  notI <- I$logical_not()
  
  for (c in 1:length(cats)){
    cat <- cats[[c]]
    cm <- CM_tensors[[names(cats)[c]]]
    
    logit <- fake[notI, cat, drop = F]
    prob <- nnf_softmax(logit, dim = 2)
    prob <- prob$clamp(eps, 1 - eps)
    
    prob_proj <- prob$matmul(cm)
    prob_proj <- prob_proj$clamp(eps, 1 - eps)
    prob_proj <- prob_proj / prob_proj$sum(dim = 2, keepdim=TRUE)
    
    logit_proj <- torch_log(prob_proj)
    
    tmp <- fake[notI, , drop = F]
    tmp[, cat] <- logit_proj
    fake[notI, ] <- tmp
  }
  for (k in 1:length(phase2_cats_inds)){
    X[notI, phase2_cats_inds[k]] <- A_cat[notI, k]
  }
  
  return (list(fake, X))
}

gradientPenalty <- function(D, real_samples, fake_samples, params, device, W_I) {
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
  # gradient_penalty <- torch_mean((torch_norm(gradients, p = 2, dim = 2) - 1) ^ 2)
  gradient_penalty <- (torch_norm(gradients, p = 2, dim = 2) - 1) ^ 2
  gradient_penalty <- gradient_penalty$reshape(c(gradient_penalty$shape[1], 1))
  gradient_penalty <- torch_sum(W_I * gradient_penalty)
  return (gradient_penalty)
}
