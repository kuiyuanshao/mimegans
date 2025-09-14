reconLoss <- function(fake, true, fake_proj, true_proj, I, params, num_inds, cat_inds, cats_p1, cats_p2, cats_mode){
  use_mm <- (length(num_inds) > 0L) && (params$alpha != 0)
  use_ce <- (length(cat_inds) > 0L) && (params$beta != 0)
  
  if (!use_mm && !use_ce)
    return (torch_tensor(0, device = fake$device))
  
  mm <- if (use_mm) {
    params$alpha * nnf_mse_loss(fake[I, num_inds, drop = F], true[I, num_inds, drop = F])
  } else NULL
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
                       torch_mean(true$view(c(true$size(1), -1)), dim = 1), 2) +
            torch_norm(torch_std(fake$view(c(fake$size(1), -1)), dim = 1) - 
                         torch_std(true$view(c(true$size(1), -1)), dim = 1), 2))
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

activationFun <- function(fake, all_cats, params, gen = F){
  hard_flag <- if (gen) TRUE else isTRUE(params$hard)
  for (cat in all_cats){
    fake[, cat] <- nnf_gumbel_softmax(fake[, cat, drop = F], tau = params$tau, hard = hard_flag)
  }
  return (fake)
}

projCat <- function(fake, CM_tensors, cats){
  fake_result <- fake$clone()
  for (c in names(cats)) {
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
  d_interpolates <- D(interpolates)[[1]]
  fake <- torch_ones(d_interpolates$size(), device = device)
  fake$requires_grad <- FALSE
  
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

lossCalc <- function(gen, true, info){
  rmse_num <- sum(sqrt(colMeans((gen$gsample[[1]][, info$phase2_vars[info$phase2_vars %in% info$num_vars]] - 
                                   true[, info$phase2_vars[info$phase2_vars %in% info$num_vars]])^2)))
  mis_cat <- 0
  for (i in info$phase2_vars[info$phase2_vars %in% info$cat_vars]){
    tb <- table(gen$gsample[[1]][[i]], true[[i]])
    mis_num <- sum(tb) - sum(diag(tb))
    mis_cat <- mis_cat + mis_num / nrow(true)
  }
  return (c(rmse_num, mis_cat))
}