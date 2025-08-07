recon_loss <- function(fake, true, I, encode_result, vars, phase2_cats, params, num_inds, cat_inds){
  mm_term <- torch_tensor(0, dtype = fake$dtype, device = fake$device)
  if (length(num_inds) > 0){
    if (params$alpha != 0){
      mm_term <- params$alpha * 
        nnf_mse_loss(fake[I, num_inds, drop = FALSE],
                     true[I, num_inds, drop = FALSE], reduction = "mean")
    }
  }
  ce_term <- torch_tensor(0, dtype = fake$dtype, device = fake$device)
  if (length(cat_inds) > 0){
    if (params$beta != 0) {
      ce_term <- params$beta *
        cross_entropy_loss(fake, true, I, encode_result, vars, phase2_cats)
    }
  }
  return (mm_term + ce_term)
}

cross_entropy_loss <- function(fake, true, I, encode_result, vars, phase2_cats){
  cats <- encode_result$binary_indices[which(sapply(names(encode_result$new_col_names), function(col_names) {
    any(col_names %in% phase2_cats)
  }))]
  cats_mode <- encode_result$binary_indices[which(sapply(encode_result$new_col_names, function(col_names) {
    any(col_names %in% vars[!(vars %in% unlist(encode_result$new_col_names[phase2_cats]))])
  }))]
  loss <- list()
  i <- 1
  for (cat in cats){
    loss[[i]] <- nnf_cross_entropy(fake[, cat, drop = F], 
                                   torch_argmax(true[, cat, drop = F], dim = 2), 
                                   reduction = "mean")
    i <- i + 1
  }
  for (catmode in cats_mode){
    loss[[i]] <- nnf_cross_entropy(fake[I, catmode, drop = F], 
                                   torch_argmax(true[I, catmode, drop = F], dim = 2), 
                                   reduction = "mean")
    i <- i + 1
  }
  
  loss_t <- torch_stack(loss, dim = 1)$sum() / (length(cats) + length(cats_mode))
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
  return (fake)
}

proj_to_p1 <- function(fake, X, A, I, CM_tensors, data_encode, phase2_cats, phase1_cats_inds, phase2_cats_inds){
  A_cat <- A[, phase1_cats_inds, drop = F]
  notI <- I$logical_not()
  cats <- data_encode$binary_indices[which(sapply(names(data_encode$new_col_names), function(col_names) {
    any(col_names %in% phase2_cats)
  }))]
  eps <- 1e-6
  for (c in 1:length(cats)){
    cat <- cats[[c]]
    cm <- CM_tensors[[names(cats)[c]]]
    
    logit <- fake[notI, cat]
    prob <- nnf_softmax(logit, dim = 2)
    prob <- prob$clamp(eps, 1 - eps)
    prob_proj <- prob$matmul(cm)
    prob_proj <- prob_proj$clamp(eps, 1 - eps)
    prob_proj <- prob_proj / prob_proj$sum(dim = 2, keepdim=TRUE)
    logit_proj <- torch_log(prob_proj)
    tmp <- fake[notI, ]
    tmp[, cat] <- logit_proj
    fake[notI, ] <- tmp
  }
  for (k in 1:length(phase2_cats_inds)){
    X[notI, phase2_cats_inds[k]] <- A_cat[notI, k]
  }
  
  return (list(fake, X))
}

gradient_penalty <- function(D, real_samples, fake_samples, params, device) {
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

boundloss <- function(output, index, data_original, data_info, 
                      lb, ub, phase2_m, 
                      num.normalizing, cat.encoding, 
                      data_encode, data_norm){
  output <- as.data.frame(as.matrix(output$clone()$detach()$cpu()))
  names(output) <- names(phase2_m)
  denormalize <- paste("denormalize", num.normalizing, sep = ".")
  decode <- paste("decode", cat.encoding, sep = ".")
  
  curr_gsample <- do.call(decode, args = list(
    data = output,
    encode_obj = data_encode
  ))
  curr_gsample <- do.call(denormalize, args = list(
    data = curr_gsample,
    num_vars = data_info$num_vars, 
    norm_obj = data_norm
  ))$data
  idx1 <- match(data_info$phase1_vars[data_info$phase1_vars %in% data_info$num_vars], 
                names(data_original))
  idx2 <- match(data_info$phase2_vars[data_info$phase2_vars %in% data_info$num_vars], 
                names(curr_gsample))
  X <- data_original[index, idx1] - curr_gsample[, idx2]
  
  UB <- matrix(ub, nrow(X), length(ub), byrow = TRUE)
  LB <- matrix(lb, nrow(X), length(lb), byrow = TRUE)
  
  bound_penalty <- 100 * mean(pmax(as.matrix(X - UB), 0)^2 + pmax(as.matrix(LB - X), 0)^2)
  
  return (bound_penalty)
}
