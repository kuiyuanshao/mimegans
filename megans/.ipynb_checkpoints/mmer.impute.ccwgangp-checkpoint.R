pacman::p_load(progress, torch)

ccwgangp_default <- function(batch_size = 500, gamma = 1, alpha = 10, beta = 1, lambda = 10, 
                             lr_g = 1e-4, lr_d = 1e-4, g_betas = c(0.5, 0.9), d_betas = c(0.5, 0.9), 
                             g_weight_decay = 1e-6, d_weight_decay = 1e-6, 
                             g_dim = c(256, 256), pac = 5, 
                             n_g_layers = 5, n_d_layers = 3, 
                             at_least_p = 0.2, discriminator_steps = 1, scaling = 1){
  list(
    batch_size = batch_size, gamma = gamma, alpha = alpha, beta = beta, lambda = lambda,
    lr_g = lr_g, lr_d = lr_d, g_betas = g_betas, d_betas = d_betas, 
    g_weight_decay = g_weight_decay, d_weight_decay = d_weight_decay, 
    g_dim = g_dim, pac = pac, n_g_layers = n_g_layers, n_d_layers = n_d_layers, 
    at_least_p = at_least_p, discriminator_steps = discriminator_steps, scaling = scaling
  )
}

mmer.impute.ccwgangp <- function(data, m = 5, num.normalizing = "mode", cat.encoding = "onehot", device = "cpu",
                                 epochs = 3000, params = list(), data_info = list(), save.model = FALSE, save.step = 1000){
  params <- do.call("ccwgangp_default", params)
  device <- torch_device(device)
  
  list2env(params, envir = environment())
  list2env(data_info, envir = environment())
  
  phase1_rows <- which(is.na(data[[data_info$phase2_vars[1]]]))
  phase2_rows <- which(!is.na(data[[data_info$phase2_vars[1]]]))
  
  if (num.normalizing == "mode"){
    cat_vars <- c(cat_vars, paste0(num_vars, "_mode"))
    phase1_vars <- c(phase1_vars, paste0(phase1_vars[!(phase1_vars %in% cat_vars)], "_mode"))
    phase2_vars <- c(phase2_vars, paste0(phase2_vars[!(phase2_vars %in% cat_vars)], "_mode"))
  }
  
  normalize <- paste("normalize", num.normalizing, sep = ".")
  encode <- paste("encode", cat.encoding, sep = ".")
  
  #Weights are removed from the normalization
  data_norm <- do.call(normalize, args = list(
    data = data[, -which(names(data) == weight_var)], #Eliminates weight variable from the training
    num_vars = num_vars, scaling = scaling
  ))
  data_encode <- do.call(encode, args = list(
    data = data_norm$data,
    cat_vars = cat_vars
  ))
  
  nrows <- nrow(data_encode$data)
  ncols <- ncol(data_encode$data)
  
  #Encoding creates new variables corresponding to the categorical variables.
  phase1_vars <- c(phase1_vars[!phase1_vars %in% cat_vars], unlist(data_encode$new_col_names[phase1_vars]))
  phase2_vars <- c(phase2_vars[!phase2_vars %in% cat_vars], unlist(data_encode$new_col_names[phase2_vars]))
  
  #Prepare training tensors
  data_training <- data_encode$data
  #Reorder the data to Phase2 | Phase1, since the Generator only generates Phase2 data.
  data_training <- data_training[, c(phase2_vars, phase1_vars, 
                                     setdiff(names(data_training), c(phase2_vars, phase1_vars)))]
  
  p1_binary_indices_reordered <- lapply(data_encode$binary_indices, function(indices) {
    match(names(data_encode$data)[indices], names(data_training[, c(phase1_vars, phase2_vars, 
                                                                    setdiff(names(data_training), 
                                                                            c(phase1_vars, phase2_vars)))]))
  })
  p2_binary_indices_reordered <- lapply(data_encode$binary_indices, function(indices) {
    match(names(data_encode$data)[indices], names(data_training))
  })
  data_encode_p1 <- data_encode
  data_encode_p1$binary_indices <- p1_binary_indices_reordered
  data_encode$binary_indices <- p2_binary_indices_reordered
  
  data_mask <- torch_tensor(1 - is.na(data_training), device = device)
  #Phase1 Error-Prone Variables Tensors
  phase1_ep <- data_training[, phase1_vars, drop = F]
  #Phase1 Error-Free Varaibles Tensors
  phase1_ef <- torch_tensor(as.matrix(data_training[, !names(data_training) %in% c(phase1_vars, phase2_vars)]), 
                            device = device)
  #Phase2 Validated Variables Tensors
  phase2_t <- data_training[, phase2_vars, drop = F]
  
  #Phase1 Error-Prone numeric and binary column indices
  p1_num_inds <- which(names(phase1_ep) %in% num_vars)
  p1_cat_inds <- which(names(phase1_ep) %in% unlist(data_encode$new_col_names))
  #Phase2 Validated Variables numeric and binary column indices
  p2_num_inds <- which(names(phase2_t) %in% num_vars)
  p2_cat_inds <- which(names(phase2_t) %in% unlist(data_encode$new_col_names))
  
  #Replace all NA values with zeros and set to device.
  phase2_t[is.na(phase2_t)] <- 0 
  phase2_t <- torch_tensor(as.matrix(phase2_t), device = device)
  phase1_ep <- torch_tensor(as.matrix(phase1_ep), device = device)
  
  gnet_p1 <- generator(n_g_layers, g_dim, ncols, length(phase1_vars))$to(device = device)
  gnet_p2 <- generator(n_g_layers, g_dim, ncols, length(phase2_vars))$to(device = device)
  dnet_p1 <- discriminator(n_d_layers, ncols, pac = pac)$to(device = device)
  dnet_p2 <- discriminator(n_d_layers, ncols, pac = pac)$to(device = device)
  
  g_solver <- torch::optim_adam(c(gnet_p1$parameters, gnet_p2$parameters), lr = lr_g,
                                betas = g_betas, weight_decay = g_weight_decay)
  d_solver <- torch::optim_adam(c(dnet_p1$parameters, dnet_p2$parameters), lr = lr_d,
                                betas = d_betas, weight_decay = d_weight_decay)
  
  training_loss <- matrix(0, nrow = epochs, ncol = 4)
  pb <- progress_bar$new(
    format = paste0("Running :what [:bar] :percent eta: :eta | G Loss: :g_loss | D Loss: :d_loss | MSE: :mse | Cross-Entropy: :cross_entropy"),
    clear = FALSE, total = epochs, width = 200)
  
  if (save.step > 0){
    step_result <- list()
    p <- 1
  } 
  for (i in 1:epochs){
    d_loss_t <- 0
    for (d in 1:discriminator_steps){
      #### Discriminator for X
      batch <- samplebatches(data, data_training, list(data_mask, phase1_ep, phase1_ef, phase2_t), 
                             phase1_rows, phase2_rows, 
                             phase1_vars, phase2_vars, 
                             num_vars, data_encode$new_col_names, weight_var, 
                             batch_size, at_least_p = at_least_p)
      M <- batch[[1]]
      Xstar <- batch[[2]]
      C <- batch[[3]]
      X <- batch[[4]]
      I <- M[, 1] == 1
      
      fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), g_dim[1]))$to(device = device)
      fakez_XstarC <- torch_cat(list(fakez, Xstar, C), dim = 2) # Noise, X^*, C
      fake_X <- gnet_p2(fakez_XstarC)
      fake_X <- activation_fun(fake_X, data_encode, phase2_vars)
      
      fake_XXstarC <- torch_cat(list(fake_X, Xstar, C), dim = 2) # X_hat, X^*, C
      true_XXstarC <- torch_cat(list(X, Xstar, C), dim = 2) # X, X^*, C
      y_fake_p2 <- dnet_p2(fake_XXstarC[I, ])
      y_true_p2 <- dnet_p2(true_XXstarC[I, ])
      gradient_penalty_p2 <- gradient_penalty(dnet_p2, true_XXstarC[I, ], fake_XXstarC[I, ], pac = pac)
      
      d_loss_p2 <- -(torch_mean(y_true_p2) - torch_mean(y_fake_p2)) + lambda * gradient_penalty_p2
      ##################################################################
      
      #### Discriminator for Xstar
      batch <- samplebatches(data, data_training, list(data_mask, phase1_ep, phase1_ef, phase2_t), 
                             phase1_rows, phase2_rows, 
                             phase1_vars, phase2_vars, 
                             num_vars, data_encode$new_col_names, weight_var, 
                             batch_size, at_least_p = at_least_p)
      M <- batch[[1]]
      Xstar <- batch[[2]]
      C <- batch[[3]]
      X <- batch[[4]]
      I <- M[, 1] == 1
      
      fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), g_dim[1]))$to(device = device)
      fakez_XstarC <- torch_cat(list(fakez, Xstar, C), dim = 2) # Noise, X^*, C
      fake_X <- gnet_p2(fakez_XstarC)
      fake_X <- activation_fun(fake_X, data_encode, phase2_vars)
      fakez_XC <- torch_cat(list(fakez, fake_X, C), dim = 2) # Noise, X_hat, C
      fake_Xstar <- gnet_p1(fakez_XC)
      fake_Xstar <- activation_fun(fake_Xstar, data_encode_p1, phase1_vars)
      
      fake_XstarC <- torch_cat(list(fake_Xstar, fake_X, C), dim = 2) # X^*_hat, X, C
      true_XstarC <- torch_cat(list(Xstar, fake_X, C), dim = 2) # X^*, X, C
      y_fake_p1 <- dnet_p1(fake_XstarC)
      y_true_p1 <- dnet_p1(true_XstarC)
      gradient_penalty_p1 <- gradient_penalty(dnet_p1, true_XstarC, fake_XstarC, pac = pac)
      
      d_loss_p1 <- -(torch_mean(y_true_p1) - torch_mean(y_fake_p1)) + lambda * gradient_penalty_p1
      ##################################################################
      
      d_loss_t <- d_loss_t + d_loss_p1$item() + d_loss_p2$item()
      
      d_solver$zero_grad()
      d_loss_p1$backward()
      d_loss_p2$backward()
      d_solver$step()
    }
    
    batch <- samplebatches(data, data_training, list(data_mask, phase1_ep, phase1_ef, phase2_t),
                           phase1_rows, phase2_rows, phase1_vars, phase2_vars, 
                           num_vars, data_encode$new_col_names, weight_var, 
                           batch_size, at_least_p = at_least_p)
    M <- batch[[1]]
    Xstar <- batch[[2]]
    C <- batch[[3]]
    X <- batch[[4]]
    I <- M[, 1] == 1
    
    # --- SYMMETRIC CYCLE MAPPING
    fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), g_dim[1]))$to(device = device)
    fakez_XstarC <- torch_cat(list(fakez, Xstar, C), dim = 2)
    fake_X <- gnet_p2(fakez_XstarC)
    fake_X <- activation_fun(fake_X, data_encode, phase2_vars)
    
    fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), g_dim[1]))$to(device = device)
    fakez_XC <- torch_cat(list(fakez, fake_X, C), dim = 2)
    fake_Xstar <- gnet_p1(fakez_XC)
    fake_Xstar <- activation_fun(fake_Xstar, data_encode_p1, phase1_vars)
    # --- End cycle mapping
    
    # Adversarial loss
    y_fake_p2 <- dnet_p2(torch_cat(list(fake_X, Xstar, C), dim = 2)[I, ])
    y_fake_p1 <- dnet_p1(torch_cat(list(fake_Xstar, fake_X, C), dim = 2))
    g_adv_loss <- gamma * (-torch_mean(y_fake_p2) - torch_mean(y_fake_p1))
    
    # Cycle consistency losses: force cycle outputs to be close to original inputs (CHANGED)
    #cycle_loss_p1 <- lambda * cycle_consistency_loss(Xstar, fake_Xstar)
    #cycle_loss_p2 <- lambda * cycle_consistency_loss(X, fake_X)
    
    # Reconstruction losses
    mse_loss_p2 <- if (length(p2_num_inds) > 0) nnf_mse_loss(fake_X[I, p2_num_inds, drop = F], X[I, p2_num_inds, drop = F]) else 0
    cross_entropy_loss_p2 <- if (length(p2_cat_inds) > 0) cross_entropy_loss(fake_X[I, drop = F], X[I, drop = F], data_encode, phase2_vars) else 0
    mse_loss_p1 <- if (length(p1_num_inds) > 0) nnf_mse_loss(fake_Xstar[, p1_num_inds, drop = F], Xstar[, p1_num_inds, drop = F]) else 0
    cross_entropy_loss_p1 <- if (length(p1_cat_inds) > 0) cross_entropy_loss(fake_Xstar, Xstar, data_encode_p1, phase1_vars) else 0
    
    g_loss_p2 <- alpha * mse_loss_p2 + beta * cross_entropy_loss_p2
    g_loss_p1 <- alpha * mse_loss_p1 + beta * cross_entropy_loss_p1
    
    g_loss <- g_adv_loss + g_loss_p2 + g_loss_p1
    g_solver$zero_grad()
    g_loss$backward()
    g_solver$step()
    
    training_loss[i, ] <- c(g_loss$item(), d_loss_t / discriminator_steps,
                            ifelse(length(p2_num_inds) > 0, alpha * mse_loss_p2$item(), 0), 
                            ifelse(length(p2_cat_inds) > 0, beta * cross_entropy_loss_p2$item(), 0))
    pb$tick(tokens = list(
      what = "ccWGAN-GP",
      g_loss = sprintf("%.4f", training_loss[i, 1]),
      d_loss = sprintf("%.4f", training_loss[i, 2]),
      mse = ifelse(length(p2_num_inds) > 0, sprintf("%.4f", training_loss[i, 3]), 0),
      cross_entropy = ifelse(length(p2_cat_inds) > 0, sprintf("%.4f", training_loss[i, 4]), 0)
    ))
    Sys.sleep(1 / 10000)
    
    if (save.step > 0){
      if (i %% save.step == 0){
        result <- generateImpute(gnet_p2, m = 1, 
                                 data, data_norm, 
                                 data_encode, data_training, data_mask,
                                 phase2_vars, num_vars, weight_var, num.normalizing, cat.encoding, 
                                 batch_size, g_dim, device, torch_cat(list(phase1_ep, phase1_ef), dim = 2), phase2_t)
        step_result[[p]] <- result$gsample
        p <- p + 1
      }
    }
  }
  training_loss <- data.frame(training_loss)
  names(training_loss) <- c("G Loss", "D Loss", "MSE", "Cross-Entropy")
  result <- generateImpute(gnet_p2, m = m, 
                           data, data_norm, 
                           data_encode, data_training, data_mask,
                           phase2_vars, num_vars, weight_var, num.normalizing, cat.encoding, 
                           batch_size, g_dim, device, torch_cat(list(phase1_ep, phase1_ef), dim = 2), phase2_t)
  if (save.model){
    current_time <- Sys.time()
    formatted_time <- format(current_time, "%d-%m-%Y.%S-%M-%H")
    save(gnet, dnet, params, data, data_norm, 
         data_encode, data_training, data_mask,
         phase2_vars, weight_var, num.normalizing, cat.encoding, 
         batch_size, g_dim, device, torch_cat(list(phase1_ep, phase1_ef), dim = 2), phase2_t, 
         file = paste0("mmer.impute.ccwgangp_", formatted_time, ".RData"))
  }
  
  return (list(imputation = result$imputation, gsample = result$gsample, 
               loss = training_loss,
               step_result = step_result))
}
