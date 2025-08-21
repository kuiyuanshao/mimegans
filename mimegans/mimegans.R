pacman::p_load(progress, torch)

init_weights_generator <- function(m, nphase2) {
  if (inherits(m, "nn_linear") && m$out_features != nphase2) {
    # Residual linears (ELU next): He/Kaiming (normal or uniform both fine)
    nn_init_kaiming_normal_(m$weight, mode = "fan_in", nonlinearity = "relu")
    if (!is.null(m$bias)) nn_init_constant_(m$bias, 0)
  } else if (inherits(m, "nn_batch_norm1d")) {
    nn_init_constant_(m$weight, 1)
    nn_init_constant_(m$bias,   0)
  } else if (inherits(m, "nn_linear") && m$out_features == nphase2){
    nn_init_xavier_uniform_(m$weight, gain = 0.2)
    if (!is.null(m$bias)) nn_init_constant_(m$bias, 0)
  }
}

init_weights_discriminator <- function(m) {
  if (inherits(m, "nn_linear") && m$out_features != 1) {
    # For LeakyReLU(0.2) stacks
    
    nn_init_kaiming_normal_(m$weight, mode = "fan_in",
                            nonlinearity = "leaky_relu", a = 0.2)
    if (!is.null(m$bias)) nn_init_constant_(m$bias, 0)
  }else if (inherits(m, "nn_linear") && m$out_features == 1){
    nn_init_xavier_uniform_(m$weight, gain = 0.02)
    if (!is.null(m$bias)) nn_init_constant_(m$bias, 0)
  }
  
}

cwgangp_default <- function(batch_size = 500, lambda = 0, 
                            alpha = 0, beta = 0.5, beta_ce = 1, at_least_p = 0.2, 
                            lr_g = 2e-4, lr_d = 2e-4, g_betas = c(0.5, 0.9), d_betas = c(0.5, 0.9), 
                            g_weight_decay = 1e-6, d_weight_decay = 1e-6, noise_dim = 128, 
                            g_dim = 256, d_dim = 256, pac = 10, 
                            n_g_layers = 3, n_d_layers = 2, discriminator_steps = 1,
                            tau = 0.2, hard = F, type_g = "mlp", type_d = "mlp",
                            mmer = T, cat_proj = T, autoscale = T){
  
  list(batch_size = batch_size, lambda = lambda, alpha = alpha, beta = beta, beta_ce = beta_ce,
       at_least_p = at_least_p, lr_g = lr_g, lr_d = lr_d, g_betas = g_betas, d_betas = d_betas, 
       g_weight_decay = g_weight_decay, d_weight_decay = d_weight_decay, noise_dim = noise_dim,
       g_dim = g_dim, d_dim = d_dim, pac = pac, n_g_layers = n_g_layers, n_d_layers = n_d_layers, 
       discriminator_steps = discriminator_steps, tau = tau, hard = hard,
       type_g = type_g, type_d = type_d, mmer = mmer, cat_proj = cat_proj, autoscale = autoscale)
}

mimegans <- function(data, m = 5, 
                     num.normalizing = "mode", cat.encoding = "onehot", 
                     device = "cpu", epochs = 250, 
                     params = list(), data_info = list(), 
                     save.step = NULL){
  params <- do.call("cwgangp_default", params)
  device <- torch_device(device)
  list2env(params, envir = environment())
  list2env(data_info, envir = environment())
  
  conditions_vars <- names(data)[which(!(names(data) %in% c(phase1_vars, phase2_vars)))]
  phase1_rows <- which(is.na(data[, phase2_vars[1]]))
  phase2_rows <- which(!is.na(data[, phase2_vars[1]]))
  
  normalize <- paste("normalize", num.normalizing, sep = ".")
  encode <- paste("encode", cat.encoding, sep = ".")
  
  weights <- as.vector(as.numeric(as.character(data[, names(data) %in% weight_var])))
  phase1_bins <- cat_vars[!(cat_vars %in% phase2_vars)] 
  phase1_bins <- if (length(phase1_bins) > 0) {
    phase1_bins[sapply(phase1_bins, function(col) {
      length(unique(data[phase1_rows, col])) > 1 & length(unique(data[phase1_rows, col])) > 1
    })]
  } else {
    character(0)
  }
  data[[weight_var]] <- NULL
  data_original <- data
  if (params$mmer){
    data[, match((phase2_vars[phase2_vars %in% num_vars]), names(data))] <-
      data[, match((phase1_vars[phase1_vars %in% num_vars]), names(data))] -
      data[, match((phase2_vars[phase2_vars %in% num_vars]), names(data))]
  }
  
  data_norm <- do.call(normalize, args = list(
    data = data,
    num_vars = num_vars, 
    c(conditions_vars, phase1_vars), phase2_vars
  ))
  
  if (num.normalizing == "mode"){
    mode_cat_vars <- c(cat_vars, setdiff(names(data_norm$data), names(data)))
    phase2_vars_mode <- c(phase2_vars, names(data_norm$data)[
      !names(data_norm$data) %in% names(data) &
        names(data_norm$data) %in% paste0(phase2_vars, sep = "_mode")])
  }
  
  data_encode <- do.call(encode, args = list(
    data = data_norm$data, mode_cat_vars, 
    cat_vars, phase1_vars, phase2_vars
  ))
  nrows <- nrow(data_encode$data)
  ncols <- ncol(data_encode$data)
  #Prepare training tensors
  data_training <- data_encode$data
  
  phase1_vars_encode <- c(phase1_vars[!phase1_vars %in% mode_cat_vars], 
                          unlist(data_encode$new_col_names[phase1_vars]))
  phase2_vars_encode <- c(phase2_vars[!phase2_vars_mode %in% mode_cat_vars], 
                          unlist(data_encode$new_col_names[phase2_vars_mode]))
  conditions_vars_encode <- c(conditions_vars[!conditions_vars %in% mode_cat_vars], 
                              unlist(data_encode$new_col_names[conditions_vars]))
  
  num_inds_p2 <- which(phase2_vars_encode %in% num_vars) # all numeric inds
  cat_inds_p2 <- which(phase2_vars_encode %in% unlist(data_encode$new_col_names)) # all one hot inds, involving modes
  
  new_order <- c(phase2_vars_encode[num_inds_p2], 
                 phase2_vars_encode[cat_inds_p2],
                 setdiff(names(data_training), phase2_vars_encode))
  data_training <- data_training[, new_order]
  
  binary_indices_reordered <- lapply(data_encode$binary_indices, function(indices) {
    match(names(data_encode$data)[indices], names(data_training))
  })
  data_encode$binary_indices <- binary_indices_reordered
  
  data_mask <- torch_tensor(1 - is.na(data_training), dtype = torch_long(), device = device)
  conditions_t <- torch_tensor(as.matrix(data_training[, names(data_training) %in% conditions_vars_encode]), 
                               device = device)
  phase2_m <- data_training[, names(data_training) %in% phase2_vars_encode, drop = F]
  phase2_m[is.na(phase2_m)] <- 0 
  phase2_t <- torch_tensor(as.matrix(phase2_m), device = device)
  
  phase1_m <- data_training[, names(data_training) %in% phase1_vars_encode, drop = F]
  phase1_t <- torch_tensor(as.matrix(phase1_m), device = device)
  
  phase1_cats <- phase1_vars[phase1_vars %in% cat_vars]
  phase2_cats <- phase2_vars[phase2_vars %in% cat_vars]
  if (params$cat_proj){
    if (length(phase2_cats) > 0){
      ind1 <- match(phase1_cats, names(data_norm$data))
      ind2 <- match(phase2_cats, names(data_norm$data))
      confusmat <- lapply(1:length(ind1), function(i){
        lv <- sort(unique(data_norm$data[, ind1[i]]))
        cm <- prop.table(table(factor(data_norm$data[, ind2[i]], levels = lv),
                               factor(data_norm$data[, ind1[i]], levels = lv)), 1)
        cm[is.na(cm)] <- 0 
        return (cm)
      })
      CM_tensors <- lapply(confusmat, function(cm) torch_tensor(cm, dtype = torch_float(), device = device))
      names(CM_tensors) <- phase2_cats
      phase1_cats_inds <- match(unlist(data_encode$new_col_names[phase1_vars[phase1_vars %in% cat_vars]]), 
                                names(phase1_m)) #
      phase2_cats_inds <- match(unlist(data_encode$new_col_names[phase2_vars[phase2_vars %in% cat_vars]]), 
                                names(data_training)) #
      cats_p1 <- relist(phase1_cats_inds, skeleton = data_encode$new_col_names[phase1_vars[phase1_vars %in% cat_vars]])
    }
  }else{
    cats_p1 <- NULL
  }
  # for categorical variables, NN outputs real categories, 
  # then times by CM_list to trasnform it to phase1 categories, and then calculate the CE
  tensor_list <- list(data_mask, conditions_t, phase2_t, phase1_t)
  
  cats_p2 <- data_encode$binary_indices[which(sapply(names(data_encode$new_col_names), function(col_names) {
    any(col_names %in% phase2_cats)
  }))]
  cats_mode <- data_encode$binary_indices[which(sapply(data_encode$new_col_names, function(col_names) {
    any(col_names %in% phase2_vars_encode[!(phase2_vars_encode %in% unlist(data_encode$new_col_names[phase2_cats]))])
  }))]

  allcats <- data_encode$binary_indices[which(sapply(data_encode$new_col_names, function(col_names) {
    any(col_names %in% phase2_vars_encode)
  }))]
  allnums <- (1:length(phase2_vars_encode))[!(1:length(phase2_vars_encode) %in% unlist(allcats))]
  
  dimensions <- c(length(num_inds_p2), sapply(allcats, length))
  gnet <- do.call(paste("generator", type_g, sep = "."), 
                  args = list(params, 
                              ncols, length(phase2_vars_encode), rate = 0.7,
                              dimensions))$to(device = device)
  dnet <- do.call(paste("discriminator", type_d, sep = "."), 
                  args = list(params, ncols,  
                              length(phase2_vars_encode)))$to(device = device)
  # gnet$apply(function(m) init_weights_generator(m, nphase2 = length(phase2_vars_encode)))
  # dnet$apply(init_weights_discriminator)
  
  g_solver <- torch::optim_adam(gnet$parameters, lr = lr_g, 
                                betas = g_betas, weight_decay = g_weight_decay)
  d_solver <- torch::optim_adam(dnet$parameters, lr = lr_d, 
                                betas = d_betas, weight_decay = d_weight_decay)
  
  training_loss <- matrix(0, nrow = epochs, ncol = 2)
  pb <- progress_bar$new(
    format = paste0("Running :what [:bar] :percent eta: :eta | G Loss: :g_loss | D Loss: :d_loss | Recon: :recon_loss"),
    clear = FALSE, total = epochs, width = 100)
  
  if (!is.null(save.step)){
    step_result <- list()
    p <- 1
  }
  if (autoscale){
    params$beta <- 1
  }
  steps_per_epoch <- nrows %/% batch_size
  for (i in 1:epochs){
    gnet$train()
    for (step in 1:steps_per_epoch){
      for (d in 1:discriminator_steps){
        batch <- sampleBatch(data, tensor_list, phase1_bins,
                             phase1_rows, phase2_rows,
                             batch_size, at_least_p = at_least_p,
                             weights)
        W <- batch[[5]] 
        A <- batch[[4]]
        X <- batch[[3]]
        C <- batch[[2]]
        M <- batch[[1]]
        I <- M[, 1] == 1
        
        fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), noise_dim))$to(device = device) 
        fakez_AC <- torch_cat(list(fakez, A, C), dim = 2)
        
        fake <- gnet(fakez_AC)
        fake <- activationFun(fake, allnums, allcats, 
                              tau = tau, hard = hard)
        
        
        fake_AC_I <- torch_cat(list(fake[I, , drop = F],
                                    A[I, , drop = F], 
                                    C[I, , drop = F]), dim = 2)
        true_AC_I <- torch_cat(list(X[I, , drop = F], 
                                    A[I, , drop = F], 
                                    C[I, , drop = F]), dim = 2)
        x_fake_I <- dnet(fake_AC_I)
        x_true_I <- dnet(true_AC_I)
        
        if (lambda > 0){
          gp <- gradientPenalty(dnet, true_AC_I, fake_AC_I, params, device = device) 
          d_loss <- -(torch_mean(x_true_I) - torch_mean(x_fake_I)) + 
            params$lambda * gp
        }else{
          d_loss <- -(torch_mean(x_true_I) - torch_mean(x_fake_I))
        }
        
        d_solver$zero_grad()
        d_loss$backward()
        d_solver$step()
      }
      batch <- sampleBatch(data, tensor_list, phase1_bins,
                           phase1_rows, phase2_rows,
                           batch_size, at_least_p = at_least_p,
                           weights)
      W <- batch[[5]]
      A <- batch[[4]]
      X <- batch[[3]]
      C <- batch[[2]]
      M <- batch[[1]]
      I <- M[, 1] == 1
      
      fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), noise_dim))$to(device = device) 
      fakez_AC <- torch_cat(list(fakez, A, C), dim = 2)
      fake <- gnet(fakez_AC)
      fakeact <- activationFun(fake, allnums, allcats, 
                               tau = tau, hard = hard)
      fake_AC <- torch_cat(list(fakeact, A, C), dim = 2)
      x_fake <- dnet(fake_AC)
      
      if (params$cat_proj){
        if (length(phase2_cats) > 0){
          fake_proj <- projP1(fakeact, CM_tensors, cats_p2)
        }
      }else{
        fake_proj <- NULL
      }
      adv_term <- -torch_mean(x_fake) 
      recon_loss <- reconLoss(fake, X, fake_proj, A, I, params, 
                              num_inds_p2, cat_inds_p2, 
                              cats_p1, cats_p2, cats_mode)
      
      if (autoscale){
        adv_grads <- autograd_grad(
          outputs = adv_term, 
          inputs = gnet$parameters,
          retain_graph = TRUE
        )[[1]]
        
        recon_grads <- autograd_grad(
          outputs = recon_loss,
          inputs = gnet$parameters,
          retain_graph = TRUE
        )[[1]]
        
        g_adv <- adv_grads$pow(2)$sum()$sqrt()
        g_recon <- recon_grads$pow(2)$sum()$sqrt()
        
        with_no_grad({
          beta_ce <- beta_ce * torch_exp(0.5 * (torch_log(g_adv + 1e-12) -
                                                      torch_log(g_recon + 1e-12)))
          beta_ce <- torch_clamp(beta_ce, 1e-6, 10.0)
        })
        recon_loss <- beta_ce * recon_loss
      }
      g_loss <- adv_term + recon_loss
      
      g_solver$zero_grad()
      g_loss$backward()
      g_solver$step()
    }
    training_loss[i, ] <- c(g_loss$item(), d_loss$item())
    
    pb$tick(tokens = list(
      what = "cWGAN-GP",
      g_loss = sprintf("%.4f", adv_term$item()),
      d_loss = sprintf("%.4f", d_loss$item()),
      recon_loss = sprintf("%.4f", recon_loss$item())
    ))
    Sys.sleep(1 / 100000)

    if (!is.null(save.step)){
      if (i %% save.step == 0){
        gnet$eval()
        for (modu in gnet$modules) {
          if (inherits(modu, "nn_dropout")) {
            modu$train(TRUE)
          }
        }
        result <- generateImpute(gnet, m = 5, 
                                 data_original, data_info, data_norm, 
                                 data_encode, data_training,
                                 phase1_vars_encode, phase2_vars_encode, 
                                 num.normalizing, cat.encoding, 
                                 batch_size, device, params, 
                                 allnums, allcats, tensor_list)
        step_result[[p]] <- result
        p <- p + 1
        
      }
    }
  }
  training_loss <- data.frame(training_loss)
  names(training_loss) <- c("G Loss", "D Loss")
  
  gnet$eval()
  for (modu in gnet$modules) {
    if (inherits(modu, "nn_dropout")) {
      modu$train(TRUE)
    }
  }
  result <- generateImpute(gnet, m = m, 
                           data_original, data_info, data_norm, 
                           data_encode, data_training,
                           phase1_vars_encode, phase2_vars_encode,
                           num.normalizing, cat.encoding, 
                           batch_size, device, params, 
                           allnums, allcats, tensor_list)
  out <- list(imputation = result$imputation,
              gsample = result$gsample,
              loss = training_loss)
  if (exists("step_result", inherits = FALSE) && !is.null(step_result)){
    out$step_result <- step_result
  }
  return (out)
}
