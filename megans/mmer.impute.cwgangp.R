pacman::p_load(progress, torch)

init_gan_g <- function(m) {
  if (inherits(m, "nn_linear")) {
    nn_init_kaiming_normal_(m$weight, a = 1.257237, mode = "fan_in", nonlinearity = "leaky_relu")
    nn_init_zeros_(m$bias)
  }
  if (inherits(m, "nn_batch_norm1d")) {
    nn_init_ones_(m$weight)
    nn_init_zeros_(m$bias)
  }
}

init_gan_d <- function(m) {
  if (inherits(m, "nn_linear")) {
    out_features <- tryCatch(m$out_features, error = function(e) NULL)
    
    if (!is.null(out_features) && out_features == 1) {
      nn_init_normal_(m$weight, mean = 0, std = 1e-3)
    } else {
      nn_init_kaiming_normal_(m$weight, a = 0.2, mode = "fan_in",
                              nonlinearity = "leaky_relu")
    }
    if (!is.null(m$bias)) nn_init_constant_(m$bias, 0)
  }
  if (inherits(m, "nn_multihead_attention")) {
    if (!is.null(m$in_proj_weight)) nn_init_xavier_uniform_(m$in_proj_weight, gain = 1.0)
    if (!is.null(m$in_proj_bias))   nn_init_constant_(m$in_proj_bias, 0)
    if (!is.null(m$out_proj)) {
      nn_init_xavier_uniform_(m$out_proj$weight, gain = 1.0)
      if (!is.null(m$out_proj$bias)) nn_init_constant_(m$out_proj$bias, 0)
    }
  }
}

cwgangp_default <- function(batch_size = 500, gamma = 1, lambda = 10, 
                            alpha = 0, beta = 1, at_least_p = 1/2, 
                            lr_g = 2e-4, lr_d = 2e-4, g_betas = c(0.5, 0.9), d_betas = c(0.5, 0.9), 
                            g_weight_decay = 1e-6, d_weight_decay = 1e-6, noise_dim = 64, 
                            g_dim = 256, d_dim = 256, pac = 10, 
                            n_g_layers = 3, n_d_layers = 2, discriminator_steps = 1,
                            tau = 0.2, hard = F, 
                            type_g = "mlp", type_d = "mlp"){
  
  list(batch_size = batch_size, gamma = gamma, lambda = lambda, alpha = alpha, beta = beta, 
       at_least_p = at_least_p, lr_g = lr_g, lr_d = lr_d, g_betas = g_betas, d_betas = d_betas, 
       g_weight_decay = g_weight_decay, d_weight_decay = d_weight_decay, noise_dim = noise_dim,
       g_dim = g_dim, d_dim = d_dim, pac = pac, n_g_layers = n_g_layers, n_d_layers = n_d_layers, 
       discriminator_steps = discriminator_steps, tau = tau, hard = hard,
       type_g = type_g, type_d = type_d)
}

mmer.impute.cwgangp <- function(data, m = 5, 
                                num.normalizing = "mode", cat.encoding = "onehot", 
                                device = "cpu", epochs = 5000, 
                                params = list(), data_info = list(),
                                HT = F, type = "mmer", 
                                save.model = FALSE, save.step = 500){
  params <- do.call("cwgangp_default", params)
  device <- torch_device(device)
  list2env(params, envir = environment())
  list2env(data_info, envir = environment())
  
  conditions_vars <- names(data)[which(!(names(data) %in% c(phase1_vars, phase2_vars)))]
  phase1_rows <- which(is.na(data[, phase2_vars[1]]))
  phase2_rows <- which(!is.na(data[, phase2_vars[1]]))
  
  normalize <- paste("normalize", num.normalizing, sep = ".")
  encode <- paste("encode", cat.encoding, sep = ".")
  
  weights <- as.numeric(as.character(data[, names(data) %in% weight_var]))
  data_original <- data
  if (type == "mmer"){
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
    phase1_vars_mode <- c(phase1_vars, names(data_norm$data)[
      !names(data_norm$data) %in% names(data) &
        names(data_norm$data) %in% paste0(phase1_vars, sep = "_mode")])
    phase2_vars_mode <- c(phase2_vars, names(data_norm$data)[
      !names(data_norm$data) %in% names(data) &
        names(data_norm$data) %in% paste0(phase2_vars, sep = "_mode")])
    conditions_vars_mode <- c(conditions_vars, names(data_norm$data)[
      !names(data_norm$data) %in% names(data) &
        names(data_norm$data) %in% paste0(conditions_vars, sep = "_mode")])
  }
  
  data_encode <- do.call(encode, args = list(
    data = data_norm$data, mode_cat_vars, 
    cat_vars, phase1_vars, phase2_vars
  ))
  nrows <- nrow(data_encode$data)
  ncols <- ncol(data_encode$data)
  #Prepare training tensors
  data_training <- data_encode$data
  
  phase1_vars_encode <- c(phase1_vars[!phase1_vars_mode %in% mode_cat_vars], 
                          unlist(data_encode$new_col_names[phase1_vars_mode]))
  phase2_vars_encode <- c(phase2_vars[!phase2_vars_mode %in% mode_cat_vars], 
                          unlist(data_encode$new_col_names[phase2_vars_mode]))
  conditions_vars_encode <- c(conditions_vars[!conditions_vars_mode %in% mode_cat_vars], 
                              unlist(data_encode$new_col_names[conditions_vars_mode]))
  
  num_inds_p1 <- which(phase1_vars_encode %in% num_vars)
  cat_inds_p1 <- which(phase1_vars_encode %in% unlist(data_encode$new_col_names))
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
  
  data_mask <- torch_tensor(1 - is.na(data_training), device = device)
  conditions_t <- torch_tensor(as.matrix(data_training[, names(data_training) %in% conditions_vars_encode]), 
                               device = device)
  phase2_m <- data_training[, names(data_training) %in% phase2_vars_encode, drop = F]
  phase2_m[is.na(phase2_m)] <- 0 
  phase2_t <- torch_tensor(as.matrix(phase2_m), device = device)
  
  phase1_m <- data_training[, names(data_training) %in% phase1_vars_encode, drop = F]
  phase1_t <- torch_tensor(as.matrix(phase1_m), device = device)
  
  phase1_cats <- phase1_vars[phase1_vars %in% cat_vars]
  phase2_cats <- phase2_vars[phase2_vars %in% cat_vars]
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
    CM_tensors <- lapply(confusmat, function(cm) torch_tensor(cm, dtype = torch_float()))
    names(CM_tensors) <- phase2_cats
    phase1_cats_inds <- match(unlist(data_encode$new_col_names[phase1_vars[phase1_vars %in% cat_vars]]), 
                              names(phase1_m))
    phase2_cats_inds <- match(unlist(data_encode$new_col_names[phase2_vars[phase2_vars %in% cat_vars]]), 
                              names(data_training))
  }
  # for categorical variables, NN outputs real categories, 
  # then times by CM_list to trasnform it to phase1 categories, and then calculate the CE
  tensor_list <- list(data_mask, conditions_t, phase2_t, phase1_t)

  gnet <- do.call(paste("generator", type_g, sep = "."), 
                  args = list(params, 
                              ncols, length(phase2_vars_encode), rate = 0.75))$to(device = device)
  dnet <- do.call(paste("discriminator", type_d, sep = "."), 
                  args = list(params, ncols,  
                              length(phase2_vars_encode)))$to(device = device)
  #gnet$apply(init_gan_g)
  #dnet$apply(init_gan_d)
  
  g_solver <- torch::optim_adam(gnet$parameters, lr = lr_g, 
                                betas = g_betas, weight_decay = g_weight_decay)
  d_solver <- torch::optim_adam(dnet$parameters, lr = lr_d, 
                                betas = d_betas, weight_decay = d_weight_decay)
  
  training_loss <- matrix(0, nrow = epochs, ncol = 2)
  pb <- progress_bar$new(
    format = paste0("Running :what [:bar] :percent eta: :eta | G Loss: :g_loss | D Loss: :d_loss"),
    clear = FALSE, total = epochs, width = 100)
  
  if (save.step > 0){
    step_result <- list()
    p <- 1
  }
  
  gnet_list <- list()
  for (i in 1:epochs){
    gnet$train()
    for (d in 1:discriminator_steps){
      batch <- samplebatches(data, data_training, tensor_list, 
                             phase1_rows, phase2_rows, phase2_vars_encode,
                             data_encode$new_col_names, batch_size, at_least_p = at_least_p, 
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
      fake <- activation_fun(fake, data_encode, phase2_vars_encode, 
                             tau = tau, hard = hard)
      
    
      fake_AC_I <- torch_cat(list(fake[I, , drop = F],
                                  A[I, , drop = F], 
                                  C[I, , drop = F]), dim = 2)
      true_AC_I <- torch_cat(list(X[I, , drop = F], 
                                  A[I, , drop = F], 
                                  C[I, , drop = F]), dim = 2)
      
      fakez_2 <- torch_normal(mean = 0, std = 1, size = c(X$size(1), noise_dim))$to(device = device) 
      fakez_AC_2 <- torch_cat(list(fakez_2, A, C), dim = 2)
      
      fake_2 <- gnet(fakez_AC_2)
      fake_2 <- activation_fun(fake_2, data_encode, phase2_vars_encode, 
                               tau = tau, hard = hard)
      fake_AC_I_2 <- torch_cat(list(fake_2[I, , drop = F],
                                    A[I, , drop = F], 
                                    C[I, , drop = F]), dim = 2)
      x_fake_I_2 <- dnet(fake_AC_I_2)
      x_fake_I <- dnet(fake_AC_I)
      x_true_I <- dnet(true_AC_I)
      if (lambda > 0){
        gp <- gradient_penalty(dnet, true_AC_I, fake_AC_I, params, device = device) + 
          gradient_penalty(dnet, true_AC_I, fake_AC_I_2, params, device = device)
      }else{
        gp <- torch_tensor(0, dtype = fake$dtype, device = device)
      }
      
      d_loss <- -(torch_mean(x_true_I) - torch_mean(x_fake_I)) + 
        -(torch_mean(x_true_I) - torch_mean(x_fake_I_2)) + 
        params$lambda * gp 
      
      d_solver$zero_grad()
      d_loss$backward()
      d_solver$step()
    }
    batch <- samplebatches(data, data_training, tensor_list, 
                           phase1_rows, phase2_rows, phase2_vars_encode,
                           data_encode$new_col_names, batch_size, at_least_p = at_least_p, 
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
    fakeact <- activation_fun(fake, data_encode, phase2_vars_encode, 
                              tau = tau, hard = hard)
    fake_AC <- torch_cat(list(fakeact, A, C), dim = 2)
    x_fake <- dnet(fake_AC)
    
    
    fakez_2 <- torch_normal(mean = 0, std = 1, size = c(X$size(1), noise_dim))$to(device = device) 
    fakez_AC_2 <- torch_cat(list(fakez_2, A, C), dim = 2)
    fake_2 <- gnet(fakez_AC_2)
    fakeact_2 <- activation_fun(fake_2, data_encode, phase2_vars_encode, 
                              tau = tau, hard = hard)
    fake_AC_2 <- torch_cat(list(fakeact_2, A, C), dim = 2)
    x_fake_2 <- dnet(fake_AC_2)
    
    if (length(phase2_cats) > 0){
      projs <- proj_to_p1(fake, X, A, I, CM_tensors, data_encode, phase2_cats, 
                          phase1_cats_inds, phase2_cats_inds)
      fake <- projs[[1]]
      
      projs_2 <- proj_to_p1(fake_2, X, A, I, CM_tensors, data_encode, phase2_cats, 
                          phase1_cats_inds, phase2_cats_inds)
      fake_2 <- projs_2[[1]]
      X <- projs[[2]]
    }

    adv_term <- params$gamma * (-torch_mean(x_fake) - torch_mean(x_fake_2))
    xrecon_loss <- recon_loss(fake, X, I, data_encode, phase2_vars_encode, 
                              phase2_cats, params, num_inds_p2, cat_inds_p2)
    xrecon_loss_2 <- recon_loss(fake_2, X, I, data_encode, phase2_vars_encode, 
                                phase2_cats, params, num_inds_p2, cat_inds_p2)
    div_loss <- nnf_l1_loss(fake_2, fake, reduction = "mean") / 
      nnf_l1_loss(fakez_2, fakez, reduction = "mean")
    div_loss <- torch_relu(1 - div_loss) #(1 / (div_loss + 1e-5))
    g_loss <- adv_term + xrecon_loss + xrecon_loss_2 + div_loss
    
    g_solver$zero_grad()
    g_loss$backward()
    g_solver$step()
    
    training_loss[i, ] <- c(g_loss$item(), d_loss$item())
    pb$tick(tokens = list(
      what = "cWGAN-GP",
      g_loss = sprintf("%.4f", g_loss$item()),
      d_loss = sprintf("%.4f", d_loss$item())
    ))
    Sys.sleep(1 / 100000)
    
    if (save.step > 0){
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
                                 batch_size, device, params, tensor_list,
                                 type)
        step_result[[p]] <- result$gsample[[1]]
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
                           batch_size, device, params, tensor_list,
                           type)
  if (save.model){
    model <- list(gnet = gnet, params = params, 
                  data = data_original, data_norm = data_norm,
                  data_encode = data_encode, data_training = data_training,
                  phase1_vars_encode = phase1_vars_encode, 
                  phase2_vars_encode = phase2_vars_encode, 
                  num_vars = num_vars, num.normalizing = num.normalizing, 
                  cat.encoding = cat.encoding, batch_size = batch_size, device = device, 
                  params = params, tensor_list = tensor_list,
                  type = type, log_shift = log_shift)
    return (list(imputation = result$imputation, 
                 gsample = result$gsample, 
                 loss = training_loss,
                 step_result = step_result,
                 model = model))
  }else{
    return (list(imputation = result$imputation, 
                 gsample = result$gsample, 
                 loss = training_loss,
                 step_result = step_result))
  }
}
