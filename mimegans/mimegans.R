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

cwgangp_default <- function(batch_size = 500, lambda = 10, 
                            alpha = 0.2, beta = 1, at_least_p = 0.5, 
                            lr_g = 2e-4, lr_d = 2e-4, g_betas = c(0.5, 0.9), d_betas = c(0.5, 0.9), 
                            g_weight_decay = 1e-6, d_weight_decay = 1e-6, noise_dim = 128, 
                            g_dim = 256, d_dim = 256, pac = 10, 
                            n_g_layers = 3, n_d_layers = 2, discriminator_steps = 1,
                            tau = 0.2, hard = F, type_g = "mlp", type_d = "mlp",
                            num = "mmer", cat = "projp1"){
  if (type_d == "snmlp" | type_d == "sninfomlp"){
    lambda <- 0
  }
  if (num != "mmer"){
    alpha <- 0
  }
  list(batch_size = batch_size, lambda = lambda, alpha = alpha, beta = beta,
       at_least_p = at_least_p, lr_g = lr_g, lr_d = lr_d, g_betas = g_betas, d_betas = d_betas, 
       g_weight_decay = g_weight_decay, d_weight_decay = d_weight_decay, noise_dim = noise_dim,
       g_dim = g_dim, d_dim = d_dim, pac = pac, n_g_layers = n_g_layers, n_d_layers = n_d_layers, 
       discriminator_steps = discriminator_steps, tau = tau, hard = hard,
       type_g = type_g, type_d = type_d, num = num, cat = cat)
}

mimegans <- function(data, m = 5, 
                     num.normalizing = "mode", cat.encoding = "onehot", 
                     device = "cpu", epochs = 10000, 
                     params = list(), data_info = list(), 
                     save.step = NULL){
  params <- do.call("cwgangp_default", params)
  device <- torch_device(device)
  list2env(params, envir = environment())
  list2env(data_info, envir = environment())

  conditions_vars <- names(data)[which(!(names(data) %in% c(phase1_vars, phase2_vars)))]
  phase1_rows <- which(is.na(data[, phase2_vars[1]]))
  phase2_rows <- which(!is.na(data[, phase2_vars[1]]))
  if (at_least_p == 1){
    params$cat_proj <- F
  }
  
  normalize <- paste("normalize", num.normalizing, sep = ".")
  encode <- paste("encode", cat.encoding, sep = ".")
  
  weights <- as.vector(as.numeric(as.character(data[, names(data) %in% weight_var])))
  phase1_bins <- cat_vars[!(cat_vars %in% phase2_vars)] 
  phase1_bins <- if (length(phase1_bins) > 0) {
    phase1_bins[sapply(phase1_bins, function(col) {
      length(unique(data[phase1_rows, col])) > 1 & length(unique(data[phase2_rows, col])) > 1
    })]
  } else {
    character(0)
  }
  data[[weight_var]] <- NULL
  data_original <- data
  if (params$num == "mmer"){
    data[, match((phase2_vars[phase2_vars %in% num_vars]), names(data))] <-
      data[, match((phase1_vars[phase1_vars %in% num_vars]), names(data))] -
      data[, match((phase2_vars[phase2_vars %in% num_vars]), names(data))]
  }
  
  data_norm <- do.call(normalize, args = list(
    data = data,
    num_vars = num_vars, 
    c(conditions_vars, phase1_vars)
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
  
  data_encode$binary_indices <- lapply(data_encode$binary_indices, function(indices) {
    match(names(data_encode$data)[indices], names(data_training))
  })
  
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
  phase1_cats_inds <- match(unlist(data_encode$new_col_names[phase1_cats]), 
                            names(phase1_m)) 
  phase2_cats_inds <- match(unlist(data_encode$new_col_names[phase2_cats]), 
                            names(data_training))
  cats_p1 <- relist(phase1_cats_inds, skeleton = data_encode$new_col_names[phase1_cats])
  
  nc <- data_encode$new_col_names
  bi <- data_encode$binary_indices
  
  idx_map <- setNames(rep(seq_along(nc), lengths(nc)), unlist(nc, use.names = FALSE))
  bins_by_enc <- function(enc) { i <- idx_map[enc]; i <- i[!is.na(i) & !duplicated(i)]; bi[i] }
  
  allcats   <- bins_by_enc(phase2_vars_encode)
  
  enc_cats  <- unlist(nc[phase2_cats], use.names = FALSE)
  cats_mode <- bins_by_enc(setdiff(phase2_vars_encode, enc_cats))
  
  i_order <- idx_map[phase2_vars_encode]; i_order <- i_order[!is.na(i_order) & !duplicated(i_order)]
  cats_p2  <- bi[i_order[names(bi)[i_order] %in% phase2_cats]]
  
  dimensions <- c(length(num_inds_p2), unlist(lapply(allcats, length)))

  if (length(phase2_cats) > 0){
    ind1 <- match(phase1_cats, names(data_norm$data))
    ind2 <- match(phase2_cats, names(data_norm$data))
    if (params$cat == "projp1"){
      confusmat <- lapply(1:length(ind1), function(i){
        lv <- sort(unique(data_norm$data[, ind1[i]]))
        cm <- prop.table(table(factor(data_norm$data[, ind2[i]], levels = lv),
                               factor(data_norm$data[, ind1[i]], levels = lv)), 1)
        cm[is.na(cm)] <- 0 
        return (cm)
      })
      CM_tensors <- lapply(confusmat, function(cm) torch_tensor(cm, dtype = torch_float(), device = device))
      names(CM_tensors) <- phase2_cats
    }else if (params$cat == "projp2"){
      confusmat <- lapply(1:length(ind1), function(i){
        lv <- sort(unique(data_norm$data[, ind1[i]]))
        cm <- prop.table(table(factor(data_norm$data[, ind1[i]], levels = lv),
                               factor(data_norm$data[, ind2[i]], levels = lv)), 1)
        cm[is.na(cm)] <- 0 
        return (cm)
      })
      CM_tensors <- lapply(confusmat, function(cm) torch_tensor(cm, dtype = torch_float(), device = device))
      names(CM_tensors) <- phase2_cats
    }else{
      CM_tensors <- NULL
    }
  }
  # for categorical variables, NN outputs real categories, 
  # then times by CM_list to trasnform it to phase1 categories, and then calculate the CE
  tensor_list <- list(data_mask, conditions_t, phase2_t, phase1_t)
  
  gnet <- do.call(paste("generator", type_g, sep = "."), 
                  args = list(params, ncols, 
                              length(phase2_vars_encode), rate = 0.7,
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
  for (i in 1:epochs){
    gnet$train()
    for (d in 1:discriminator_steps){
      batch <- sampleBatch(data, tensor_list, phase1_bins,
                           phase1_rows, phase2_rows,
                           batch_size, at_least_p,
                           weights, net = "D")
      A <- batch[[4]]
      X <- batch[[3]]
      C <- batch[[2]]
      M <- batch[[1]]
      
      fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), noise_dim))$to(device = device) 
      fakez_AC <- torch_cat(list(fakez, A, C), dim = 2)
      
      fake <- gnet(fakez_AC)
      if (params$cat == "projp2"){
        fake <- projCat(fake, CM_tensors, cats_p2)
      }
      fake <- activationFun(fake, cats_mode, cats_p2, params)
      fake_AC <- torch_cat(list(fake, A, C), dim = 2)
      true_AC <- torch_cat(list(X, A, C), dim = 2)
      if (params$type_d == "infomlp"){
        x_fake <- dnet(fake_AC)[[1]]
        x_true <- dnet(true_AC)[[1]]
      }else{
        x_fake <- dnet(fake_AC)
        x_true <- dnet(true_AC)
      }
      
      if (lambda > 0){
        gp <- gradientPenalty(dnet, fake_AC, true_AC, params, device = device) 
        d_loss <- -(torch_mean(x_true) - torch_mean(x_fake)) + 
          params$lambda * gp
      }else{
        d_loss <- -(torch_mean(x_true) - torch_mean(x_fake))
      }
      
      d_solver$zero_grad()
      d_loss$backward()
      d_solver$step()
    }
    batch <- sampleBatch(data, tensor_list, phase1_bins,
                         phase1_rows, phase2_rows,
                         batch_size, at_least_p = at_least_p,
                         weights, net = "G")
    A <- batch[[4]]
    X <- batch[[3]]
    C <- batch[[2]]
    M <- batch[[1]]
    I <- M[, 1] == 1
    
    fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), noise_dim))$to(device = device) 
    fakez_AC <- torch_cat(list(fakez, A, C), dim = 2)
    fake <- gnet(fakez_AC)
    if (length(phase2_cats) > 0){
      if (params$cat == "projp1"){
        fake_proj <- projCat(fake, CM_tensors, cats_p2)
      }else if (params$cat == "projp2"){
        fake_proj <- fake
        fake <- projCat(fake, CM_tensors, cats_p2)
      }
    }else{
      fake_proj <- NULL
    }
    fakeact <- activationFun(fake, cats_mode, cats_p2, params)
    fake_AC <- torch_cat(list(fakeact, A, C), dim = 2)
    true_AC <- torch_cat(list(X, A, C), dim = 2)
    if (params$type_d == "infomlp"){
      x_fake <- dnet(fake_AC)[[1]]
      info_fake_I <- dnet(fake_AC[I, ])[[2]]
      info_true_I <- dnet(true_AC[I, ])[[2]]
      info_loss <- infoLoss(info_fake_I, info_true_I)
      adv_term <- -torch_mean(x_fake) + info_loss
    }else{
      x_fake <- dnet(fake_AC)
      adv_term <- -torch_mean(x_fake)
    }
    recon_loss <- reconLoss(fake, X, fake_proj, A, C, I, params, 
                            num_inds_p2, cat_inds_p2, 
                            cats_p1, cats_p2, cats_mode)
    g_loss <- adv_term + recon_loss
    
    g_solver$zero_grad()
    g_loss$backward()
    g_solver$step()
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
                                 batch_size, device, params, CM_tensors,
                                 cats_mode, cats_p1, cats_p2, tensor_list)
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
                           batch_size, device, params, CM_tensors,
                           cats_mode, cats_p1, cats_p2, tensor_list)
  out <- list(imputation = result$imputation,
              gsample = result$gsample,
              loss = training_loss)
  if (exists("step_result", inherits = FALSE) && !is.null(step_result)){
    out$step_result <- step_result
  }
  return (out)
}

