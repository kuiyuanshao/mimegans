pacman::p_load(progress, torch)

cwgangp_default <- function(batch_size = 500, gamma = 1, lambda = 10, 
                            alpha = 0, beta = 1, at_least_p = 1/2, 
                            lr_g = 1e-4, lr_d = 5e-4, g_betas = c(0.5, 0.9), d_betas = c(0.5, 0.9), 
                            g_weight_decay = 1e-7, d_weight_decay = 1e-7, 
                            g_dim = 256, d_dim = 256, pac = 10, 
                            n_g_layers = 1, n_d_layers = 3, discriminator_steps = 1,
                            tau = 0.2, hard = F, 
                            tokenize = T, token_dim = 8, 
                            type_g = "attn", type_d = "mlp"){
  
  list(
    batch_size = batch_size, gamma = gamma, lambda = lambda, alpha = alpha, beta = beta, 
    at_least_p = at_least_p, lr_g = lr_g, lr_d = lr_d, g_betas = g_betas, d_betas = d_betas, 
    g_weight_decay = g_weight_decay, d_weight_decay = d_weight_decay, 
    g_dim = g_dim, d_dim = d_dim, pac = pac, n_g_layers = n_g_layers, n_d_layers = n_d_layers, 
    discriminator_steps = discriminator_steps, tau = tau, hard = hard,
    tokenize = tokenize, token_dim = token_dim,
    type_g = type_g, type_d = type_d
  )
}

mmer.impute.cwgangp <- function(data, m = 5, 
                                num.normalizing = "mode", cat.encoding = "onehot", 
                                device = "cpu",
                                epochs = 2500, 
                                params = list(), data_info = list(),
                                save.model = FALSE, save.step = 500){
  params <- do.call("cwgangp_default", params)
  device <- torch_device(device)
  list2env(params, envir = environment())
  list2env(data_info, envir = environment())
  
  phase2_vars <- names(data)[which(sapply(data, function(x) any(is.na(x))))]
  phase1_vars <- names(data)[which(!(names(data) %in% phase2_vars))]
  phase1_rows <- which(is.na(data[, phase2_vars[1]]))
  phase2_rows <- which(!is.na(data[, phase2_vars[1]]))
  normalize <- paste("normalize", num.normalizing, sep = ".")
  encode <- paste("encode", cat.encoding, sep = ".")
  
  #Weights are removed from the normalization
  data_norm <- do.call(normalize, args = list(
    data = data,
    num_vars = num_vars
  ))
  
  if (num.normalizing == "mode"){
    cat_vars <- c(cat_vars, setdiff(names(data_norm$data), names(data)))
    phase1_vars <- c(phase1_vars, names(data_norm$data)[
      !names(data_norm$data) %in% names(data) &
        names(data_norm$data) %in% paste0(phase1_vars, sep = "_mode")])
    phase2_vars <- c(phase2_vars, names(data_norm$data)[
      !names(data_norm$data) %in% names(data) &
        names(data_norm$data) %in% paste0(phase2_vars, sep = "_mode")])
  }
  
  data_encode <- do.call(encode, args = list(
    data = data_norm$data,
    cat_vars = cat_vars, 
    phase1_vars, type_g
  ))
  
  nrows <- nrow(data_encode$data)
  ncols <- ncol(data_encode$data)
  
  #Encoding creates new variables corresponding to the categorical variables.
  phase1_vars <- c(phase1_vars[!phase1_vars %in% cat_vars], 
                   unlist(data_encode$new_col_names[phase1_vars]))
  phase2_vars <- c(phase2_vars[!phase2_vars %in% cat_vars], 
                   unlist(data_encode$new_col_names[phase2_vars]))
  
  #Prepare training tensors
  data_training <- data_encode$data
  #Reorder the data to Phase2 | Phase1, since the Generator only generates Phase2 data.
  data_training <- data_training[, c(phase2_vars, 
                                     setdiff(names(data_training), phase2_vars))]
  
  binary_indices_reordered <- lapply(data_encode$binary_indices, function(indices) {
    match(names(data_encode$data)[indices], names(data_training))
  })
  data_encode$binary_indices <- binary_indices_reordered
  
  data_mask <- torch_tensor(1 - is.na(data_training), device = device)
  #Phase1 Variables Tensors
  phase1_t <- torch_tensor(as.matrix(data_training[, !names(data_training) %in% phase2_vars]), device = device)
  #Phase2 Variables Tensors
  phase2_t <- data_training[, phase2_vars, drop = F]
  #numeric and binary column indices
  num_inds <- which(names(phase2_t) %in% num_vars)
  cat_inds <- which(names(phase2_t) %in% unlist(data_encode$new_col_names))
  #Replace all NA values with zeros and set to device.
  phase2_t[is.na(phase2_t)] <- 0 
  phase2_t <- torch_tensor(as.matrix(phase2_t), device = device)
  
  if (type_g == "attn"){
    if (tokenize){
      cat_inds_p1 <- (unlist(binary_indices_reordered) - 
                        length(phase2_vars))[(unlist(binary_indices_reordered) - length(phase2_vars)) > 0]
      num_inds_p1 <- which(!(1:(ncols - length(phase2_vars)) %in% cat_inds_p1))
      tokenizer <- Tokenizer(dim(phase1_t)[2], cat_inds_p1, 
                             params$token_dim, unlist(data_encode$n_unique))
      ncols <- params$token_dim * (dim(phase1_t)[2] + 1) + length(phase2_vars)
    }else{
      ncols <- ncols
    }
  }else{
    ncols <- ncols
  }
  tensor_list <- list(data_mask, phase1_t, phase2_t)
  
  gnet <- do.call(paste("generator", type_g, sep = "."), 
                  args = list(n_g_layers, params, 
                              ncols, length(phase2_vars)))$to(device = device)
  dnet <- do.call(paste("discriminator", type_d, sep = "."), 
                  args = list(n_d_layers, params, ncols))$to(device = device)
  
  g_solver <- torch::optim_adam(gnet$parameters, lr = lr_g, 
                                betas = g_betas, weight_decay = g_weight_decay)
  d_solver <- torch::optim_adam(dnet$parameters, lr = lr_d, 
                                betas = d_betas, weight_decay = d_weight_decay)
  
  #nn_utils_clip_grad_norm_(gnet$parameters, max_norm = 10)
  training_loss <- matrix(0, nrow = epochs, ncol = 2)
  pb <- progress_bar$new(
    format = paste0("Running :what [:bar] :percent eta: :eta | G Loss: :g_loss | D Loss: :d_loss"),
    clear = FALSE, total = epochs, width = 100)
  
  if (save.step > 0){
    step_result <- list()
    p <- 1
  } 
  
  #alpha_init <- params$alpha
  for (i in 1:epochs){
    #params$alpha <- 0.1 * alpha_init + (alpha_init - 0.1 * alpha_init) * 
    #  1 / (1 + exp(params$zeta/epochs * (i - 1/2 * epochs)))
    for (d in 1:discriminator_steps){
      batch <- samplebatches(data, data_training, 
                             tensor_list, 
                             phase1_rows, phase2_rows, 
                             phase1_vars, phase2_vars, 
                             num_vars, data_encode$new_col_names, 
                             batch_size, at_least_p = at_least_p)
      X <- batch[[3]]
      C <- batch[[2]]
      M <- batch[[1]]
      I <- M[, 1] == 1
      
      fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), g_dim))$to(device = device)
      
      if (tokenize){
        C <- tokenizer(C[, num_inds_p1, drop = F], 
                       C[, cat_inds_p1, drop = F])
        C <- C$reshape(c(C$size(1), C$size(2) * C$size(3)))
        fakez_C <- torch_cat(list(fakez, C), dim = 2)
      }else{
        fakez_C <- torch_cat(list(fakez, C), dim = 2)
      }
      
      fake <- gnet(fakez_C)
      
      fake_I <- fake[I, ]
      C_I <- C[I, ]
      true_I <- X[I, ]
      
      fake_I <- activation_fun(fake_I, data_encode, phase2_vars, tau = tau, hard = hard)
      
      fake_I_noise <- fake_I + torch_randn_like(fake_I) * 0.05
      true_I_noise <- true_I + torch_randn_like(true_I) * 0.05
      
      fake_C_I <- torch_cat(list(fake_I, C_I), dim = 2)
      true_C_I <- torch_cat(list(true_I, C_I), dim = 2)
      
      d_loss <- d_loss(dnet, true_C_I, fake_C_I, params, device)
      
      d_solver$zero_grad()
      d_loss$backward()
      d_solver$step()
    }
    
    batch <- samplebatches(data, data_training, tensor_list,
                           phase1_rows, phase2_rows, 
                           phase1_vars, phase2_vars, 
                           num_vars, data_encode$new_col_names, 
                           batch_size, at_least_p = at_least_p)
    X <- batch[[3]]
    C <- batch[[2]]
    M <- batch[[1]]
    I <- M[, 1] == 1
    
    fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), g_dim))$to(device = device)
    if (tokenize){
      C <- tokenizer(C[, num_inds_p1, drop = F], 
                     C[, cat_inds_p1, drop = F])
      C <- C$reshape(c(C$size(1), C$size(2) * C$size(3)))
      fakez_C <- torch_cat(list(fakez, C), dim = 2)
    }else{
      fakez_C <- torch_cat(list(fakez, C), dim = 2)
    }
    fake <- gnet(fakez_C)
    
    fake_I <- fake[I, ]
    C_I <- C[I, ]
    true_I <- X[I, ]
    
    fake_I <- activation_fun(fake_I, data_encode, phase2_vars, tau = tau, hard = hard)
    fake_C_I <- torch_cat(list(fake_I, C_I), dim = 2)
    
    y_fake <- dnet(fake_C_I)
    g_loss <- g_loss(y_fake, fake_I, true_I, data_encode, 
                     phase2_vars, params, num_inds, cat_inds)
    
    g_solver$zero_grad()
    g_loss$backward()
    g_solver$step()
    
    training_loss[i, ] <- c(g_loss$item(), d_loss$item())
    pb$tick(tokens = list(
      what = "cWGAN-GP",
      g_loss = sprintf("%.4f", g_loss$item()),
      d_loss = sprintf("%.4f", d_loss$item())
    ))
    Sys.sleep(1 / 10000)
    
    if (save.step > 0){
      if (i %% save.step == 0){
        if (tokenize){
          tokenizer_list <- list(tokenizer = tokenizer, cat_inds_p1 = cat_inds_p1, num_inds_p1 = num_inds_p1)
        }else{
          tokenizer_list <- NULL
        }
        result <- generateImpute(gnet, m = 1, 
                                 data, data_norm, 
                                 data_encode, data_training,
                                 phase1_vars, phase2_vars, num_vars, num.normalizing, cat.encoding, 
                                 batch_size, g_dim, device, params, tensor_list, 
                                 tokenizer_list)#, phase1_rows, phase2_rows, vars_to_pmm)
        step_result[[p]] <- result$gsample
        p <- p + 1
      }
    }
  }
  training_loss <- data.frame(training_loss)
  names(training_loss) <- c("G Loss", "D Loss")
  if (tokenize){
    tokenizer_list <- list(tokenizer = tokenizer, cat_inds_p1 = cat_inds_p1, num_inds_p1 = num_inds_p1)
  }else{
    tokenizer_list <- NULL
  }
  result <- generateImpute(gnet, m = m, 
                           data, data_norm, 
                           data_encode, data_training,
                           phase1_vars, phase2_vars, num_vars, 
                           num.normalizing, cat.encoding, 
                           batch_size, g_dim, device, params, tensor_list, 
                           tokenizer_list)#, phase1_rows, phase2_rows, vars_to_pmm)
  if (save.model){
    current_time <- Sys.time()
    formatted_time <- format(current_time, "%d-%m-%Y.%S-%M-%H")
    save(gnet, dnet, params, data, data_norm, 
         data_encode, data_training, data_mask,
         phase1_vars, phase2_vars, num.normalizing, cat.encoding, 
         batch_size, g_dim, device, phase1_t, phase2_t, 
         file = paste0("mmer.impute.cwgangp_", formatted_time, ".RData"))
  }
  
  return (list(imputation = result$imputation, 
               gsample = result$gsample, 
               loss = training_loss,
               step_result = step_result))
}
