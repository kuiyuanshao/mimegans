pacman::p_load(progress, torch)

cwgangp_default <- function(batch_size = 500, gamma = 1, lambda = 10, 
                            lr_g = 1e-4, lr_d = 1e-4, g_betas = c(0.5, 0.9), d_betas = c(0.5, 0.9), 
                            g_weight_decay = 1e-6, d_weight_decay = 1e-6, 
                            g_dim = c(256, 256), pac = 5, 
                            n_g_layers = 5, n_d_layers = 3, 
                            at_least_p = 0.2, discriminator_steps = 1, scaling = 1,
                            type_g = "mlp", type_d = "mlp"){
  list(
    batch_size = batch_size, gamma = gamma, lambda = lambda,
    lr_g = lr_g, lr_d = lr_d, g_betas = g_betas, d_betas = d_betas, 
    g_weight_decay = g_weight_decay, d_weight_decay = d_weight_decay, 
    g_dim = g_dim, pac = pac, n_g_layers = n_g_layers, n_d_layers = n_d_layers, 
    at_least_p = at_least_p, discriminator_steps = discriminator_steps, scaling = scaling, 
    type_g = type_g, type_d = type_d
  )
}

mmer.impute.cwgangp <- function(data, m = 5, num.normalizing = "mode", cat.encoding = "onehot", device = "cpu",
                                epochs = 10000, params = list(), data_info = list(), save.model = FALSE, save.step = 1000){
  params <- do.call("cwgangp_default", params)
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
    data = data[, -which(names(data) == weight_var)],
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
  
  gnet <- generator(n_g_layers, g_dim, ncols, 
                    length(phase2_vars), unlist(binary_indices_reordered), type_g)$to(device = device)
  dnet <- discriminator(n_d_layers, ncols, 
                        pac, unlist(binary_indices_reordered), type_d)$to(device = device)
  
  
  g_solver <- torch::optim_adam(gnet$parameters, lr = lr_g, betas = g_betas, weight_decay = g_weight_decay)
  d_solver <- torch::optim_adam(dnet$parameters, lr = lr_d, betas = d_betas, weight_decay = d_weight_decay)
  
  training_loss <- matrix(0, nrow = epochs, ncol = 2)
  pb <- progress_bar$new(
    format = paste0("Running :what [:bar] :percent eta: :eta | G Loss: :g_loss | D Loss: :d_loss"),
    clear = FALSE, total = epochs, width = 200)
  
  if (save.step > 0){
    step_result <- list()
    p <- 1
  } 
  
  for (i in 1:epochs){
    d_loss_t <- 0
    
    for (d in 1:discriminator_steps){
      batch <- samplebatches(data, data_training, list(data_mask, phase1_t, phase2_t), 
                             phase1_rows, phase2_rows, 
                             phase1_vars, phase2_vars, 
                             num_vars, data_encode$new_col_names, weight_var, 
                             batch_size, at_least_p = at_least_p)
      X <- batch[[3]]
      C <- batch[[2]]
      M <- batch[[1]]
      I <- M[, 1] == 1
      
      fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), g_dim[1]))$to(device = device)
      
      fakez_C <- torch_cat(list(fakez, C), dim = 2)
      fake <- gnet(fakez_C)
      
      fake_I <- fake[I, ]
      C_I <- C[I, ]
      true_I <- X[I, ]
      
      fake_I <- activation_fun(fake_I, data_encode, phase2_vars)
      
      fake_C_I <- torch_cat(list(fake_I, C_I), dim = 2)
      true_C_I <- torch_cat(list(true_I, C_I), dim = 2)
      
      y_fake <- dnet(fake_C_I)
      y_true <- dnet(true_C_I)
      gradient_penalty<- gradient_penalty(dnet, true_C_I, fake_C_I, pac = pac, device = device)
      
      d_loss <- -(torch_mean(y_true) - torch_mean(y_fake)) 
      d_loss_gp <- d_loss + lambda * gradient_penalty
      
      d_solver$zero_grad()
      d_loss_gp$backward()
      d_solver$step()
    }
    
    batch <- samplebatches(data, data_training, list(data_mask, phase1_t, phase2_t),
                           phase1_rows, phase2_rows, 
                           phase1_vars, phase2_vars, 
                           num_vars, data_encode$new_col_names, weight_var, 
                           batch_size, at_least_p = at_least_p)
    X <- batch[[3]]
    C <- batch[[2]]
    M <- batch[[1]]
    I <- M[, 1] == 1
    
    fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), g_dim[1]))$to(device = device)
    fakez_C <- torch_cat(list(fakez, C), dim = 2)
    fake <- gnet(fakez_C)
    
    fake_I <- fake[I, ]
    C_I <- C[I, ]
    true_I <- X[I, ]
    
    fake_I_act <- activation_fun(fake_I, data_encode, phase2_vars)
    fake_act_C_I <- torch_cat(list(fake_I_act, C_I), dim = 2)
    
    y_fake <- dnet(fake_act_C_I)
    g_loss <- gamma * -torch_mean(y_fake) 
    
    g_solver$zero_grad()
    g_loss$backward()
    g_solver$step()
    
    training_loss[i, ] <- c(g_loss$item(), d_loss_gp$item())
    pb$tick(tokens = list(
      what = "cWGAN-GP",
      g_loss = sprintf("%.4f", g_loss$item()),
      d_loss = sprintf("%.4f", d_loss_gp$item())
    ))
    Sys.sleep(1 / 10000)
    
    if (save.step > 0){
      if (i %% save.step == 0){
        result <- generateImpute(gnet, m = 1, 
                                 data, data_norm, 
                                 data_encode, data_training, data_mask,
                                 phase2_vars, num_vars, weight_var, num.normalizing, cat.encoding, 
                                 batch_size, g_dim, device, phase1_t, phase2_t)
        step_result[[p]] <- result$gsample
        p <- p + 1
      }
    }
  }
  training_loss <- data.frame(training_loss)
  names(training_loss) <- c("G Loss", "D Loss")
  result <- generateImpute(gnet, m = m, 
                           data, data_norm, 
                           data_encode, data_training, data_mask,
                           phase2_vars, num_vars, weight_var, num.normalizing, cat.encoding, 
                           batch_size, g_dim, device, phase1_t, phase2_t)
  if (save.model){
    current_time <- Sys.time()
    formatted_time <- format(current_time, "%d-%m-%Y.%S-%M-%H")
    save(gnet, dnet, params, data, data_norm, 
         data_encode, data_training, data_mask,
         phase2_vars, weight_var, num.normalizing, cat.encoding, 
         batch_size, g_dim, device, phase1_t, phase2_t, file = paste0("mmer.impute.cwgangp_", formatted_time, ".RData"))
  }
  
  return (list(imputation = result$imputation, gsample = result$gsample, 
               loss = training_loss,
               step_result = step_result))
}
