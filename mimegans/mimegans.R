pacman::p_load(progress, torch)

cwgangp_default <- function(batch_size = 500, lambda = 10, 
                            alpha = 0, beta = 1, at_least_p = 0.5, g_dropout = 0.5,
                            lr_g = 2e-4, lr_d = 2e-4, g_betas = c(0.5, 0.9), d_betas = c(0.5, 0.9), 
                            g_weight_decay = 1e-6, d_weight_decay = 1e-6, noise_dim = 128, 
                            g_dim = c(256, 256), d_dim = c(256, 256), pac = 5, discriminator_steps = 1,
                            tau = 0.2, hard = F, type_g = "mlp", type_d = "mlp",
                            num = "mmer", cat = "projp1", component = "none", 
                            info_loss = T, balancebatch = T, unconditional = F){
  batch_size <- pac * round(batch_size / pac)
  if (component == "gen_loss"){
    unconditional <- F
    balancebatch <- T
  }
  if (info_loss){
    req <- ceiling(at_least_p * batch_size)
    n <- ceiling(req / pac) * pac
    if (n > batch_size) n <- floor(batch_size / pac) * pac
    at_least_p <- n / batch_size
  }
  list(batch_size = batch_size, lambda = lambda, alpha = alpha, beta = beta,
       at_least_p = at_least_p, g_dropout = g_dropout, lr_g = lr_g, lr_d = lr_d, g_betas = g_betas, d_betas = d_betas, 
       g_weight_decay = g_weight_decay, d_weight_decay = d_weight_decay, noise_dim = noise_dim,
       g_dim = g_dim, d_dim = d_dim, pac = pac, discriminator_steps = discriminator_steps, tau = tau, hard = hard,
       type_g = type_g, type_d = type_d, num = num, cat = cat, component = component, 
       info_loss = info_loss, balancebatch = balancebatch, unconditional = unconditional)
}

mimegans <- function(data, m = 5, 
                     num.normalizing = "mode", cat.encoding = "onehot", 
                     device = "cpu", epochs = 10000, 
                     params = list(), data_info = list(), 
                     save.step = NULL){
  params <- do.call("cwgangp_default", params)
  device <- torch_device(device)
  if (params$at_least_p == 1){
    params$cat <- "general"
  }
  conditions_vars <- names(data)[which(!(names(data) %in% c(data_info$phase1_vars, data_info$phase2_vars)))]
  phase1_rows <- which(is.na(data[, data_info$phase2_vars[1]]))
  phase2_rows <- which(!is.na(data[, data_info$phase2_vars[1]]))
  normalize <- paste("normalize", num.normalizing, sep = ".")
  encode <- paste("encode", cat.encoding, sep = ".")
  
  weights <- as.vector(as.numeric(as.character(data[, names(data) %in% data_info$weight_var])))
  data_original <- data
  if (params$num == "mmer"){
    p2n <- intersect(data_info$phase2_vars, data_info$num_vars)
    p1n <- intersect(data_info$phase1_vars, data_info$num_vars)
    data[p2n] <- Map(function(p1, p2) data[[p1]] - data[[p2]], p1n, p2n)
  }
  
  data_norm <- do.call(normalize, args = list(
    data = data,
    num_vars = data_info$num_vars, 
    c(conditions_vars, data_info$phase1_vars)
  ))
  norm_data <- data_norm$data
  mode_cat_vars <- union(data_info$cat_vars, setdiff(names(norm_data), names(data)))
  phase2_vars_mode <- union(data_info$phase2_vars,
                            names(norm_data)[names(norm_data) %in% 
                                               paste0(data_info$phase2_vars, "_mode")])
  
  data_encode <- do.call(encode, args = list(
    data = norm_data, mode_cat_vars, 
    data_info$cat_vars, data_info$phase1_vars, data_info$phase2_vars
  ))
  nrows <- nrow(data_encode$data)
  ncols <- ncol(data_encode$data)
  data_training <- data_encode$data
  
  phase1_vars_encode <- c(setdiff(data_info$phase1_vars, mode_cat_vars),
                          unlist(data_encode$new_col_names[data_info$phase1_vars]))
  phase2_vars_encode <- c(setdiff(phase2_vars_mode, mode_cat_vars),
                          unlist(data_encode$new_col_names[phase2_vars_mode]))
  conditions_vars_encode <- c(setdiff(conditions_vars, mode_cat_vars),
                              unlist(data_encode$new_col_names[conditions_vars]))
  
  num_inds_p1 <- which(phase1_vars_encode %in% data_info$num_vars)
  cat_inds_p1 <- which(phase1_vars_encode %in% unlist(data_encode$new_col_names))
  num_inds_p2 <- which(phase2_vars_encode %in% data_info$num_vars)
  cat_inds_p2 <- which(phase2_vars_encode %in% unlist(data_encode$new_col_names))
  
  new_order <- c(phase2_vars_encode[num_inds_p2], 
                 phase2_vars_encode[cat_inds_p2],
                 phase1_vars_encode[num_inds_p1], 
                 phase1_vars_encode[cat_inds_p1],
                 setdiff(names(data_training), 
                         c(phase2_vars_encode, phase1_vars_encode)))
  data_training <- data_training[, new_order]
  
  data_encode$binary_indices <- 
    lapply(data_encode$binary_indices, 
           function(idx) match(names(data_encode$data)[idx], names(data_training)))
  
  data_mask <- torch_tensor(1 - is.na(data_training), dtype = torch_long(), device = device)
  conditions_t <- torch_tensor(as.matrix(data_training[, conditions_vars_encode, drop = F]), 
                               device = device)
  phase2_m <- data_training[, phase2_vars_encode, drop = F]; phase2_m[is.na(phase2_m)] <- 0 
  phase2_t <- torch_tensor(as.matrix(phase2_m), device = device)
  
  phase1_m <- data_training[, phase1_vars_encode, drop = F]
  phase1_t <- torch_tensor(as.matrix(phase1_m), device = device)
  
  tensor_list <- list(data_mask, conditions_t, phase2_t, phase1_t)
  
  p1set <- initset(data_training, phase1_rows, 
                   phase1_vars_encode, phase2_vars_encode, conditions_vars_encode)
  p2set <- initset(data_training, phase2_rows, 
                   phase1_vars_encode, phase2_vars_encode, conditions_vars_encode)
  
  p2b_t <- as.integer(params$batch_size * params$at_least_p)
  p1b_t <- params$batch_size - p2b_t
  if (params$balancebatch){
    phase1_bins <- data_info$cat_vars[!(data_info$cat_vars %in% data_info$phase2_vars)] 
    phase1_bins <- if (length(phase1_bins) > 0) {
      phase1_bins[sapply(phase1_bins, function(col) {
        length(unique(data_original[phase1_rows, col])) > 1 & length(unique(data_original[phase2_rows, col])) > 1
      })]
    } else {
      character(0)
    }
    bins_l <- length(phase1_bins)
    phase1_bins_indices <- data_encode$binary_indices[phase1_bins]
    
    p1sampler <- BalancedSampler(data_original[phase1_rows,], phase1_bins, p1b_t, epochs)
    p2sampler <- BalancedSampler(data_original[phase2_rows,], phase1_bins, p2b_t, epochs)
    p1loader <- dataloader(p1set, sampler = p1sampler, pin_memory = T,
                           collate_fn = function(bl) bl[[1]])
    p2loader <- dataloader(p2set, sampler = p2sampler, pin_memory = T,
                           collate_fn = function(bl) bl[[1]])
    Dloader <- dataloader(p2set, 
                          sampler = BalancedSampler(data_original[phase2_rows,], 
                                                    phase1_bins, params$batch_size, epochs), pin_memory = T,
                          collate_fn = function(bl) bl[[1]])
  }else{
    p1sampler <- SRSSampler(length(phase1_rows), p1b_t, epochs)
    p2sampler <- SRSSampler(length(phase2_rows), p2b_t, epochs)
    p1loader <- dataloader(p1set, sampler = p1sampler, pin_memory = T,
                           collate_fn = function(bl) bl[[1]])
    p2loader <- dataloader(p2set, sampler = p2sampler, pin_memory = T,
                           collate_fn = function(bl) bl[[1]])
    Dloader <- dataloader(p2set, 
                          sampler = SRSSampler(length(phase2_rows), params$batch_size, epochs),
                          pin_memory = T, collate_fn = function(bl) bl[[1]])
  }
  
  phase1_cats <- intersect(data_info$phase1_vars, data_info$cat_vars)
  phase2_cats <- intersect(data_info$phase2_vars, data_info$cat_vars)
  
  cats_p1 <- relist(match(unlist(data_encode$new_col_names[phase1_cats]), 
                          names(phase1_m)), 
                    skeleton = data_encode$new_col_names[phase1_cats])
  
  nc <- data_encode$new_col_names
  bi <- data_encode$binary_indices
  
  idx_map <- setNames(rep(seq_along(data_encode$new_col_names), 
                          lengths(data_encode$new_col_names)), 
                      unlist(data_encode$new_col_names, use.names = FALSE))
  bins_by_enc <- function(enc) { 
    i <- idx_map[enc]
    i <- i[!is.na(i) & !duplicated(i)]
    data_encode$binary_indices[i] 
  }
  
  allcats <- data_encode$binary_indices
  allcats_p2 <- bins_by_enc(phase2_vars_encode)
  cats_mode <- bins_by_enc(setdiff(phase2_vars_encode, 
                                   unlist(data_encode$new_col_names[phase2_cats], use.names = FALSE)))
  i_order <- idx_map[phase2_vars_encode]; i_order <- i_order[!is.na(i_order) & !duplicated(i_order)]
  cats_p2 <- data_encode$binary_indices[i_order[names(data_encode$binary_indices)[i_order] %in% phase2_cats]]
  
  if (params$component == "gen_loss"){
    mask_A <- torch_tensor(matrix(1, nrow = params$batch_size, 
                                  ncol = length(phase1_vars_encode)), dtype = torch_long(), device = device)
    mask_A[, unlist(cats_p1)] <- 0
    mask_C <- torch_tensor(matrix(1, nrow = params$batch_size, 
                                  ncol = length(conditions_vars_encode)), dtype = torch_long(), device = device)
    mask_C[, which(conditions_vars_encode %in% unlist(data_encode$new_col_names))] <- 0
    m_A_list <- list()
    m_C_list <- list()
    for (i in 1:length(phase1_bins)){
      Aind <- which(phase1_vars_encode %in% data_encode$new_col_names[[phase1_bins[i]]])
      Cind <- which(conditions_vars_encode %in% data_encode$new_col_names[[phase1_bins[i]]])
      curr_M <- mask_A
      curr_C <- mask_C
      if (length(Aind) >= 1){
        curr_M[, Aind] <- 1
      }else{
        curr_C[, Cind] <- 1
      }
      m_A_list[[i]] <- curr_M
      m_C_list[[i]] <- curr_C
    }
  }
  
  if (length(phase2_cats) > 0){
    if (params$cat == "projp1"){
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
    }
  }
  
  gnet <- do.call(paste("generator", params$type_g, sep = "."), 
                  args = list(params, ncols, 
                              length(phase2_vars_encode), 
                              length(phase1_vars_encode)))$to(device = device)
  if (params$component == "gen_loss"){
    dnet <- do.call(paste("discriminator", params$type_d, sep = "."), 
                    args = list(params, ncols + length(phase1_vars_encode) + 
                                  length(conditions_vars_encode)))$to(device = device)
  }else if (params$component == "match_p1"){
    fnet <- do.call(paste("forwardA", params$type_g, sep = "."), 
                    args = list(params, ncols, 
                                length(phase1_vars_encode)))$to(device = device)
    dnet <- do.call(paste("discriminator", params$type_d, sep = "."), 
                    args = list(params, ncols))$to(device = device)
    f_solver <- torch::optim_adam(fnet$parameters, lr = params$lr_g * 2, 
                                  betas = params$g_betas, weight_decay = params$g_weight_decay * 100)
  }else{
    dnet <- do.call(paste("discriminator", params$type_d, sep = "."), 
                    args = list(params, ncols))$to(device = device)
  }
  d_solver <- torch::optim_adam(dnet$parameters, lr = params$lr_d, 
                                betas = params$d_betas, weight_decay = params$d_weight_decay)
  g_solver <- torch::optim_adam(gnet$parameters, lr = params$lr_g, 
                                betas = params$g_betas, weight_decay = params$g_weight_decay)
  
  training_loss <- matrix(0, nrow = epochs, ncol = 2)
  
  if (params$info_loss){
    pb <- progress_bar$new(
      format = paste0("Running [:bar] :percent eta: :eta | G Loss: :g_loss | D Loss: :d_loss | Recon: :recon_loss | Info: :info_loss"),
      clear = FALSE, total = epochs %/% 10, width = 100)
  }else{
    pb <- progress_bar$new(
      format = paste0("Running [:bar] :percent eta: :eta | G Loss: :g_loss | D Loss: :d_loss | Recon: :recon_loss"),
      clear = FALSE, total = epochs %/% 10, width = 100)
  }
  
  if (!is.null(save.step)){
    step_result <- list()
    p <- 1
  }
  it_D <- dataloader_make_iter(Dloader)
  
  if (params$component == "match_p1"){
    pb.p1 <- progress_bar$new(
      format = paste0("Running Forward A [:bar] :percent eta: :eta | MSE: :mse_loss | CE: :ce_loss"),
      clear = FALSE, total = epochs %/% 5, width = 100)
    fnet$train()
    for (i in 1:(epochs %/% 5)){
      f_solver$zero_grad()
      batch <- dataloader_next(it_D)
      A <- batch$A$to(device = device)
      X <- batch$X$to(device = device)
      C <- batch$C$to(device = device)
      input <- torch_cat(list(X, C), dim = 2)
      A_fake <- fnet(input)
      mse_loss <- nnf_mse_loss(A_fake[, num_inds_p1, drop = F],
                               A[, num_inds_p1, drop = F])
      ce_loss <- torch_tensor(0, device = device)
      if (length(cats_p1) > 0){
        for (p1_inds in cats_p1){
          ce_loss <- ce_loss + nnf_cross_entropy(A_fake[, p1_inds, drop = F],
                                                 torch_argmax(A[, p1_inds, drop = F], dim = 2))
        }
        ce_loss <- ce_loss / length(cats_p1)
      }
      f_loss <- mse_loss + ce_loss
      f_loss$backward()
      f_solver$step()
      
      pb.p1$tick(tokens = list(mse_loss = sprintf("%.3f", mse_loss$item()),
                               ce_loss = sprintf("%.3f", ce_loss$item())))
      Sys.sleep(1 / 100000)
    }
    pb.p1$terminate()
    fnet$eval()
    for (p in fnet$parameters) p$requires_grad_(FALSE)
  }
  it_p1 <- dataloader_make_iter(p1loader)
  it_p2 <- dataloader_make_iter(p2loader)
  it_D <- dataloader_make_iter(Dloader)
  for (i in 1:epochs){
    gnet$train()
    #### D block
    batch <- dataloader_next(it_D)
    A <- batch$A$to(device = device)
    X <- batch$X$to(device = device)
    C <- batch$C$to(device = device)
    M <- batch$M$to(device = device)
    
    if (params$component == "gen_loss"){
      mask_A <- m_A_list[[(i - 1) %% bins_l + 1]]
      mask_C <- m_C_list[[(i - 1) %% bins_l + 1]]
      fakez <- torch_normal(mean = 0, std = 1, size = c(params$batch_size, params$noise_dim))$to(device = device) 
      fake <- gnet(fakez, A * mask_A, C * mask_C)
    }else{
      if (params$unconditional){
        fakez <- torch_normal(mean = 0, std = 1, size = c(params$batch_size, X$size(2)))$to(device = device) 
        fakezX <- M * X + (1 - M) * fakez
        fake <- gnet(fakezX, A, C)
      }else{
        fakez <- torch_normal(mean = 0, std = 1, size = c(params$batch_size, params$noise_dim))$to(device = device) 
        fake <- gnet(fakez, A, C)
      }
    }
    X_fake <- fake[[1]]$detach()
    if (params$component == "gen_loss"){
      F_fake <- fake[[2]]$detach()
      F_fakeact <- activationFun(F_fake, allcats, params)
      fake_AC <- torch_cat(list(F_fakeact, A * mask_A, C * mask_C), dim = 2)
      true_AC <- torch_cat(list(X, A, C, A * mask_A, C * mask_C), dim = 2)
    }else{
      X_fakeact <- activationFun(X_fake, allcats_p2, params)
      fake_AC <- torch_cat(list(X_fakeact, A, C), dim = 2)
      true_AC <- torch_cat(list(X, A, C), dim = 2)
    }
    x_fake <- dnet(fake_AC)[[1]]
    x_true <- dnet(true_AC)[[1]]
    
    d_loss <- -(torch_mean(x_true) - torch_mean(x_fake))
    if (params$lambda > 0){
      gp <- gradientPenalty(dnet, fake_AC, true_AC, params, device = device) 
      d_loss <- d_loss + params$lambda * gp
    }
    d_solver$zero_grad()
    d_loss$backward()
    d_solver$step()
    
    #### G Block
    batch_p1 <- dataloader_next(it_p1)
    batch_p2 <- dataloader_next(it_p2)
    A <- torch_cat(list(batch_p1$A, batch_p2$A), dim = 1)$to(device = device)
    X <- torch_cat(list(batch_p1$X, batch_p2$X), dim = 1)$to(device = device)
    C <- torch_cat(list(batch_p1$C, batch_p2$C), dim = 1)$to(device = device)
    M <- torch_cat(list(batch_p1$M, batch_p2$M), dim = 1)$to(device = device)
    I <- M[, 1] == 1
    
    if (params$component == "gen_loss"){
      mask_A <- m_A_list[[(i - 1) %% bins_l + 1]]
      mask_C <- m_C_list[[(i - 1) %% bins_l + 1]]
      fakez <- torch_normal(mean = 0, std = 1, size = c(params$batch_size, params$noise_dim))$to(device = device) 
      fake <- gnet(fakez, A * mask_A, C * mask_C)
    }else{
      if (params$unconditional){
        fakez <- torch_normal(mean = 0, std = 1, size = c(params$batch_size, X$size(2)))$to(device = device) 
        fakezX <- M * X + (1 - M) * fakez
        fake <- gnet(fakezX, A, C)
      }else{
        fakez <- torch_normal(mean = 0, std = 1, size = c(params$batch_size, params$noise_dim))$to(device = device) 
        fake <- gnet(fakez, A, C)
      }
    }
    X_fake <- fake[[1]]
    
    if (length(phase2_cats) > 0){
      if (params$cat == "projp1"){
        fake_proj <- projCat(X_fake, CM_tensors, cats_p2)
      }else{
        fake_proj <- NULL
      }
    }else{
      fake_proj <- NULL
    }
    
    recon_loss <- reconLoss(X_fake, X, fake_proj, A, I, params, 
                            num_inds_p2, cat_inds_p2, 
                            cats_p1, cats_p2, cats_mode)
    if (params$component == "gen_loss"){
      F_fake <- fake[[2]]
      F_tensor <- torch_cat(list(X, A, C), dim = 2)
      curr_ind <- phase1_bins_indices[[(i - 1) %% bins_l + 1]]
      
      gen_loss <- nnf_cross_entropy(F_fake[, curr_ind, drop = F]$clone(), 
                                    torch_argmax(F_tensor[, curr_ind, drop = F], dim = 2))
      recon_loss <- recon_loss + gen_loss
      
      F_fakeact <- activationFun(F_fake, allcats, params)
      fake_AC <- torch_cat(list(F_fakeact, A * mask_A, C * mask_C), dim = 2)
      true_AC <- torch_cat(list(X, A, C, A * mask_A, C * mask_C), dim = 2)
    }else{
      X_fakeact <- activationFun(X_fake, allcats_p2, params)
      fake_AC <- torch_cat(list(X_fakeact, A, C), dim = 2)
      true_AC <- torch_cat(list(X, A, C), dim = 2)
    }
    d_fake <- dnet(fake_AC)
    adv_term <- -torch_mean(d_fake[[1]])
    
    if (params$info_loss){
      info_true <- dnet(true_AC[I, ])[[2]]
      info_fake <- d_fake[[2]]
      info_loss <- infoLoss(info_fake, info_true)
      g_loss <- adv_term + recon_loss + info_loss
    }else{
      g_loss <- adv_term + recon_loss
    }
    
    if (params$component == "match_p1"){
      A_fake <- fnet(torch_cat(list(X_fake, C), dim = 2))
      mse_loss <- nnf_mse_loss(A_fake[, num_inds_p1, drop = F],
                               A[, num_inds_p1, drop = F])
      ce_loss <- torch_tensor(0, device = device)
      if (length(cats_p1) > 0){
        for (p1_inds in cats_p1){
          ce_loss <- ce_loss + nnf_cross_entropy(A_fake[, p1_inds, drop = F],
                                                 torch_argmax(A[, p1_inds, drop = F], dim = 2))
        }
        ce_loss <- ce_loss / length(cats_p1)
      }
      g_loss <- g_loss + mse_loss + ce_loss
    }
    
    g_solver$zero_grad()
    g_loss$backward()
    g_solver$step()
    training_loss[i, ] <- c(g_loss$item(), d_loss$item())
    
    if (i %% 10 == 1){
      if (params$info_loss){
        pb$tick(tokens = list(
          g_loss = sprintf("%.3f", adv_term$item()),
          d_loss = sprintf("%.3f", d_loss$item()),
          recon_loss = sprintf("%.3f", recon_loss$item()),
          info_loss = sprintf("%.3f", info_loss$item())
        ))
      }else{
        pb$tick(tokens = list(
          g_loss = sprintf("%.3f", adv_term$item()),
          d_loss = sprintf("%.3f", d_loss$item()),
          recon_loss = sprintf("%.3f", recon_loss$item())
        ))
      }
      Sys.sleep(1 / 100000)
    }
    
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
                                 device, params, allcats_p2, tensor_list)
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
                           device, params, allcats_p2, tensor_list)
  out <- list(imputation = result$imputation,
              gsample = result$gsample,
              loss = training_loss)
  if (exists("step_result", inherits = FALSE) && !is.null(step_result)){
    out$step_result <- step_result
  }
  return (out)
}