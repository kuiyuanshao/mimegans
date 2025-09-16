pacman::p_load(progress, torch)

cwgangp_default <- function(batch_size = 500, lambda = 10, 
                            alpha = 0, beta = 1, at_least_p = 0.5, 
                            lr_g = 2e-4, lr_d = 2e-4, g_betas = c(0, 0.9), d_betas = c(0, 0.9), 
                            g_weight_decay = 0, d_weight_decay = 0, noise_dim = 128, 
                            g_dim = c(256, 256), d_dim = c(256, 256), pac = 10, discriminator_steps = 1,
                            tau = 0.2, hard = F, type_g = "mlp", type_d = "mlp",
                            num = "mmer", cat = "projp1", component = "none", info_loss = F){
  batch_size <- pac * round(batch_size / pac)
  if (component == "match_p1"){
    alpha <- 0.05
  }
  if (info_loss){
    req <- ceiling(at_least_p * batch_size)
    n <- ceiling(req / pac) * pac
    if (n > batch_size) n <- floor(batch_size / pac) * pac
    at_least_p <- n / batch_size
  }
  list(batch_size = batch_size, lambda = lambda, alpha = alpha, beta = beta,
       at_least_p = at_least_p, lr_g = lr_g, lr_d = lr_d, g_betas = g_betas, d_betas = d_betas, 
       g_weight_decay = g_weight_decay, d_weight_decay = d_weight_decay, noise_dim = noise_dim,
       g_dim = g_dim, d_dim = d_dim, pac = pac, discriminator_steps = discriminator_steps, tau = tau, hard = hard,
       type_g = type_g, type_d = type_d, num = num, cat = cat, component = component, info_loss = info_loss)
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
  phase1_bins <- data_info$cat_vars[!(data_info$cat_vars %in% data_info$phase2_vars)] 
  phase1_bins <- if (length(phase1_bins) > 0) {
    phase1_bins[sapply(phase1_bins, function(col) {
      length(unique(data[phase1_rows, col])) > 1 & length(unique(data[phase2_rows, col])) > 1
    })]
  } else {
    character(0)
  }
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
  tensor_list <- list(data_mask, conditions_t, phase2_t, phase1_t)
  
  gnet <- do.call(paste("generator", params$type_g, sep = "."), 
                  args = list(params, ncols, 
                              length(phase2_vars_encode), 
                              length(phase1_vars_encode),
                              rate = 0.5))$to(device = device)
  if (params$component == "match_p1"){
    dnet <- do.call(paste("discriminator", params$type_d, sep = "."), 
                    args = list(params, ncols - length(phase2_vars_encode)))$to(device = device)
    sample_net <- "G"
  }else{
    dnet <- do.call(paste("discriminator", params$type_d, sep = "."), 
                    args = list(params, ncols))$to(device = device)
    sample_net <- "D"
  }

  g_solver <- torch::optim_adam(gnet$parameters, lr = params$lr_g, 
                                betas = params$g_betas, weight_decay = params$g_weight_decay)
  d_solver <- torch::optim_adam(dnet$parameters, lr = params$lr_d, 
                                betas = params$d_betas, weight_decay = params$d_weight_decay)
  
  training_loss <- matrix(0, nrow = epochs, ncol = 2)
  
  if (params$info_loss){
    pb <- progress_bar$new(
      format = paste0("Running [:bar] :percent eta: :eta | G Loss: :g_loss | D Loss: :d_loss | Recon: :recon_loss | Info: :info_loss"),
      clear = FALSE, total = epochs, width = 100)
  }else{
    pb <- progress_bar$new(
      format = paste0("Running [:bar] :percent eta: :eta | G Loss: :g_loss | D Loss: :d_loss | Recon: :recon_loss"),
      clear = FALSE, total = epochs, width = 100)
  }
  
  if (!is.null(save.step)){
    step_result <- list()
    p <- 1
  }
  
  for (i in 1:epochs){
    gnet$train()
    for (d in 1:params$discriminator_steps){
      batch <- sampleBatch(data, tensor_list, phase1_bins,
                           phase1_rows, phase2_rows,
                           params$batch_size, params$at_least_p,
                           weights, net = sample_net)
      A <- batch[[4]]
      X <- batch[[3]]
      C <- batch[[2]]
      M <- batch[[1]]
      I <- M[, 1] == 1
      fakez <- torch_normal(mean = 0, std = 1, size = c(params$batch_size, params$noise_dim))$to(device = device) 
      fakez_AC <- torch_cat(list(fakez, A, C), dim = 2)
      
      fake <- gnet(fakez_AC)
      X_fake <- fake[[1]]
      if (params$component == "match_p1"){
        A_fake <- fake[[2]]
        A_fakeact <- activationFun(A_fake, cats_p1, params)
        fake_AC <- torch_cat(list(A_fakeact, C), dim = 2)
        true_AC <- torch_cat(list(A, C), dim = 2)
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
    }
    batch <- sampleBatch(data, tensor_list, phase1_bins,
                         phase1_rows, phase2_rows,
                         params$batch_size, params$at_least_p,
                         weights, net = "G")
    A <- batch[[4]]
    X <- batch[[3]]
    C <- batch[[2]]
    M <- batch[[1]]
    I <- M[, 1] == 1
    
    fakez <- torch_normal(mean = 0, std = 1, 
                          size = c(params$batch_size, params$noise_dim))$to(device = device) 
    fakez_AC <- torch_cat(list(fakez, A, C), dim = 2)
    fake <- gnet(fakez_AC)
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
    if (params$component == "match_p1"){
      A_fake <- fake[[2]]
      A_fakeact <- activationFun(A_fake, cats_p1, params)
      fake_AC <- torch_cat(list(A_fakeact, C), dim = 2)
      true_AC <- torch_cat(list(A, C), dim = 2)
      # mse_loss <- nnf_mse_loss(A_fake[, num_inds_p1, drop = F],
      #                          A[, num_inds_p1, drop = F])
      # ce_loss <- torch_tensor(0, device = device)
      # if (length(cats_p1) > 0){
      #   for (p1_inds in cats_p1){
      #     ce_loss <- ce_loss + nnf_cross_entropy(A_fake[, p1_inds, drop = F],
      #                                            torch_argmax(A[, p1_inds, drop = F], dim = 2))
      #   }
      #   ce_loss <- ce_loss / length(cats_p1)
      # }
      # recon_loss <- recon_loss + params$alpha * mse_loss + params$beta * ce_loss
    }else if (params$component == "gen_loss"){
      curr_var <- batch[[7]]
      curr_inds <- data_encode$binary_indices[[curr_var]]
      F_fake <- fake[[2]]
      F_tensor <- torch_cat(list(X, A, C), dim = 2)
      gen_loss <- nnf_cross_entropy(F_fake[, curr_inds, drop = F],
                                    torch_argmax(F_tensor[, curr_inds, drop = F], dim = 2))
      recon_loss <- recon_loss + gen_loss
      
      X_fakeact <- activationFun(X_fake, allcats_p2, params)
      fake_AC <- torch_cat(list(X_fakeact, A, C), dim = 2)
      true_AC <- torch_cat(list(X, A, C), dim = 2)
    }else{
      X_fakeact <- activationFun(X_fake, allcats_p2, params)
      fake_AC <- torch_cat(list(X_fakeact, A, C), dim = 2)
      true_AC <- torch_cat(list(X, A, C), dim = 2)
    }
    d_fake <- dnet(fake_AC)
    x_fake <- d_fake[[1]]
    adv_term <- -torch_mean(x_fake)
    
    if (params$info_loss){
      info_true <- dnet(true_AC[I, ])[[2]]
      info_fake <- d_fake[[2]] # dnet(fake_AC[I, ])[[2]]
      info_loss <- infoLoss(info_fake, info_true)
      g_loss <- adv_term + recon_loss + info_loss
    }else{
      g_loss <- adv_term + recon_loss
    }
    
    g_solver$zero_grad()
    g_loss$backward()
    g_solver$step()
    training_loss[i, ] <- c(g_loss$item(), d_loss$item())
    
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
    
    if (!is.null(save.step)){
      if (i %% save.step == 0){
        gnet$eval()
        for (modu in gnet$modules) {
          if (inherits(modu, "nn_dropout")) {
            modu$train(TRUE)
          }
        }
        result <- generateImpute(gnet, dnet, m = 5, 
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
  result <- generateImpute(gnet, dnet, m = m, 
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


# mimegans.cv <- function(data, fold = 5, data_info, parameters_grid, seed = 1){
#   set.seed(seed)
#   phase2_data <- data[!is.na(data[[data_info$phase2_vars[1]]]), ]
#   ind <- sample(1:nrow(phase2_data)) 
#   map <- setNames(rep_len(seq_len(fold), length(ind)), ind)
#   map <- map[as.character(1:nrow(phase2_data))]
#   cv_split <- function(data, map, i){
#     inds <- list(train = which(map != i), val = which(map == i))
#     curr_data <- data
#     curr_data[inds[[2]], data_info$phase2_vars] <- NA
#     return (curr_data)
#   }
#   splits <- lapply(1:fold, function(i) cv_split(phase2_data, map, i))
#   result <- parameters_grid
#   result$rmse_num <- 0
#   result$mis_cat <- 0 
#   result$w_loss <- 0
#   for (i in 1:nrow(parameters_grid)){
#     params <- as.list(parameters_grid[i, , drop = FALSE])
#     curr_loss <- c(0, 0, 0)
#     for (j in 1:fold){
#       curr_data <- splits[[j]]
#       curr_imp <- mimegans(curr_data, epochs = 2000,
#                            m = 1, params = params, data_info = data_info)
#       curr_wloss <- curr_imp$w_loss
#       curr_foldloss <- lossCalc(curr_imp, phase2_data, data_info, 
#                                 is.na(curr_data[, data_info$phase2_vars[1]]))
#       curr_loss <- curr_loss + c(curr_foldloss, curr_wloss)
#     }
#     result$rmse_num[i] <- curr_loss[1]
#     result$mis_cat[i] <- curr_loss[2]
#     result$w_loss[i] <- curr_loss[3]
#     cat(i, ":", curr_loss, "\n")
#   }
#   return (result)
# }