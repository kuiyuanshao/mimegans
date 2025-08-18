library(torch)
library(progress)
source("../mimegans/encoding.R")
gain <- function(data, data_info, device = "cpu", batch_size = 128, hint_rate = 0.9, 
                 alpha = 10, beta = 1, n = 10000){
  device <- torch_device(device)
  loss_mat <- matrix(NA, nrow = n, ncol = 2)
  
  numCol <- which(names(data) %in% data_info$num_vars)
  norm_result <- normalize(data, numCol)
  norm_data <- norm_result$norm_data
  norm_parameters <- norm_result$norm_parameters
  
  data_encode <- encode.onehot(norm_data, data_info$cat_vars, data_info$cat_vars, 
                               data_info$phase1_vars, data_info$phase2_vars)
  data_training <- data_encode$data
  
  num_inds <- which(names(data_training) %in% num_vars)
  cat_inds <- unlist(data_encode$new_col_names)
  
  data_mask <- 1 - is.na(data_training)
  data_training[is.na(data_training)] <- 0
  data_mat <- as.matrix(data_training)
  
  nRow <- dim(data_training)[1]
  nCol <- dim(data_training)[2]
  
  X_t <- torch::torch_tensor(data_mat, device = device)
  M_t <- torch::torch_tensor(data_mask, device = device)
  
  GAIN_Generator <- torch::nn_module(
    initialize = function(nCol){
      self$seq <- torch::nn_sequential()
      self$seq$add_module(module = torch::nn_linear(nCol * 2, nCol),
                          name = "Linear1")
      self$seq$add_module(module = torch::nn_relu(),
                          name = "Activation1")
      self$seq$add_module(module = torch::nn_linear(nCol, nCol),
                          name = "Linear2")
      self$seq$add_module(module = torch::nn_relu(),
                          name = "Activation2")
      self$seq$add_module(module = torch::nn_linear(nCol, nCol),
                          name = "Linear3")
      self$seq$add_module(module = torch::nn_sigmoid(),
                          name = "Output")
    },
    forward = function(input){
      input <- self$seq(input)
      input
    }
  )
  
  GAIN_Discriminator <- torch::nn_module(
    initialize = function(nCol){
      self$seq <- torch::nn_sequential()
      self$seq$add_module(module = torch::nn_linear(nCol * 2, nCol),
                          name = "Linear1")
      self$seq$add_module(module = torch::nn_relu(),
                          name = "Activation1")
      self$seq$add_module(module = torch::nn_linear(nCol, nCol),
                          name = "Linear2")
      self$seq$add_module(module = torch::nn_relu(),
                          name = "Activation2")
      self$seq$add_module(module = torch::nn_linear(nCol, nCol),
                          name = "Linear3")
      self$seq$add_module(module = torch::nn_sigmoid(),
                          name = "Output")
    },
    forward = function(input){
      input <- self$seq(input)
      input
    }
  )
  
  G_layer <- GAIN_Generator(nCol)$to(device = device)
  D_layer <- GAIN_Discriminator(nCol)$to(device = device)
  
  G_solver <- torch::optim_adam(G_layer$parameters)
  D_solver <- torch::optim_adam(D_layer$parameters)
  
  generator <- function(X, M){
    input <- torch_cat(list(X, M), dim = 2)
    return (G_layer(input))
  }
  discriminator <- function(X, H){
    input <- torch_cat(list(X, H), dim = 2)
    return (D_layer(input))
  }
  
  G_loss <- function(X, M, H){
    G_sample <- generator(X, M)
    X_hat <- X * M + G_sample * (1 - M)
    D_prob <- discriminator(X_hat, H)
    
    G_loss1 <- -torch_mean((1 - M) * torch_log(D_prob + 1e-8))

    mse_loss <- torch_mean((M[, num_inds, drop = F] * X[, num_inds, drop = F] - 
                             M[, num_inds, drop = F] * G_sample[, num_inds, drop = F]) ^ 2) / torch_mean(M[, num_inds, drop = F])
    cross_entropy <- -torch_mean(X[, cat_inds, drop = F] * M[, cat_inds, drop = F] * 
                                 torch_log(G_sample[, cat_inds, drop = F] + 1e-8) + 
                                 (1 - X[, cat_inds, drop = F]) * M[, cat_inds, drop = F] * 
                                   torch_log(1 - (G_sample[, cat_inds, drop = F] + 1e-8)))
    return (G_loss1 + alpha * mse_loss + beta * cross_entropy)
  }
  D_loss <- function(X, M, H){
    G_sample <- generator(X, M)
    X_hat <- X * M + G_sample * (1 - M)
    D_prob <- discriminator(X_hat, H)
    
    D_loss1 <- -torch_mean(M * torch_log(D_prob + 1e-8) + (1 - M) * torch_log(1 - D_prob + 1e-8))
    return (D_loss1)
  }
  
  
  pb <- progress_bar$new(
    format = "Running :what [:bar] :percent eta: :eta | G Loss: :g_loss | D Loss: :d_loss",
    clear = FALSE, total = n, width = 100)
  
  for (i in 1:n){
    ind_batch <- new_batch(X_t, M_t, nRow, batch_size, device)
    X_mb <- ind_batch[[1]]
    M_mb <- ind_batch[[2]]
    
    Z_mb <- ((-0.01) * torch::torch_rand(c(batch_size, nCol)) + 0.01)$to(device = device)
    H_mb <- 1 * (matrix(runif(batch_size * nCol, 0, 1), nrow = batch_size, ncol = nCol) < hint_rate)
    H_mb <- torch_tensor(H_mb, device = device)
    
    H_mb <- M_mb * H_mb
    X_mb <- M_mb * X_mb + (1 - M_mb) * (Z_mb)
    X_mb <- X_mb$to(device = device)
    
    d_loss <- D_loss(X_mb, M_mb, H_mb)
  
    D_solver$zero_grad()
    d_loss$backward()
    D_solver$step()
    
    g_loss <- G_loss(X_mb, M_mb, H_mb)
    
    G_solver$zero_grad()
    g_loss$backward()
    G_solver$step()
    
    pb$tick(tokens = list(what = "GAIN   ", 
                          g_loss = sprintf("%.4f", g_loss$item()),
                          d_loss = sprintf("%.4f", d_loss$item())))
    Sys.sleep(1 / 10000)
    loss_mat[i, ] <- c(g_loss$item(), d_loss$item())
  }
  imputed_data_list <- vector("list", m)
  gsample_data_list <- vector("list", m)
  for (j in 1:m){
    Z <- ((-0.01) * torch::torch_rand(c(nRow, nCol)) + 0.01)$to(device = device)
    X <- M_t * X_t + (1 - M_t) * Z
    X <- X$to(device = device)
    M <- M_t
    
    G_sample <- generator(X, M)
    
    imputed_data <- M_t * X + (1 - M_t) * G_sample
    imputed_data <- imputed_data$detach()$cpu()
    imputed_data <- as.data.frame(imputed_data)
    names(imputed_data) <- names(data_training)
    
    imputed_data <- decode.onehot(imputed_data, data_encode)
    imputed_data <- renormalize(imputed_data, norm_parameters, num_inds)
    
    imputed_data_list[[j]] <- imputed_data
  }
  loss_mat <- as.data.frame(loss_mat)
  names(loss_mat) <- c("G_loss", "D_loss")
  return (list(imputations = imputed_data_list, 
               loss = loss_mat))
}