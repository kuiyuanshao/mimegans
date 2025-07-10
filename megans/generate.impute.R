pmm <- function(yhatobs, yhatmis, yobs, k) {
  idx <- mice::matchindex(d = yhatobs, t = yhatmis, k = k)
  yobs[idx]
}

create_bfi <- function(data, batch_size, tensor_list){
  n <- ceiling(nrow(data) / batch_size)
  idx <- 1
  batches <- vector("list", n)
  for (i in 1:n){
    if (i == n){
      batch_size <- nrow(data) - batch_size * (n - 1)
    }
    batches[[i]] <- lapply(tensor_list, function(tensor) tensor[idx:(idx + batch_size - 1), , drop = FALSE])
    idx <- idx + batch_size
  }
  return (batches = batches)
}

generateImpute <- function(gnet, m = 5, 
                           data_original, data_norm, 
                           data_encode, data_training,
                           phase1_vars, phase2_vars, 
                           num_vars, num.normalizing, cat.encoding, 
                           batch_size, device, params,
                           tensor_list, type, log_shift){
  imputed_data_list <- vector("list", m)
  gsample_data_list <- vector("list", m)
  batchforimpute <- create_bfi(data_original, batch_size, tensor_list)
  data_mask <- as.matrix(1 - is.na(data_original))
  
  phase2_num_name <- intersect(phase2_vars, names(data_original))[intersect(phase2_vars, names(data_original)) %in% num_vars]
  ub <- sapply(phase2_num_name, function(v) max(data_original[[v]], na.rm = T), 
               simplify = TRUE, USE.NAMES = TRUE)
  lb <- sapply(phase2_num_name, function(v) min(data_original[[v]], na.rm = T), 
               simplify = TRUE, USE.NAMES = TRUE)
  names(ub) <- phase2_num_name
  names(lb) <- phase2_num_name
  max_attempt <- 25
  
  num_col <- sapply(data_original, is.numeric)
  phase2_num_idx <- which(num_col & names(data_original) %in% phase2_vars)
  data_original[which(is.na(data_original[, phase2_vars[1]])), phase2_num_idx] <- 0
  
  A_t <- tensor_list[[4]]
  C_t <- tensor_list[[2]]
  
  for (z in 1:m){
    output_list <- vector("list", length(batchforimpute))
    #noise_mat <- NULL
    for (i in 1:length(batchforimpute)){
      batch <- batchforimpute[[i]]
      A <- batch[[4]]
      X <- batch[[3]]
      C <- batch[[2]]

      fakez <- torch_normal(mean = 0, std = 1, 
                            size = c(X$size(1), params$noise_dim))$to(device = device)
      fakez_C <- torch_cat(list(fakez, A, C), dim = 2)
      
      gsample <- gnet(fakez_C)
      
      gsample <- activation_fun(gsample, data_encode, phase2_vars)
      gsample <- torch_cat(list(gsample, A, C), dim = 2)
      #noise_mat <- rbind(noise_mat, as.matrix(fakez$detach()$cpu()))
      output_list[[i]] <- as.matrix(gsample$detach()$cpu())
    }
    output_mat <- as.data.frame(do.call(rbind, output_list))
    names(output_mat) <- names(data_training)
    
    denormalize <- paste("denormalize", num.normalizing, sep = ".")
    decode <- paste("decode", cat.encoding, sep = ".")
    
    curr_gsample <- do.call(decode, args = list(
      data = output_mat,
      encode_obj = data_encode
    ))
    curr_gsample <- do.call(denormalize, args = list(
      data = curr_gsample,
      num_vars = num_vars, 
      norm_obj = data_norm
    ))
    #get the original order
    gsamples <- curr_gsample$data
    gsamples <- gsamples[, names(data_original)]
    gsamples[, which(names(gsamples) %in% phase1_vars)] <- 
      data_original[, which(names(gsamples) %in% phase1_vars)]
    
    if (type == "mmer"){
      gsamples[, match(phase2_vars[phase2_vars %in% num_vars], names(gsamples))] <-
        data_original[, match(phase1_vars[phase1_vars %in% num_vars], names(data_original))] -
        gsamples[, match(phase2_vars[phase2_vars %in% num_vars], names(gsamples))]
    }
    
    acc_prob <- function(mat) {
      nr <- nrow(mat)
      hi <- pmax(mat - rep(ub, each = nr), 0)
      lo <- pmax(rep(lb, each = nr) - mat, 0)
      worst <- apply(pmax(hi, lo), 1L, max)
      exp(-4 * worst^2)
    }
    M <- as.matrix(gsamples[ , match(phase2_vars[phase2_vars %in% num_vars], names(gsamples)), drop = FALSE])
    acc <- acc_prob(M)
    todo <- which(runif(length(acc)) > acc)
    iter <- 0
    while (length(todo) > 0 && iter < max_attempt) {
      iter <- iter + 1
      n_bad <- length(todo)
      z_bad <- torch_normal(mean = 0, std = 1, size = c(n_bad, params$noise_dim))$to(device = device)
      #z_bad <- torch_tensor(noise_mat[todo, ], device = device)
      fake_bad <- gnet(torch_cat(list(z_bad, A_t[todo,], C_t[todo, ]), dim = 2))
      fake_bad <- activation_fun(fake_bad, data_encode, phase2_vars)
      fake_bad <- torch_cat(list(fake_bad, A_t[todo,], C_t[todo, ]), dim = 2)
      fake_bad_cpu <- as.data.frame(as.matrix(fake_bad$detach()$cpu()))
      names(fake_bad_cpu) <- names(data_training)

      fake_bad_dec <- do.call(decode, list(data = fake_bad_cpu, encode_obj = data_encode))
      fake_bad_den <- do.call(denormalize, list(data = fake_bad_dec,
                                                num_vars = num_vars,
                                                norm_obj = data_norm))$data

      if (type == "mmer") {
        fake_bad_den[, match(phase2_vars[phase2_vars %in% num_vars], names(fake_bad_den))] <-
          data_original[todo, match(phase1_vars[phase1_vars %in% num_vars], names(data_original))] -
          fake_bad_den[, match(phase2_vars[phase2_vars %in% num_vars], names(fake_bad_den))]
      }
      M[todo, ] <- as.matrix(fake_bad_den[, match(phase2_vars[phase2_vars %in% num_vars], names(fake_bad_den))])
      acc_new <- acc_prob(M[todo, , drop = FALSE])
      todo <- todo[runif(length(acc_new)) > acc_new]
    }
    # if (length(todo) > 0) {
    #   warning("Reached max_attempt; remaining out-of-bound values clipped.")
    #   M <- sweep(M, 2, ub, pmin)
    #   M <- sweep(M, 2, lb, pmax)
    # }
    gsamples[, match(phase2_vars[phase2_vars %in% num_vars], names(gsamples))] <- M
    # vars_to_pmm <- "T_I"
    # if (!is.null(vars_to_pmm)){
    #   for (i in vars_to_pmm){
    #     if (i %in% num_vars){
    #       pmm_matched <- pmm(gsamples[data_original$R == 1, i],
    #                          gsamples[data_original$R == 0, i],
    #                          data_original[data_original$R == 1, i], 5)
    #       gsamples[data_original$R == 0, i] <- pmm_matched
    #     }
    #   }
    # }
    
    imputations <- data_original
    # ---- numeric columns: use matrix arithmetic ---
    if (any(num_col)) {
      out_num <- data_mask[, num_col] * as.matrix(data_original[, num_col]) +
        (1 - data_mask[, num_col]) * as.numeric(as.matrix(gsamples[, num_col]))
      imputations[, num_col] <- out_num
    }
    # ---- non-numeric columns (factor / character / etc.) -------------
    if (any(!num_col)) {
      fac_cols <- which(!num_col)
      
      # add missing levels once per factor column
      for (j in fac_cols) {
        new_lvls <- unique(as.character(gsamples[, j]))
        levels(imputations[[j]]) <- union(levels(imputations[[j]]), new_lvls)
      }
      to_replace <- data_mask[, fac_cols] == 0
      idx <- which(to_replace, arr.ind = TRUE)
      col_ids <- fac_cols[idx[, "col"]]
      
      imputations[cbind(idx[, "row"], col_ids)] <-
        gsamples[cbind(idx[, "row"], col_ids)]
    }
    
    imputed_data_list[[z]] <- imputations
    gsample_data_list[[z]] <- gsamples
  }
  return (list(imputation = imputed_data_list, gsample = gsample_data_list))
}
