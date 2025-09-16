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

acc_prob_row <- function(mat, lb, ub, alpha = 1) {
  nr <- nrow(mat)
  UB <- matrix(ub, nr, length(ub), byrow = TRUE)
  LB <- matrix(lb, nr, length(lb), byrow = TRUE)
  span <- matrix((ub - lb) / 2, nr, length(lb), byrow = TRUE)
  
  hi <- pmax(mat - UB, 0)
  lo <- pmax(LB - mat, 0)
  d <- pmax(hi, lo) / span
  
  total_violation <- rowSums(d)
  exp(-alpha * total_violation ^ 2)
}

gen_rows <- function(idx_vec, gnet, A_t, C_t, params, 
                     data_original, data_training, decode, denormalize,
                     data_encode, data_norm, data_info, 
                     num_inds_gen, num_inds_ori, all_cats_p2, device){
  n <- length(idx_vec)
  z <- torch_normal(0, 1, size = c(n, params$noise_dim))$to(device = device)
  tmp <- gnet(torch_cat(list(z, A_t[idx_vec, , drop = F], 
                             C_t[idx_vec, , drop = F]), dim = 2))[[1]]
  tmp <- activationFun(tmp, all_cats_p2, params, gen = T)
  tmp <- torch_cat(list(tmp, A_t[idx_vec, , drop = F], C_t[idx_vec, , drop = F]), dim = 2)
  df <- as.data.frame(as.matrix(tmp$detach()$cpu()))
  names(df) <- names(data_training)
  deco <- do.call(decode, args = list(
    data = df,
    encode_obj = data_encode
  ))
  denorm <- do.call(denormalize, args = list(
    data = deco,
    num_vars = data_info$num_vars,
    norm_obj = data_norm
  ))$data
  
  denorm <- denorm[, names(data_original)]
  
  if (params$num == "mmer"){
    denorm[, num_inds_gen] <- data_original[idx_vec, num_inds_ori] - denorm[, num_inds_gen]
  }
  denorm
}

generateImpute <- function(gnet, dnet, m = 5, 
                           data_original, data_info, data_norm, 
                           data_encode, data_training,
                           phase1_vars, phase2_vars,
                           num.normalizing, cat.encoding, 
                           device, params, all_cats_p2,
                           tensor_list){
  imputed_data_list <- vector("list", m)
  gsample_data_list <- vector("list", m)
  w_loss <- 1:m
  
  denormalize <- paste("denormalize", num.normalizing, sep = ".")
  decode <- paste("decode", cat.encoding, sep = ".")
  
  batchforimpute <- create_bfi(data_original, params$batch_size, tensor_list)
  data_mask <- as.matrix(1 - is.na(data_original))
  
  phase2_rows <- which(!is.na(data_original[, data_info$phase2_vars[1]]))
  p2_num <- intersect(phase2_vars, data_info$num_vars)
  lb <- vapply(p2_num, function(v) min(data_original[[v]], na.rm = TRUE), numeric(1))
  ub <- vapply(p2_num, function(v) max(data_original[[v]], na.rm = TRUE), numeric(1))
  names(ub) <- names(lb) <- p2_num
  max_attempt <- 5
  
  num_col <- names(data_original) %in% data_info$num_vars
  suppressWarnings(data_original[is.na(data_original)] <- 0)
  
  data_names <- names(data_training)
  out_names  <- names(data_original)
  
  p2_idx_out <- match(data_info$phase2_vars, out_names)
  p2_idx_out <- sort(p2_idx_out)
  p2_num_idx_out <- match(p2_num, out_names[p2_idx_out])
  num_inds_gen <- match(phase2_vars[phase2_vars %in% data_info$num_vars], names(data_original))
  num_inds_ori <- match(phase1_vars[phase1_vars %in% data_info$num_vars], names(data_original))
  
  A_t <- tensor_list[[4]]
  C_t <- tensor_list[[2]]
  
  for (z in 1:m){
    output_list <- vector("list", length(batchforimpute))
    for (i in 1:length(batchforimpute)){
      batch <- batchforimpute[[i]]
      A <- batch[[4]]
      X <- batch[[3]]
      C <- batch[[2]]
      
      fakez <- torch_normal(mean = 0, std = 1, 
                            size = c(X$size(1), params$noise_dim))$to(device = device)
      fakez_C <- torch_cat(list(fakez, A, C), dim = 2)
      gsample <- gnet(fakez_C)[[1]]
      gsample <- activationFun(gsample, all_cats_p2, params, gen = T)
      gsample <- torch_cat(list(gsample, A, C), dim = 2)
      output_list[[i]] <- gsample
    }
    output_mat <- torch_cat(output_list)
    output_frame <- as.data.frame(as.matrix(output_mat$detach()$cpu()))
    names(output_frame) <- names(data_training)
    
    curr_gsample <- do.call(decode, args = list(
      data = output_frame,
      encode_obj = data_encode
    ))
    curr_gsample <- do.call(denormalize, args = list(
      data = curr_gsample,
      num_vars = data_info$num_vars, 
      norm_obj = data_norm
    ))
    #get the original order
    gsamples <- curr_gsample$data
    gsamples <- gsamples[, names(data_original)]

    if (params$num == "mmer"){
      gsamples[, num_inds_gen] <- data_original[, num_inds_ori] - gsamples[, num_inds_gen]
    }
    
    M <- gsamples[, p2_idx_out, drop = FALSE]
    row_acc <- acc_prob_row(as.matrix(M[, p2_num_idx_out, drop = FALSE]), lb, ub)
    accept_row <- runif(nrow(M)) < row_acc
    iter <- 0L
    while (iter < max_attempt && any(!accept_row)) {
      bad <- which(!accept_row)
      new_rows <- gen_rows(bad, gnet, A_t, C_t, params,
                           data_original, data_training, decode, denormalize,
                           data_encode, data_norm, data_info,
                           num_inds_gen, num_inds_ori,
                           all_cats_p2, device)
      M[bad, ] <- new_rows[, p2_idx_out, drop = FALSE]
      row_acc <- acc_prob_row(as.matrix(M[, p2_num_idx_out, drop = FALSE]), lb, ub)
      accept_row <- runif(nrow(M)) < row_acc
      iter <- iter + 1L
    }
    gsamples[, p2_idx_out] <- M
    
    for (v in names(lb)){
      oob_ind <- which(gsamples[[v]] < lb[[v]] | gsamples[[v]] > ub[[v]])
      if (length(oob_ind) > 0){
        yhatmis <- pmm(gsamples[[v]][data_original$R == 1],
                       gsamples[[v]][oob_ind],
                       data_original[[v]][data_original$R == 1], 5)
        gsamples[[v]][oob_ind] <- yhatmis
      }
    }
    
    imputations <- data_original
    if (any(num_col)) {
      out_num <- data_mask[, num_col] * as.matrix(data_original[, num_col]) +
        (1 - data_mask[, num_col]) * as.numeric(as.matrix(gsamples[, num_col]))
      imputations[, num_col] <- out_num
    }
    if (any(!num_col)) {
      fac_cols <- which(!num_col)
      for (j in fac_cols) {
        new_lvls <- unique(as.character(gsamples[, j]))
        levels(imputations[[j]]) <- union(levels(imputations[[j]]), new_lvls)
      }
      to_replace <- data_mask[, fac_cols] == 0
      idx <- which(to_replace, arr.ind = TRUE)
      col_ids <- fac_cols[idx[, "col"]]
      if (nrow(idx) > 0){
        imputations[cbind(idx[, "row"], col_ids)] <-
          gsamples[cbind(idx[, "row"], col_ids)]
      }
    }
    
    imputed_data_list[[z]] <- imputations
    gsample_data_list[[z]] <- gsamples
  }
  return (list(imputation = imputed_data_list, gsample = gsample_data_list))
}
