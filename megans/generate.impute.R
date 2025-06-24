pmm <- function(yhatobs, yhatmis, yobs, k) {
  idx <- mice::matchindex(d = yhatobs, t = yhatmis, k = k)
  yobs[idx]
}

match_types <- function(new_df, orig_df) {
  common <- intersect(names(orig_df), names(new_df))
  out <- new_df
  
  for (nm in common) {
    tmpl <- orig_df[[nm]]
    col <- out[[nm]]
    if (is.integer(tmpl))        out[[nm]] <- as.integer(col)
    else if (is.numeric(tmpl))   out[[nm]] <- as.numeric(col)
    else if (is.logical(tmpl))   out[[nm]] <- as.logical(col)
    else if (is.factor(tmpl)) {
      out[[nm]] <- factor(col,
                          levels = levels(tmpl),
                          ordered = is.ordered(tmpl))
    }
    else if (inherits(tmpl, "Date")) {
      out[[nm]] <- as.Date(col)
    } else if (inherits(tmpl, "POSIXct")) {
      tz <- attr(tmpl, "tzone")
      out[[nm]] <- as.POSIXct(col, tz = tz)
    }
    else {
      out[[nm]] <- as.character(col)
    }
  }
  out
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
                           batch_size, g_dim, device, params,
                           tensor_list, tokenizer_list){ #, phase1_rows, phase2_rows, vars_to_pmm){
  imputed_data_list <- vector("list", m)
  gsample_data_list <- vector("list", m)
  batchforimpute <- create_bfi(data_original, batch_size, tensor_list)
  data_mask <- as.matrix(1 - is.na(data_original))
  
  num_col <- sapply(data_original, is.numeric)
  phase2_num_idx <- which(num_col & names(data_original) %in% phase2_vars)
  data_original[which(is.na(data_original[, phase2_vars[1]])), phase2_num_idx] <- 0
  
  for (z in 1:m){
    output_list <- vector("list", length(batchforimpute))
    for (i in 1:length(batchforimpute)){
      batch <- batchforimpute[[i]]
      X <- batch[[3]]
      C <- batch[[2]]
      M <- batch[[1]]

      fakez <- torch_normal(mean = 0, std = 1, size = c(X$size(1), g_dim))$to(device = device)
      if (params$tokenize){
        C_token <- tokenizer_list$tokenizer(C[, tokenizer_list$num_inds_p1, drop = F], 
                                            C[, tokenizer_list$cat_inds_p1, drop = F])
        C_token <- C_token$reshape(c(C_token$size(1), 
                                     C_token$size(2) * C_token$size(3)))
        fakez_C <- torch_cat(list(fakez, C_token), dim = 2)
      }else{
        fakez_C <- torch_cat(list(fakez, C), dim = 2)
      }
      gsample <- gnet(fakez_C)
      gsample <- activation_fun(gsample, data_encode, phase2_vars, tau = 0.2, hard = T, gen = F)
      gsample <- torch_cat(list(gsample, C), dim = 2)
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
      to_replace <- data_mask[, !num_col] == 0
      idx <- which(to_replace, arr.ind = TRUE)
      col_ids <- which(!num_col)[idx[, "col"]]
      imputations[cbind(idx[, "row"], col_ids)] <- gsamples[cbind(idx[, "row"], col_ids)]
    }
    
    imputed_data_list[[z]] <- imputations
    gsample_data_list[[z]] <- gsamples
  }
  return (list(imputation = imputed_data_list, gsample = gsample_data_list))
}
