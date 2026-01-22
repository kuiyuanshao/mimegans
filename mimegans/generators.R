Residual <- torch::nn_module(
  "Residual",
  initialize = function(dim1, dim2, rate, ...){
    self$rate <- rate
    self$resid <- resid
    self$linear <- nn_linear(dim1, dim2)
    self$norm <- nn_batch_norm1d(dim2)
    self$act <- nn_elu()
    self$dropout <- nn_dropout(rate)
  },
  forward = function(input){
    output <- self$act(self$norm(self$linear(input)))
    if (self$rate > 0){
      output <- self$dropout(output)
    }
    return (torch_cat(list(output, input), dim = 2))
  }
)

generator.mlp <- nn_module(
  "Generator",
  initialize = function(params, ...){
    self$params <- params
    self$nphase2 <- params$nphase2

    dim1 <- params$noise_dim + params$cond_dim
    
    self$dropout <- nn_dropout(params$g_dropout / 2)

    self$cond_encoder <- nn_sequential(
      nn_linear(params$ncols - params$nphase2, params$cond_dim),
      nn_batch_norm1d(params$cond_dim),
      nn_leaky_relu(0.2)
    )

    self$seq <- nn_sequential()
    for (i in 1:length(params$g_dim)){
      self$seq$add_module(paste0("Residual_", i), Residual(dim1, params$g_dim[i], params$g_dropout))
      dim1 <- dim1 + params$g_dim[i]
    }
    self$seq$add_module("Linear", nn_linear(dim1, params$nphase2))
  },
  forward = function(N, A, C, ...){
    cond <- torch_cat(list(A, C), dim = 2)
    cond <- self$cond_encoder(cond)
    if (self$params$g_dropout > 0){
      cond <- self$dropout(cond)
    }
    input <- torch_cat(list(N, cond), dim = 2)

    X_fake <- self$seq(input)
    return (X_fake)
  }
)


generator.mlpc <- nn_module(
  "Generator",
  initialize = function(params, ...){
    self$params <- params
    
    self$cat_groups <- params$cat_groups
    self$num_inds_target <- params$num_inds
    
    self$emb_dim <- 16 
    
    self$cat_embedders <- nn_module_list()
    self$cat_delta_heads <- nn_module_list()
    self$cat_decoders <- nn_module_list()
    
    total_cat_emb_dim <- 0
    
    dim1 <- params$noise_dim + params$cond_dim + self$emb_dim * length(self$cat_groups)
    
    self$seq <- nn_sequential()
    dim_curr <- dim1
    
    for (i in 1:length(params$g_dim)){
      self$seq$add_module(paste0("ResBlock_", i), Residual(dim_curr, params$g_dim[i], params$g_dropout))
      dim_curr <- dim_curr + params$g_dim[i]
    }
    
    if (params$g_dropout > 0){
      self$dropout <- nn_dropout(params$g_dropout / 2)
    }
    self$cond_encoder <- nn_sequential(
      nn_linear(params$ncols - params$nphase2, params$cond_dim),
      nn_batch_norm1d(params$cond_dim),
      nn_leaky_relu(0.2)
    )
    self$num_out <- nn_linear(dim_curr, length(params$num_inds))
    
    for (grp in self$cat_groups) {
      self$cat_delta_heads$append(nn_linear(dim_curr, self$emb_dim))
      self$cat_embedders$append(nn_linear(length(grp), self$emb_dim))
      self$cat_decoders$append(nn_linear(self$emb_dim, length(grp)))
    }
  },
  
  forward = function(N, A, C){
    batch_size <- N$size(1)
    
    cat_emb_list <- list()
    cat_proxy_indices <- c()
    
    for (i in seq_along(self$cat_groups)) {
      grp <- self$cat_groups[[i]]
      a_slice <- A[, grp]
      cat_proxy_indices <- c(cat_proxy_indices, grp)
      emb <- self$cat_embedders[[i]](a_slice)
      cat_emb_list[[i]] <- emb
    }
    
    if (length(cat_emb_list) > 0) {
      cat_embs_flat <- torch_cat(cat_emb_list, dim = 2)
    } else {
      cat_embs_flat <- torch_empty(c(batch_size, 0), device = N$device)
    }
    
    cond_in <- torch_cat(list(A, C), dim = 2)
    cond_feat <- self$cond_encoder(cond_in)
    if (self$params$g_dropout > 0){
      cond_feat <- self$dropout(cond_feat)
    } 
    backbone_input <- torch_cat(list(N, cond_feat, cat_embs_flat), dim = 2)
    features <- self$seq(backbone_input)
    
    output_parts <- list()
    
    if (length(self$num_inds_target) > 0) {
      delta_num <- self$num_out(features)
      output_parts[[1]] <- delta_num
    }
    
    for (i in seq_along(self$cat_groups)) {
      delta_emb <- self$cat_delta_heads[[i]](features)
      target_emb <- cat_emb_list[[i]] + delta_emb
      logits <- self$cat_decoders[[i]](target_emb)
      output_parts[[length(output_parts) + 1]] <- logits
    }
    return(torch_cat(output_parts, dim = 2))
  }
)