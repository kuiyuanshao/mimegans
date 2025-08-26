generator.mlp <- torch::nn_module(
  "Generator",
  initialize = function(params, ncols, nphase2, rate, ...){
    dim1 <- params$noise_dim + ncols - nphase2
    dim2 <- params$g_dim
    self$seq <- torch::nn_sequential()
    for (i in 1:params$n_g_layers){
      self$seq$add_module(paste0("Residual_", i), Residual(dim1, dim2, rate))
      dim1 <- dim1 + dim2
    }
    self$seq$add_module("Linear", nn_linear(dim1, nphase2))
    
  },
  forward = function(input, ...){
    output <- self$seq(input)
    return (output)
  }
)

generator.seqmlp <- torch::nn_module(
  "Generator",
  initialize = function(params, ncols, nphase2, rate, dimensions){
    dim1 <- params$noise_dim + ncols - nphase2
    dim2 <- params$g_dim
    self$seq <- torch::nn_sequential()
    for (i in 1:params$n_g_layers){
      self$seq$add_module(paste0("Residual_", i), Residual(dim1, dim2, rate = 0.25))
      dim1 <- dim1 + dim2
    }
    self$out <- nn_module_list()
    for (i in 1:length(dimensions)){
      self$out$append(nn_linear(dim1, dimensions[i]))
      dim1 <- dim1 + dimensions[i]
    }
    
  },
  forward = function(input, ...){
    hidden_output <- self$seq(input)
    
    outs <- vector("list", length(self$out))
    for (i in seq_along(outs)){
      outs[[i]] <- self$out[[i]](hidden_output)
      hidden_output <- torch_cat(list(outs[[i]], hidden_output), dim = 2)
    }
    return (torch_cat(outs, dim = 2))
  }
)

generator.mcmlp <- torch::nn_module(
  "Generator",
  initialize = function(params, ncols, nphase2, rate, dimensions){
    dim1 <- params$noise_dim + ncols - nphase2
    dim2 <- params$g_dim
    self$seq <- torch::nn_sequential()
    for (i in 1:params$n_g_layers){
      self$seq$add_module(paste0("Residual_", i), Residual(dim1, dim2, rate = 0.25))
      dim1 <- dim1 + dim2
    }
    self$out <- nn_module_list()
    for (i in 1:length(dimensions)){
      self$out$append(nn_linear(dim1, dimensions[i]))
    }
    
  },
  forward = function(input, ...){
    hidden_output <- self$seq(input)
    
    outs <- vector("list", length(self$out))
    for (i in seq_along(outs)){
      outs[[i]] <- self$out[[i]](hidden_output)
    }
    return (torch_cat(outs, dim = 2))
  }
)

generator.sattn <- torch::nn_module(
  "Generator",
  initialize = function(params, ncols, nphase2, rate, ...){
    dim1 <- params$noise_dim + ncols - nphase2
    dim2 <- params$g_dim
    
    self$proj_layer <- nn_linear(dim1, dim2)
    dim1 <- dim2
    self$seq <- torch::nn_sequential()
    for (i in 1:(params$n_g_layers - 1)){
      self$seq$add_module(paste0("Residual_", i), 
                          Residual(dim1, dim2, rate, resid = "add"))
      # dim1 <- dim1 + dim2
    }
    self$attn <- nn_multihead_attention(dim1, max(1, min(8, round(dim1 / 64))), 
                                        batch_first = T, dropout = rate)
    self$norm <- nn_layer_norm(dim1)
    
    self$post_seq <- torch::nn_sequential()
    self$post_seq$add_module(paste0("Residual_", params$n_g_layers), 
                             Residual(dim1, dim2, rate))
    dim1 <- dim1 + dim2
    self$post_seq$add_module("Linear", nn_linear(dim1, nphase2))
    
  },
  forward = function(input, ...){
    out1 <- self$seq(self$proj_layer(input))$unsqueeze(2)
    attn_score <- self$norm(out1 + self$attn(out1, out1, out1)[[1]])$squeeze(2)
    out <- self$post_seq(attn_score)
    return (out)
  }
)