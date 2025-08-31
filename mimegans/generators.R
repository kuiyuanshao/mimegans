generator.mlp <- torch::nn_module(
  "Generator",
  initialize = function(params, ncols, nphase2, rate, ...){
    dim1 <- params$noise_dim + ncols - nphase2
    dim2 <- params$g_dim
    self$input_drop <- nn_dropout(0.25)
    self$seq <- torch::nn_sequential()
    for (i in 1:params$n_g_layers){
      self$seq$add_module(paste0("Residual_", i), Residual(dim1, dim2, rate))
      dim1 <- dim1 + dim2
    }
    self$seq$add_module("Linear", nn_linear(dim1, nphase2))
    
  },
  forward = function(input, ...){
    output <- self$seq(self$input_drop(input))
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
      self$seq$add_module(paste0("Residual_", i), Residual(dim1, dim2, rate = rate))
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

generator.sagan <- torch::nn_module(
  "Generator",
  initialize = function(params, ncols, nphase2, rate, ...){
    dim1 <- params$noise_dim + ncols - nphase2
    dim2 <- params$g_dim
    
    self$input_drop <- nn_dropout(0.25)
    self$proj_layer <- nn_sequential(nn_linear(dim1, dim2),
                                     nn_batch_norm1d(dim2),
                                     nn_elu(),
                                     nn_dropout(0.25))
                                     
    dim1 <- dim2
    self$seq <- torch::nn_sequential()
    for (i in 1:(params$n_g_layers - 1)){
      self$seq$add_module(paste0("Residual_", i), Residual(dim1, dim2, rate, "add"))
      # dim1 <- dim1 + dim2
    }
    self$attn <- nn_multihead_attention(dim1, max(1, min(8, round(dim1 / 64))), 
                                        batch_first = T)
    self$dropout <- nn_dropout(0.5)
    self$gamma <- nn_parameter(torch_tensor(0))
    self$post_seq <- torch::nn_sequential(Residual(dim1, dim2, rate, "add"),
                                          nn_linear(dim1, nphase2))
    
  },
  forward = function(input, ...){
    out1 <- self$seq(self$proj_layer(self$input_drop(input)))$unsqueeze(2)
    attn_score <- (out1 + self$gamma * self$dropout(self$attn(out1, out1, out1)[[1]]))$squeeze(2)
    out <- self$post_seq(attn_score)
    return (out)
  }
)