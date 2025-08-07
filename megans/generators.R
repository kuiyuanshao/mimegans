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

generator.attn <- torch::nn_module(
  "Generator",
  initialize = function(n_g_layers, params, ncols, nphase2, rate){
    dim1 <- params$noise_dim + ncols - nphase2
    dim2 <- params$g_dim

    self$seq <- torch::nn_sequential()
    for (i in 1:n_g_layers){
      self$seq$add_module(paste0("Residual_", i), 
                          Residual(dim1, dim2, rate))
      dim1 <- dim1 + dim2
    }
    self$attn <- nn_multihead_attention(dim1, max(1, min(8, round(dim1 / 256))), 
                                        batch_first = T, dropout = rate)
    self$gamma_attn <- nn_parameter(torch_tensor(0))
    
    self$post_seq <- torch::nn_sequential()
    self$post_seq$add_module(paste0("Residual_", n_g_layers + 1), 
                             Residual(dim1, dim2, rate, params$sn_g))
    dim1 <- dim1 + dim2
    self$post_seq$add_module("Linear", nn_linear(dim1, nphase2))
    
  },
  forward = function(input, ...){
    out1 <- self$seq(input)$unsqueeze(2)
    attn_score <- (out1 + self$gamma_attn * 
      self$attn(out1, out1, out1)[[1]])$squeeze(2)
    out <- self$post_seq(attn_score)
    return (out)
  }
)