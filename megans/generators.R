generator.mlp <- torch::nn_module(
  "Generator",
  initialize = function(n_g_layers, params, ncols, nphase2){
    dim1 <- params$g_dim + ncols - nphase2
    dim2 <- params$g_dim
    self$seq <- torch::nn_sequential()
    for (i in 1:n_g_layers){
      self$seq$add_module(paste0("Residual_", i), Residual(dim1, dim2))
      dim1 <- dim1 + dim2
    }
    self$seq$add_module("Linear", nn_linear(dim1, nphase2))
  },
  forward = function(input){
    out <- self$seq(input)
    return (out)
  }
)

generator.attn <- torch::nn_module(
  "Generator",
  initialize = function(n_g_layers, params, ncols, nphase2){
    self$params <- params
    dim1 <- params$g_dim + (ncols - nphase2)
    dim2 <- params$g_dim
    self$proj_layer <- torch::nn_sequential(
      nn_linear(dim1, dim2),
      nn_batch_norm1d(dim2),
      nn_relu()
    )
    
    self$seq <- torch::nn_sequential()
    for (i in 1:n_g_layers){
      self$seq$add_module(paste0("Encoder_", i), Encoder(dim2, 8))
    }
    
    self$output_layer <- torch::nn_sequential(
      nn_batch_norm1d(dim2),
      nn_relu(),
      nn_linear(dim2, nphase2)
    )
  },
  forward = function(input){
    out <- input %>% 
      self$proj_layer() %>% 
      self$seq() %>%
      self$output_layer()
    return (out)
  }
)