generator.mlp <- torch::nn_module(
  "Generator",
  initialize = function(n_g_layers, params, ncols, nphase2, nnum, ncat, ...){
    dim1 <- params$noise_dim + ncols - nphase2
    dim2 <- params$g_dim
    self$seq <- torch::nn_sequential()
    for (i in 1:n_g_layers){
      self$seq$add_module(paste0("Residual_", i), Residual(dim1, dim2))
      dim1 <- dim1 + dim2
    }
    
    #self$num_head <- nn_linear(dim1, nnum)
    #self$cat_head <- nn_linear(dim1, ncat)
    self$seq$add_module("Linear", nn_linear(dim1, nphase2))
  },
  forward = function(input, ...){
    output <- self$seq(input)
    #out <- self$seq(input)
    #numout <- self$num_head(out)
    #catout <- self$cat_head(out)
    #output <- torch_cat(list(numout, catout), dim = 2)
    return (output)
  }
)

generator.attn <- torch::nn_module(
  "Generator",
  initialize = function(n_g_layers, params, ncols, nphase2, num_ind, cat_ind){
    self$params <- params
    dim1 <- params$noise_dim + (ncols - nphase2)
    dim2 <- params$g_dim
    self$proj_layer <- torch::nn_sequential(
      nn_linear(dim1, dim2),
      #nn_layer_norm(dim2),
      #nn_relu()
    )
    
    self$seq <- torch::nn_sequential()
    for (i in 1:n_g_layers){
      self$seq$add_module(paste0("Encoder_", i), nn_transformer_encoder_layer(dim2, nhead = 8,
                                                                              batch_first = T))
    }
    
    self$output_layer <- torch::nn_sequential(
      #nn_layer_norm(dim2),
      #nn_relu(),
      nn_linear(dim2, nphase2)
    )
    # self$output_layer_cat <- torch::nn_sequential(
    #   nn_layer_norm(dim2),
    #   nn_relu(),
    #   nn_linear(dim2, length(cat_ind))
    # )
  },
  forward = function(input, ...){
    out <- input %>%
      self$proj_layer() %>% 
      torch_unsqueeze(2) %>%
      self$seq() %>%
      torch_squeeze(2) %>%
      self$output_layer()
    # num <- h %>% self$output_layer_num()
    # cat <- h %>% self$output_layer_cat()
    # out <- torch_cat(list(num, cat), dim = 2)
    return (out)
  }
)