generator.mlp <- torch::nn_module(
  "Generator",
  initialize = function(n_g_layers, params, ncols, nphase2, ...){
    dim1 <- params$g_dim + ncols - nphase2
    dim2 <- params$g_dim
    self$seq <- torch::nn_sequential()
    for (i in 1:n_g_layers){
      self$seq$add_module(paste0("Residual_", i), Residual(dim1, dim2))
      dim1 <- dim1 + dim2
    }
    self$seq$add_module("Linear", nn_linear(dim1, nphase2))
  },
  forward = function(input, ...){
    out <- self$seq(input)
    return (out)
  }
)

generator.mmer <- nn_module(
  "Generator",
  initialize = function(n_g_layers, params, ncols, nphase2,
                        num_ind, cat_ind, ...) {

    self$num_ind <- num_ind
    self$cat_ind <- cat_ind

    dim1 <- params$g_dim + ncols - nphase2      # z + fully-observed C
    dim2 <- params$g_dim

    self$trunk <- nn_sequential()
    for (i in 1:n_g_layers) {
      self$trunk$add_module(paste0("Residual_", i), Residual(dim1, dim2))
      dim1 <- dim1 + dim2
    }
    self$head_num <- nn_linear(dim1, length(num_ind)) # Δ numeric
    self$head_cat <- nn_linear(dim1, length(cat_ind)) # Δ logits
  },

  forward = function(input, x_star_num, x_star_cat, ...) {
    h <- self$trunk(input)
    delta_num <- self$head_num(h)
    delta_cat <- self$head_cat(h)

    raw_residual <- torch_cat(list(delta_num, delta_cat), dim = 2)

    x_hat_num <- x_star_num + delta_num
    base_logits <- x_star_cat * 3
    logits_hat <- base_logits + delta_cat

    x_hat <- torch_cat(list(x_hat_num, logits_hat), dim = 2)

    return(list(delta_num, delta_cat, logits_hat, x_hat = x_hat))
  }
)

generator.attn <- torch::nn_module(
  "Generator",
  initialize = function(n_g_layers, params, ncols, nphase2, ...){
    self$params <- params
    dim1 <- params$g_dim + (ncols - nphase2)
    dim2 <- params$g_dim
    self$proj_layer <- torch::nn_sequential(
      nn_linear(dim1, dim2),
      nn_layer_norm(dim2),
      nn_relu()
    )
    
    self$seq <- torch::nn_sequential()
    for (i in 1:n_g_layers){
      self$seq$add_module(paste0("Encoder_", i), nn_transformer_encoder_layer(dim2, nhead = 8,
                                                                              batch_first = T))
    }
    
    self$output_layer <- torch::nn_sequential(
      nn_layer_norm(dim2),
      nn_relu(),
      nn_linear(dim2, nphase2)
    )
  },
  forward = function(input, ...){
    out <- input %>%
      self$proj_layer() %>% 
      torch_unsqueeze(2) %>%
      self$seq() %>%
      torch_squeeze(2) %>%
      self$output_layer()
    return (out)
  }
)