discriminator.mlp <- torch::nn_module(
  "Discriminator",
  initialize = function(n_d_layers, params, ncols) {
    self$pacdim <- ncols * params$pac
    self$seq <- torch::nn_sequential()
    
    dim <- self$pacdim
    for (i in 1:n_d_layers) {
      self$seq$add_module(paste0("Linear", i), nn_linear(dim, params$d_dim))
      self$seq$add_module(paste0("LeakyReLU", i), nn_leaky_relu(0.2))
      self$seq$add_module(paste0("Dropout", i), nn_dropout(0.5))
      dim <- params$d_dim
    }
    self$seq$add_module("Linear", nn_linear(dim, 1))
    
  },
  forward = function(input) {
    input <- input$reshape(c(-1, self$pacdim))
    out <- self$seq(input)
    return(out)
  }
)

discriminator.attn <- torch::nn_module(
  "DiscriminatorAttn",
  initialize = function(n_d_layers, params, ncols) {
    self$pacdim <- ncols * params$pac
    
    self$proj_layer <- torch::nn_sequential(
      nn_linear(self$pacdim, params$d_dim),
      nn_layer_norm(params$d_dim),
      nn_relu(),
    )
    
    self$seq <- torch::nn_sequential()
    for (i in 1:n_d_layers) {
      self$seq$add_module(paste0("Encoder_", i), nn_transformer_encoder_layer(params$d_dim, nhead = 8,
                                                                              batch_first = T))
    }

    self$output_layer <- torch::nn_sequential(
      nn_layer_norm(params$d_dim),
      nn_relu(),
      nn_dropout(0.5),
      nn_linear(params$d_dim, 1)
    )
  },
  forward = function(input) {
    input <- input$reshape(c(-1, self$pacdim))
    out <- input %>%
      self$proj_layer() %>%
      torch_unsqueeze(2) %>%
      self$seq() %>%
      torch_squeeze(2) %>%
      self$output_layer()
    
    return(out)
  }
)