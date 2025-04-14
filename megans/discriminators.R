discriminator.mlp <- torch::nn_module(
  "Discriminator",
  initialize = function(n_d_layers, params, ncols, cat_inds) {
    self$pacdim <- ncols * params$pac
    self$seq <- torch::nn_sequential()
    
    dim <- self$pacdim
    for (i in 1:n_d_layers) {
      self$seq$add_module(paste0("Linear", i), nn_linear(dim, dim))
      self$seq$add_module(paste0("LeakyReLU", i), nn_leaky_relu(0.2))
      self$seq$add_module(paste0("Dropout", i), nn_dropout(0.25))
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
  initialize = function(n_d_layers, params, ncols, cat_inds) {
    self$ncols <- ncols
    self$cat_inds <- cat_inds
    self$num_inds <- which(!(1:ncols %in% cat_inds))
    self$tokenizer <- Tokenizer(ncols, cat_inds, params$bias_token, params$d_token)
    
    self$pacdim <- self$tokenizer$d_token * (ncols + 1) * params$pac
    
    self$proj_layer <- torch::nn_sequential(
      nn_linear(self$pacdim, params$d_dim[2]),
      nn_layer_norm(params$d_dim[2]),
      nn_relu()
    )
    
    self$seq <- torch::nn_sequential()
    for (i in 1:n_d_layers) {
      self$seq$add_module(paste0("Encoder_", i), Encoder(params$d_dim[2], 8))
    }

    self$output_layer <- torch::nn_sequential(
      nn_layer_norm(params$d_dim[2]),
      nn_relu(),
      nn_linear(params$d_dim[2], 1)
    )
  },
  forward = function(input) {
    input <- self$tokenizer(input[, self$num_inds, drop = F], 
                            input[, self$cat_inds, drop = F])
    input <- input$reshape(c(input$size(1), input$size(2) * input$size(3)))
    input <- input$reshape(c(-1, self$pacdim))
    out <- input %>%
      self$proj_layer() %>%
      self$seq() %>%
      self$output_layer()
    
    return(out)
  }
)