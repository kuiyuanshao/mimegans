discriminator.mlp <- torch::nn_module(
  "Discriminator",
  initialize = function(params, ncols, ...) {
    self$params <- params
    self$pacdim <- ncols * params$pac
    self$seq <- torch::nn_sequential()
    dim <- self$pacdim
    for (i in 1:params$n_d_layers) {
      self$seq$add_module(paste0("Linear", i), nn_linear(dim, params$d_dim))
      self$seq$add_module(paste0("LeakyReLU", i), nn_leaky_relu(0.2))
      self$seq$add_module(paste0("Dropout", i), nn_dropout(0.5))
      dim <- params$d_dim
    }
    self$seq$add_module("Linear", nn_linear(dim, 1))
    
    if (params$info_loss){
      self$seq_info <- torch::nn_sequential()
      dim <- self$pacdim
      for (i in 1:params$n_d_layers) {
        self$seq_info$add_module(paste0("Linear_info", i), nn_linear(dim, params$d_dim))
        self$seq_info$add_module(paste0("LeakyReLU_info", i), nn_leaky_relu(0.2))
        if (i != params$n_d_layers){
          self$seq_info$add_module(paste0("Dropout", i), nn_dropout(0.5))
        }
        dim <- params$d_dim
      }
    }
    
  },
  forward = function(input, ...) {
    input <- input$reshape(c(-1, self$pacdim))
    out <- self$seq(input)
    if (self$params$info_loss){
      info_out <- self$seq_info(input)
      return (list(out, info_out))
    }else{
      return (list(out))
    }
  }
)