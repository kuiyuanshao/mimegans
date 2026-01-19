discriminator.mlp <- torch::nn_module(
  "Discriminator",
  initialize = function(params, ...) {
    self$params <- params
    self$pacdim <- params$ncols * params$pac
    self$seq <- torch::nn_sequential()
    dim <- self$pacdim
    for (i in 1:length(params$d_dim)) {
      self$seq$add_module(paste0("Linear", i), nn_linear(dim, params$d_dim[i]))
      self$seq$add_module(paste0("LeakyReLU", i), nn_leaky_relu(0.2))
      self$seq$add_module(paste0("Dropout", i), nn_dropout(0.5))
      dim <- params$d_dim[i]
    }
    self$seq$add_module("Linear", nn_linear(dim, 1))
  },
  forward = function(input, ...) {
    input <- input$reshape(c(-1, self$pacdim))
    out <- self$seq(input)
    return (out)
  }
)
