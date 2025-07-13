discriminator.mlp <- torch::nn_module(
  "Discriminator",
  initialize = function(n_d_layers, params, ncols, ...) {
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
  forward = function(input, ...) {
    input <- input$reshape(c(-1, self$pacdim))
    out <- self$seq(input)
    return(out)
  }
)

discriminator.proj <- torch::nn_module(
  "Discriminator",
  initialize = function(n_d_layers, params, ncols, nphase2, ...) {
    self$nphase2 <- nphase2
    self$ncols <- ncols
    self$pacdim_c <- (ncols - nphase2) * params$pac
    self$pacdim <- nphase2 * params$pac
    self$seq <- torch::nn_sequential()
    
    dim <- self$pacdim
    for (i in 1:n_d_layers) {
      self$seq$add_module(paste0("Linear", i), nn_linear(dim, params$d_dim))
      self$seq$add_module(paste0("LeakyReLU", i), nn_leaky_relu(0.2))
      self$seq$add_module(paste0("Dropout", i), nn_dropout(0.5))
      dim <- params$d_dim
    }
    self$seq$add_module("Linear", nn_linear(dim, dim))
    
    self$proj <- nn_sequential(
      nn_linear(self$pacdim_c, params$d_dim),
      nn_leaky_relu(0.2),
      nn_linear(params$d_dim, params$d_dim, bias = F)
    )
  },
  forward = function(input) {
    x <- input[, 1:self$nphase2]
    y <- input[, (self$nphase2 + 1):self$ncols]
    input <- x$reshape(c(-1, self$pacdim))
    out <- self$seq(input)
    h <- self$proj(y$reshape(c(-1, self$pacdim_c)))
    output <- (out * h)$sum(dim = 2, keepdim = T)
    return(out)
  }
)

discriminator.attn <- torch::nn_module(
  "DiscriminatorAttn",
  initialize = function(n_d_layers, params, ncols, nphase2, ...) {
    self$nphase2 <- nphase2
    self$ncols <- ncols
    self$pacdim_c <- (ncols - nphase2) * params$pac
    self$pacdim <- nphase2 * params$pac
    
    self$proj_layer <- torch::nn_sequential(
      nn_linear(self$pacdim, params$d_dim),
      nn_layer_norm(params$d_dim),
      nn_relu(),
    )
    self$proj_layer_c <- torch::nn_sequential(
      nn_linear(self$pacdim_c, params$d_dim),
      nn_layer_norm(params$d_dim),
      nn_relu(),
    )
    
    self$encoder <- Encoder(params$d_dim, num_heads = 8)
    
    self$output_layer <- torch::nn_sequential(
      nn_layer_norm(params$d_dim),
      nn_relu(),
      nn_dropout(0.5),
      nn_linear(params$d_dim, 1)
    )
  },
  forward = function(input) {
    x <- input[, 1:self$nphase2]
    y <- input[, (self$nphase2 + 1):self$ncols]
    x <- x$reshape(c(-1, self$pacdim))
    y <- y$reshape(c(-1, self$pacdim_c))
    
    proj_x <- self$proj_layer(x)
    proj_y <- self$proj_layer_c(y)
    out <- self$encoder(proj_x, proj_y) %>%
      self$output_layer()
    
    return(out)
  }
)