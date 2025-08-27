discriminator.mlp <- torch::nn_module(
  "Discriminator",
  initialize = function(params, ncols, ...) {
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
    
  },
  forward = function(input, ...) {
    input <- input$reshape(c(-1, self$pacdim))
    out <- self$seq(input)
    return (out)
  }
)

discriminator.infomlp <- torch::nn_module(
  "Discriminator",
  initialize = function(params, ncols, ...) {
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
    
    self$seq_info <- torch::nn_sequential(nn_linear(self$pacdim, params$d_dim),
                                          nn_leaky_relu(0.2))
  },
  forward = function(input, ...) {
    input <- input$reshape(c(-1, self$pacdim))
    out <- self$seq(input)
    info_out <- self$seq_info(input)
    return (list(out, info_out))
  }
)

discriminator.snmlp <- torch::nn_module(
  "Discriminator",
  initialize = function(params, ncols, ...) {
    self$pacdim <- ncols * params$pac
    self$seq <- torch::nn_sequential()
    
    dim <- self$pacdim
    for (i in 1:params$n_d_layers) {
      self$seq$add_module(paste0("Linear", i), spectral_norm(nn_linear(dim, params$d_dim)))
      self$seq$add_module(paste0("LeakyReLU", i), nn_leaky_relu(0.2))
      dim <- params$d_dim
    }
    self$seq$add_module("Linear", spectral_norm(nn_linear(dim, 1)))
    
  },
  forward = function(input, ...) {
    input <- input$reshape(c(-1, self$pacdim))
    out <- self$seq(input)
    return (out)
  }
)

discriminator.sninfomlp <- torch::nn_module(
  "Discriminator",
  initialize = function(params, ncols, ...) {
    self$pacdim <- ncols * params$pac
    self$seq <- torch::nn_sequential()
    
    dim <- self$pacdim
    for (i in 1:params$n_d_layers) {
      self$seq$add_module(paste0("Linear", i), spectral_norm(nn_linear(dim, params$d_dim)))
      self$seq$add_module(paste0("LeakyReLU", i), nn_leaky_relu(0.2))
      dim <- params$d_dim
    }
    self$seq$add_module("Linear", spectral_norm(nn_linear(dim, 1)))
    
    self$seq_info <- torch::nn_sequential()
    dim <- self$pacdim
    for (i in 1:(params$n_d_layers - 1)) {
      self$seq_info$add_module(paste0("Linear_info", i), spectral_norm(nn_linear(dim, params$d_dim)))
      self$seq_info$add_module(paste0("LeakyReLU_info", i), nn_leaky_relu(0.2))
      dim <- params$d_dim
    }
    
  },
  forward = function(input, ...) {
    input <- input$reshape(c(-1, self$pacdim))
    out <- self$seq(input)
    info_out <- self$seq_info(input)
    return (list(out, info_out))
  }
)


discriminator.sattn <- torch::nn_module(
  "DiscriminatorSAttn",
  initialize = function(params, ncols, nphase2, ...) {
    n_d_layers <- params$n_d_layers
    head_dim <- if (ncols >= 64) {32} else {16}
    proj_dim <- (ncols %/% head_dim + 1) * head_dim
    self$pacdim <- proj_dim * params$pac
    dim <- self$pacdim
    self$proj_layer <- spectral_norm(nn_linear(ncols, proj_dim))
    self$attn <- nn_multihead_attention(proj_dim, 
                                        num_heads = max(1, min(8, round(proj_dim / head_dim))),
                                        batch_first = T)
    for(i in seq_along(self$attn)) {
      layer <- self$attn[[i]]
      if (inherits(layer, "nn_linear")) {
        self$attn[[i]] <- spectral_norm(layer)
      }
    }
    
    self$seq <- torch::nn_sequential()
    for (i in 1:n_d_layers) {
      self$seq$add_module(paste0("Linear", i), spectral_norm(nn_linear(dim, params$d_dim)))
      self$seq$add_module(paste0("LeakyReLU", i), nn_leaky_relu(0.2))
      dim <- params$d_dim
    }
    self$seq$add_module("Linear", spectral_norm(nn_linear(dim, 1)))
  },
  forward = function(input) {
    input <- self$proj_layer(input)$unsqueeze(2)
    attn_out <- (input + self$attn(input, input, input)[[1]])$squeeze(2)
    attn_out <- attn_out$reshape(c(-1, self$pacdim))
    out <- self$seq(attn_out)
    return (out)
  }
)


discriminator.cattn <- torch::nn_module(
  "DiscriminatorCAttn",
  initialize = function(params, ncols, nphase2, ...) {
    self$nphase2 <- nphase2
    self$ncols <- ncols
    head_dim_target <- if (nphase2 >= 64) 32 else 16
    proj_dim <- ((ncols - nphase2) %/% head_dim_target + 1) * head_dim_target
    self$pacdim <- proj_dim * params$pac
    self$proj_layer <- spectral_norm(nn_linear(nphase2, proj_dim))
    self$proj_layer_c <- spectral_norm(nn_linear((ncols - nphase2), proj_dim))
    
    self$attn <- nn_multihead_attention(proj_dim, 
                                        num_heads = max(1, min(8, round(proj_dim / head_dim_target))),
                                        batch_first = T) 
    self$dropout <- nn_dropout()
    self$norm <- nn_layer_norm(proj_dim)
    
    self$seq <- torch::nn_sequential()
    dim <- self$pacdim
    for (i in 1:params$n_d_layers) {
      self$seq$add_module(paste0("Linear", i), spectral_norm(nn_linear(dim, params$d_dim)))
      self$seq$add_module(paste0("LeakyReLU", i), nn_leaky_relu(0.2))
      dim <- params$d_dim
    }
    self$seq$add_module("Linear", spectral_norm(nn_linear(dim, 1)))
  },
  forward = function(input) {
    x <- input[, 1:self$nphase2]
    y <- input[, (self$nphase2 + 1):self$ncols]
    
    proj_x <- self$proj_layer(x)$unsqueeze(2)
    proj_y <- self$proj_layer_c(y)$unsqueeze(2)
    attn_out <- self$norm(proj_x + self$dropout(self$attn(proj_x, proj_y, proj_y)[[1]]))$squeeze(2)
    attn_out <- attn_out$reshape(c(-1, self$pacdim))
    out <- self$seq(attn_out)
    return(out)
  }
)
