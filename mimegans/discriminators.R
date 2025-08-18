discriminator.mlp <- torch::nn_module(
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
    return(out)
  }
)

discriminator.sattn <- torch::nn_module(
  "DiscriminatorEncoder",
  initialize = function(params, ncols, nphase2, ...) {
    n_d_layers <- params$n_d_layers
    self$pacdim <- ncols * params$pac
    dim <- self$pacdim
    
    self$attn <- nn_multihead_attention(params$d_dim, 
                                        num_heads = max(1, min(8, round(params$d_dim / 128))),
                                        batch_first = T)
    for(i in seq_along(self$attn)) {
      layer <- self$attn[[i]]
      if (inherits(layer, "nn_linear")) {
        self$attn[[i]] <- spectral_norm(layer)
      }
    }
    self$gamma <- nn_parameter(torch_tensor(0))
    self$seq1 <- torch::nn_sequential()
    for (i in 1:(n_d_layers - 1)) {
      self$seq1$add_module(paste0("Linear", i), spectral_norm(nn_linear(dim, params$d_dim)))
      self$seq1$add_module(paste0("LeakyReLU", i), nn_leaky_relu(0.2))
      dim <- params$d_dim
    }
    self$seq2 <- torch::nn_sequential(
      spectral_norm(nn_linear(params$d_dim, params$d_dim)),
      nn_leaky_relu(0.2)
    )
    self$seq2$add_module("Linear", spectral_norm(nn_linear(dim, 1)))
  },
  forward = function(input) {
    # input <- self$proj_layer(input)$unsqueeze(2)
    input <- input$reshape(c(-1, self$pacdim))
    input <- self$seq1(input)$unsqueeze(2)
    attn_out <- (input + self$gamma * self$attn(input, input, input)[[1]])$squeeze(2)
    # attn_out <- attn_out$reshape(c(-1, self$pacdim))
    out <- self$seq2(attn_out)
    return (out)
  }
)
