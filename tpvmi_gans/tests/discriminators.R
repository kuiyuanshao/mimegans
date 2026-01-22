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

discriminator.multimlp <- torch::nn_module(
  "Discriminator",
  initialize = function(params, ncols, ...) {
    self$pacdim <- ncols * params$pac
    self$seq <- torch::nn_sequential()
    dim <- self$pacdim
    for (i in 1:params$n_d_layers) {
      self$seq$add_module(paste0("Linear", i), nn_linear(dim, params$d_dim))
      self$seq$add_module(paste0("LeakyReLU", i), nn_leaky_relu(0.2))
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
    
    self$seq_info <- torch::nn_sequential()
    dim <- self$pacdim
    for (i in 1:params$n_d_layers) {
      self$seq_info$add_module(paste0("Linear", i), nn_linear(dim, params$d_dim))
      self$seq_info$add_module(paste0("LeakyReLU", i), nn_leaky_relu(0.2))
      if (i != params$n_d_layers){
        self$seq_info$add_module(paste0("Dropout", i), nn_dropout(0.5))
      }
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

discriminator.sattn <- torch::nn_module(
  "DiscriminatorSAGAN",
  initialize = function(params, ncols, nphase2, ...) {
    n_d_layers <- params$n_d_layers
    head_dim <- if (ncols >= 64) {32} else {16}
    proj_dim <- (ncols %/% head_dim + 1) * head_dim
    self$pacdim <- proj_dim * params$pac
    dim <- self$pacdim
    self$proj_layer <- nn_linear(ncols, proj_dim)
    self$attn <- nn_multihead_attention(proj_dim, 
                                        num_heads = max(1, min(8, round(proj_dim / head_dim))),
                                        batch_first = T)
    
    self$seq <- torch::nn_sequential()
    for (i in 1:n_d_layers) {
      self$seq$add_module(paste0("Linear", i), nn_linear(dim, params$d_dim))
      self$seq$add_module(paste0("LeakyReLU", i), nn_leaky_relu(0.2))
      self$seq$add_module(paste0("Dropout", i), nn_dropout(0.5))
      dim <- params$d_dim
    }
    self$seq$add_module("Linear", nn_linear(dim, 1))
  },
  forward = function(input) {
    input <- self$proj_layer(input)$unsqueeze(2)
    attn_out <- (input + self$attn(input, input, input)[[1]])$squeeze(2)
    attn_out <- attn_out$reshape(c(-1, self$pacdim))
    out <- self$seq(attn_out)
    return (out)
  }
)

discriminator.sagan <- torch::nn_module(
  "DiscriminatorSAGAN",
  initialize = function(params, ncols, nphase2, ...) {
    n_d_layers <- params$n_d_layers
    self$pacdim <- ncols * params$pac
    dim <- self$pacdim

    self$seq <- torch::nn_sequential()
    for (i in 1:(n_d_layers - 1)) {
      self$seq$add_module(paste0("Linear", i), nn_linear(dim, params$d_dim))
      self$seq$add_module(paste0("LeakyReLU", i), nn_leaky_relu(0.2))
      self$seq$add_module(paste0("Dropout", i), nn_dropout(0.5))
      dim <- params$d_dim
    }
    self$attn <- nn_multihead_attention(dim, num_heads = max(1, min(8, round(params$d_dim / 64))),
                                        batch_first = T)
    self$dropout <- nn_dropout(0.5)
    self$gamma <- nn_parameter(torch_tensor(0))
    self$out_seq <- torch::nn_sequential(nn_linear(dim, params$d_dim),
                                         nn_leaky_relu(0.2),
                                         nn_dropout(0.5))
    
    self$out_seq$add_module("OutLinear", nn_linear(dim, 1))
  },
  forward = function(input) {
    input <- input$reshape(c(-1, self$pacdim))
    hidden <- self$seq(input)$unsqueeze(2)
    attn_out <- (hidden + self$gamma * self$dropout(self$attn(hidden, hidden, hidden)[[1]]))$squeeze(2)
    out <- self$out_seq(attn_out)
    return (out)
  }
)


discriminator.infosagan <- torch::nn_module(
  "DiscriminatorSAGAN",
  initialize = function(params, ncols, nphase2, ...) {
    n_d_layers <- params$n_d_layers
    self$pacdim <- ncols * params$pac
    dim <- self$pacdim
    
    self$seq <- torch::nn_sequential()
    for (i in 1:(n_d_layers - 1)) {
      self$seq$add_module(paste0("Linear", i), nn_linear(dim, params$d_dim))
      self$seq$add_module(paste0("LeakyReLU", i), nn_leaky_relu(0.2))
      self$seq$add_module(paste0("Dropout", i), nn_dropout(0.5))
      dim <- params$d_dim
    }
    self$attn <- nn_multihead_attention(dim, num_heads = max(1, min(8, round(params$d_dim / 64))),
                                        batch_first = T)
    self$gamma <- nn_parameter(torch_tensor(0))
    self$out_seq <- torch::nn_sequential(nn_linear(dim, params$d_dim),
                                         nn_leaky_relu(0.2),
                                         nn_dropout(0.5))
    
    self$out_seq$add_module("OutLinear", nn_linear(dim, 1))
    
    dim <- self$pacdim
    self$seq_info <- torch::nn_sequential()
    for (i in 1:(n_d_layers - 1)) {
      self$seq_info$add_module(paste0("Linear", i), nn_linear(dim, params$d_dim))
      self$seq_info$add_module(paste0("LeakyReLU", i), nn_leaky_relu(0.2))
      self$seq_info$add_module(paste0("Dropout", i), nn_dropout(0.5))
      dim <- params$d_dim
    }
    self$attn_info <- nn_multihead_attention(dim, num_heads = max(1, min(8, round(params$d_dim / 64))),
                                             batch_first = T)
    self$out_seq_info <- torch::nn_sequential(nn_linear(dim, params$d_dim),
                                              nn_leaky_relu(0.2))
  },
  forward = function(input) {
    input <- input$reshape(c(-1, self$pacdim))
    hidden <- self$seq(input)$unsqueeze(2)
    attn_out <- (hidden + self$gamma * self$attn(hidden, hidden, hidden)[[1]])$squeeze(2)
    out <- self$out_seq(attn_out)
    
    hidden_info <- self$seq_info(input)$unsqueeze(2)
    attn_out_info <- (hidden_info + self$gamma$item() * self$attn_info(hidden_info, hidden_info, hidden_info)[[1]])$squeeze(2)
    out_info <- self$out_seq_info(attn_out_info)
    return (list(out, out_info))
  }
)