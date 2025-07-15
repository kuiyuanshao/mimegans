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
    proj_dim <- ((nphase2 + 3) %/% 64) * 64
    self$pacdim <- proj_dim * params$pac
    
    self$proj_layer <- nn_linear(nphase2, proj_dim)
    self$proj_layer_c <- nn_linear((ncols - nphase2), proj_dim)
    # self$attn1 <- nn_multihead_attention(proj_dim, num_heads = max(1, min(8, round(proj_dim / 64))))
    # self$attn2 <- nn_multihead_attention(proj_dim, num_heads = max(1, min(8, round(proj_dim / 64))))
    # self$norm1 <- nn_layer_norm(proj_dim)
    # self$norm2 <- nn_layer_norm(proj_dim)
    # nn_init_xavier_uniform_(self$proj_layer$weight)
    # nn_init_xavier_uniform_(self$proj_layer_c$weight)
    # self$proj_layer <- nn_sequential(nn_linear(nphase2, proj_dim),
    #                                  nn_layer_norm(proj_dim))
    # self$proj_layer_c <- nn_sequential(nn_linear((ncols - nphase2), proj_dim),
    #                                    nn_layer_norm(proj_dim))
    #self$proj_layer_c$weight$requires_grad_(FALSE)
    
    self$encoder <- Encoder(proj_dim, num_heads = max(1, min(8, round(proj_dim / 64))))
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
    x <- input[, 1:self$nphase2]
    y <- input[, (self$nphase2 + 1):self$ncols]
    proj_x <- self$proj_layer(x)
    proj_y <- self$proj_layer_c(y)
    # attn_x <- self$attn1(proj_x$unsqueeze(2), 
    #                      proj_x$unsqueeze(2), 
    #                      proj_x$unsqueeze(2))[[1]]$squeeze(2)
    # norm_x <- self$norm1(proj_x + attn_x)
    # attn_y <- self$attn1(proj_y$unsqueeze(2), 
    #                      proj_y$unsqueeze(2), 
    #                      proj_y$unsqueeze(2))[[1]]$squeeze(2)
    # norm_y <- self$norm2(proj_y + attn_y)
    attn_out <- self$encoder(proj_x, proj_y)
    attn_out <- attn_out$reshape(c(-1, self$pacdim))
    out <- self$seq(attn_out)
    return(out)
  }
)