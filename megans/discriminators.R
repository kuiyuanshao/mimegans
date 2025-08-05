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

    head_dim_target <- if (nphase2 >= 64) 32 else 16
    proj_dim <- ((nphase2 + 3) %/% head_dim_target + 1) * head_dim_target
    self$pacdim <- proj_dim * params$pac
    if (params$sn_d){
      self$proj_layer <- spectral_norm(nn_linear(nphase2, proj_dim))
      self$proj_layer_c <- spectral_norm(nn_linear((ncols - nphase2), proj_dim))
    }else{
      self$proj_layer <- nn_linear(nphase2, proj_dim)
      self$proj_layer_c <- nn_linear((ncols - nphase2), proj_dim)
    }

    self$attn1 <- nn_multihead_attention(proj_dim, max(1, min(8, round(proj_dim / head_dim_target))),
                                        batch_first = T)
    self$attn2 <- nn_multihead_attention(proj_dim, max(1, min(8, round(proj_dim / head_dim_target))),
                                         batch_first = T)
    self$gamma_attn <- nn_parameter(torch_tensor(0))
    self$seq <- torch::nn_sequential()
    dim <- self$pacdim
    for (i in 1:n_d_layers) {
      if (params$sn_d){
        self$seq$add_module(paste0("Linear", i), spectral_norm(nn_linear(dim, params$d_dim)))
      }else{
        self$seq$add_module(paste0("Linear", i), nn_linear(dim, params$d_dim))
      }
      self$seq$add_module(paste0("LeakyReLU", i), nn_leaky_relu(0.2))
      self$seq$add_module(paste0("Dropout", i), nn_dropout(0.5))
      dim <- params$d_dim
    }
    self$seq$add_module("Linear", nn_linear(dim, 1))

    if (params$sn_d){
      for(i in seq_along(self$attn)) {
        layer <- self$attn[[i]]
        if (inherits(layer, "nn_linear")) {
          self$attn[[i]] <- spectral_norm(layer)
        }
      }
    }
  },
  crossattn = function(x, y){
    proj_x <- self$proj_layer(x)$unsqueeze(2)
    proj_y <- self$proj_layer_c(y)$unsqueeze(2)
    crossattn_score <- proj_x + self$attn1(proj_x, proj_y, proj_y)[[1]]
    selfattn_score <- crossattn_score + self$attn2(crossattn_score, crossattn_score, crossattn_score)[[1]]
    
    return (selfattn_score)
  },
  head_forward = function(attn_score){
    attn_score <- attn_score$reshape(c(-1, self$pacdim))
    out <- self$seq(attn_score)
    return (out)
  },
  forward = function(input) {
    x <- input[, 1:self$nphase2]
    y <- input[, (self$nphase2 + 1):self$ncols]

    attn_score <- self$crossattn(x, y)
    attn_score <- attn_score$reshape(c(-1, self$pacdim))
    out <- self$seq(attn_score)
    return(out)
  }
)

# discriminator.attn <- torch::nn_module(
#   "DiscriminatorAttn",
#   initialize = function(n_d_layers, params, ncols, nphase2, ...) {
#     self$nphase2 <- nphase2
#     self$ncols <- ncols
#     
#     head_dim_target <- if (nphase2 >= 64) 32 else 16
#     proj_dim <- ((ncols + 3) %/% head_dim_target + 1) * head_dim_target
#     self$pacdim <- proj_dim * params$pac
#     self$proj_layer <- nn_linear(ncols, proj_dim)
#     
#     self$attn <- nn_multihead_attention(proj_dim, max(1, min(8, round(proj_dim / head_dim_target))), 
#                                         batch_first = T)
#     self$gamma_attn <- nn_parameter(torch_tensor(0))
#     self$seq <- torch::nn_sequential()
#     dim <- self$pacdim
#     for (i in 1:n_d_layers) {
#       if (params$sn_d){
#         self$seq$add_module(paste0("Linear", i), spectral_norm(nn_linear(dim, params$d_dim)))
#       }else{
#         self$seq$add_module(paste0("Linear", i), nn_linear(dim, params$d_dim))
#       }
#       self$seq$add_module(paste0("LeakyReLU", i), nn_leaky_relu(0.2))
#       self$seq$add_module(paste0("Dropout", i), nn_dropout(0.5))
#       dim <- params$d_dim
#     }
#     self$seq$add_module("Linear", nn_linear(dim, 1))
#     
#     if (params$sn_d){
#       for(i in seq_along(self$attn)) {
#         layer <- self$attn[[i]]
#         if (inherits(layer, "nn_linear")) {
#           self$attn[[i]] <- spectral_norm(layer)
#         }
#       }
#     }
#   },
#   selfattn = function(input){
#     proj_in <- self$proj_layer(input)$unsqueeze(2)
#     attn_score <- self$attn(proj_in, proj_in, proj_in)[[1]]
#     attn_res <- (proj_in + self$gamma_attn * attn_score)$squeeze(2)
#     return (attn_res)
#   },
#   head_forward = function(attn_score){
#     attn_score <- attn_score$reshape(c(-1, self$pacdim))
#     out <- self$seq(attn_score)
#     return (out)
#   },
#   forward = function(input) {
#     attn_score <- self$selfattn(input)
#     attn_score <- attn_score$reshape(c(-1, self$pacdim))
#     out <- self$seq(attn_score)
#     return(out)
#   }
# )

# discriminator.encoder <- torch::nn_module(
#   "DiscriminatorEncoder",
#   initialize = function(n_d_layers, params, ncols, nphase2, ...) {
#     self$nphase2 <- nphase2
#     self$ncols <- ncols
#     head_dim_target <- if (nphase2 >= 64) 32 else 16
#     proj_dim <- ((nphase2 + 3) %/% head_dim_target) * head_dim_target
#     self$pacdim <- proj_dim * params$pac
#     self$proj_layer <- nn_linear(nphase2, proj_dim)
#     self$proj_layer_c <- nn_linear((ncols - nphase2), proj_dim)
# 
#     self$encoder <- Encoder(proj_dim, num_heads = max(1, min(8, round(proj_dim / head_dim_target))))
#     self$gamma_attn <- self$encoder$gamma_attn # nn_parameter(torch_tensor(0.1))
# 
#     self$seq <- torch::nn_sequential()
#     dim <- self$pacdim
#     for (i in 1:n_d_layers) {
#       self$seq$add_module(paste0("Linear", i), nn_linear(dim, params$d_dim))
#       self$seq$add_module(paste0("LeakyReLU", i), nn_leaky_relu(0.2))
#       self$seq$add_module(paste0("Dropout", i), nn_dropout(0.5))
#       dim <- params$d_dim
#     }
#     self$seq$add_module("Linear", nn_linear(dim, 1))
#   },
#   encode = function(x, y){
#     proj_x <- self$proj_layer(x)
#     proj_y <- self$proj_layer_c(y)
#     encode_out <- self$encoder(proj_x, proj_y) # * self$gamma_attn
# 
#     # denom <- encode_out$norm(p = 2, dim = 2, keepdim = TRUE)$detach() + 1e-8
#     return (encode_out) #  / denom)
#   },
#   head_forward = function(encode_out){
#     encode_out <- encode_out$reshape(c(-1, self$pacdim))
#     out <- self$seq(encode_out)
#     return (out)
#   },
#   forward = function(input) {
#     x <- input[, 1:self$nphase2]
#     y <- input[, (self$nphase2 + 1):self$ncols]
# 
#     encode_out <- self$encode(x, y)
#     encode_out <- encode_out$reshape(c(-1, self$pacdim))
#     out <- self$seq(encode_out)
#     return(out)
#   }
# )

discriminator.encoder <- torch::nn_module(
  "DiscriminatorEncoder",
  initialize = function(n_d_layers, params, ncols, nphase2, ...) {
    self$nphase2 <- nphase2
    self$ncols <- ncols
    head_dim_target <- if (ncols >= 64) 32 else 16
    proj_dim <- ((ncols + 3) %/% head_dim_target + 1) * head_dim_target
    self$pacdim <- proj_dim * params$pac
    self$proj_layer <- nn_linear(ncols, proj_dim)
    self$encoder <- Encoder(proj_dim, num_heads = max(1, min(8, round(proj_dim / head_dim_target))))
    self$gamma_attn <- nn_parameter(torch_tensor(0.1))

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
  encode = function(input){
    input <- self$proj_layer(input)
    encode_out <- self$encoder(input, input) # * self$gamma_attn

    denom <- encode_out$norm(p = 2, dim = 2, keepdim = TRUE)$detach() + 1e-8
    return (encode_out / denom)
  },
  head_forward = function(encode_out){
    encode_out <- encode_out$reshape(c(-1, self$pacdim))
    out <- self$seq(encode_out)
    return (out)
  },
  forward = function(input) {
    encode_out <- self$encode(input = input)
    encode_out <- encode_out$reshape(c(-1, self$pacdim))
    out <- self$seq(encode_out)
    return(out)
  }
)