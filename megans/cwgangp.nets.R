Residual <- torch::nn_module(
  "Residual",
  initialize = function(dim1, dim2){
    self$linear <- nn_linear(dim1, dim2)
    self$bn <- nn_batch_norm1d(dim2)
    self$relu <- nn_relu()
  },
  forward = function(input){
    output <- self$linear(input)
    output <- self$bn(output)
    output <- self$relu(output)
    return (torch_cat(list(output, input), dim = 2))
  }
)


Tokenizer <- nn_module(
  "Tokenizer",
  initialize = function(ncols, cat_inds, bias = F, d_token = 8) {
    self$cat_inds <- cat_inds
    self$d_token <- d_token
    d_bias <- ncols
    binary_offsets <- cumsum(c(1, rep(2, length(cat_inds) - 1)))
    
    self$register_buffer("binary_offsets", torch_tensor(binary_offsets, dtype = torch_long()))
    self$binary_embeddings <- nn_embedding(length(cat_inds) * 2, d_token)
    self$binary_embeddings$weight$requires_grad <- F
    nn_init_kaiming_uniform_(self$binary_embeddings$weight, a = sqrt(5))
    
    self$weight <- nn_parameter(torch_empty(ncols - length(cat_inds) + 1, d_token))
    self$weight$requires_grad <- F
    if (bias){
      self$bias <- nn_parameter(torch_empty(d_bias, d_token))
    }else{
      self$bias <- NULL
    }
    
    nn_init_kaiming_uniform_(self$weight, a = sqrt(5))
    if (!is.null(self$bias)){
      nn_init_kaiming_uniform_(self$bias, a = sqrt(5))
      self$bias$requires_grad <- F
    }
  },
  
  n_tokens = function() {
    if (length(self$cat_inds) <= 1){
      return (self$weight$size(1))
    }else{
      return (self$binary_offsets$size(1) + self$weight$size(1))
    }
  },
  
  forward = function(xnum, xcat) {
    if (length(self$cat_inds) > 0){
      x_some <- xcat
    }else{
      x_some <- xnum
    }
    
    if (is.null(xnum)){
      x_num <- torch_ones(c(x_some$size(1), 1), device = x_some$device)
    }else{
      x_num <- torch_cat(list(torch_ones(c(x_some$size(1), 1), device = x_some$device),
                              xnum), dim = 2)
    }
    
    x <- self$weight$unsqueeze(1) * x_num$unsqueeze(3)
    
    if (length(self$cat_inds) > 0){
      x <- torch_cat(list(x, self$binary_embeddings(x_some$to(dtype = torch_long()) + self$binary_offsets$unsqueeze(1))), dim = 2)
    }
    
    if (!is.null(self$bias)){
      bias <- torch_cat(list(torch_zeros(c(1, self$bias$shape(2)), device = x$device)))
      x <- x + bias$unsqueeze(1)
    }
    
    return (x)
  }
)

Encoder <- torch::nn_module(
  "Encoder",
  initialize = function(embed_dim, num_heads){
    self$attn <- nn_multihead_attention(embed_dim, num_heads, batch_first = T)
    
    self$dropout1 <- nn_dropout()
    self$dropout2 <- nn_dropout()
    
    self$norm1 <- nn_layer_norm(embed_dim)
    self$norm2 <- nn_layer_norm(embed_dim)
    
    self$linear1 <- nn_linear(embed_dim, embed_dim)
    self$linear2 <- nn_linear(embed_dim, embed_dim)
    
    self$gelu <- nnf_gelu
  },
  forward = function(input){
    input <- input$unsqueeze(2)
    attn_out <- self$dropout1(self$attn(input, input, input)[[1]])
    attn_out <- self$norm1(input + attn_out)
    out <- self$dropout2(self$linear2(self$gelu(self$linear1(attn_out))))
    out <- self$norm2(attn_out + out)
    out <- out$squeeze(2)
    return (out)
  }
)


generator <- function(n_g_layers, g_dim, ncols, nphase2, cat_inds, type = "mlp"){
  generator.mlp <- torch::nn_module(
    "Generator",
    initialize = function(n_g_layers, g_dim, ncols, nphase2, cat_inds){
      dim1 <- g_dim[1] + ncols - nphase2
      dim2 <- g_dim[2]
      self$seq <- torch::nn_sequential()
      for (i in 1:n_g_layers){
        self$seq$add_module(paste0("Residual_", i), Residual(dim1, dim2))
        dim1 <- dim1 + dim2
      }
      self$seq$add_module("Linear", nn_linear(dim1, nphase2))
    },
    forward = function(input){
      input <- self$tokenizer(input[, self$num_inds, drop = F], 
                              input[, self$cat_inds, drop = F])
      input <- input$reshape(c(input$size(1), input$size(2) * input$size(3)))
      input <- input$reshape(c(-1, self$pacdim))
      out <- self$seq(input)
      out <- self$lastlinear(out)
      
      out <- self$seq(input)
      return (out)
    }
  )
  
  generator.attn <- torch::nn_module(
    "Generator",
    initialize = function(n_g_layers, g_dim, ncols, nphase2, cat_inds){
      self$cat_inds <- (cat_inds - nphase2)[(cat_inds - nphase2) > 0]
      self$num_inds <- which(!(1:(ncols - nphase2) %in% self$cat_inds))
      self$tokenizer <- Tokenizer((ncols - nphase2), self$cat_inds)
      
      self$noise_dim <- g_dim[1]
      dim1 <- g_dim[1] + (ncols - nphase2 + 1) * self$tokenizer$d_token
      #dim2 <- g_dim[2]
      
      self$seq <- torch::nn_sequential()
      for (i in 1:n_g_layers){
        self$seq$add_module(paste0("Encoder_", i), Encoder(dim1, self$tokenizer$d_token))
      }
      self$seq$add_module("Linear", nn_linear(dim1, nphase2))
    },
    forward = function(input){
      conditions <- input[, (self$noise_dim + 1):(input$size(2))]
      conditions <- self$tokenizer(conditions[, self$num_inds, drop = F], 
                                   conditions[, self$cat_inds, drop = F])
      conditions <- conditions$reshape(c(conditions$size(1), 
                                         conditions$size(2) * conditions$size(3)))
      input <- torch_cat(list(input[, 1:self$noise_dim], conditions), dim = 2)
      out <- self$seq(input)
      return (out)
    }
  )
  g <- paste("generator", type, sep = ".")
  gnet <- do.call(g, args = list(n_g_layers, g_dim, ncols, nphase2, cat_inds))
  return (gnet)
}

discriminator <- function(n_d_layers, ncols, pac, cat_inds, type = "mlp"){
  discriminator.mlp <- torch::nn_module(
    "Discriminator",
    initialize = function(n_d_layers, ncols, pac, cat_inds) {
      self$pac <- pac
      self$pacdim <- ncols * pac
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
    initialize = function(n_d_layers, ncols, pac, cat_inds) {
      self$ncols <- ncols
      self$pac <- pac
      self$cat_inds <- cat_inds
      self$num_inds <- which(!(1:ncols %in% cat_inds))
      self$tokenizer <- Tokenizer(ncols, cat_inds)
      
      self$pacdim <- self$tokenizer$d_token * (ncols + 1) * pac
      
      self$seq <- torch::nn_sequential()
      
      for (i in 1:n_d_layers) {
        self$seq$add_module(paste0("Encoder_", i), Encoder(self$pacdim, pac))
      }
      
      self$lastlinear <- nn_linear(self$pacdim, 1)
    },
    forward = function(input) {
      input <- self$tokenizer(input[, self$num_inds, drop = F], 
                              input[, self$cat_inds, drop = F])
      input <- input$reshape(c(input$size(1), input$size(2) * input$size(3)))
      input <- input$reshape(c(-1, self$pacdim))
      out <- self$seq(input)
      out <- self$lastlinear(out)
      
      return(out)
    }
  )
  d <- paste("discriminator", type, sep = ".")
  dnet <- do.call(d, args = list(n_d_layers, ncols, pac, cat_inds))
  return (dnet)
}

gradient_penalty <- function(D, real_samples, fake_samples, pac, device) {
  # Generate alpha for each pack (here batch_size/pac groups)
  alp <- torch_rand(c(ceiling(real_samples$size(1) / pac), 1, 1))$to(device = device)
  pac <- torch_tensor(as.integer(pac), device = device)
  size <- torch_tensor(real_samples$size(2), device = device)
  
  alp <- alp$repeat_interleave(pac, dim = 2)$repeat_interleave(size, dim = 3)
  alp <- alp$reshape(c(-1, real_samples$size(2)))
  
  interpolates <- (alp * real_samples + (1 - alp) * fake_samples)$requires_grad_(TRUE)
  d_interpolates <- D(interpolates)
  
  fake <- torch_ones(d_interpolates$size(), device = device)
  fake$requires_grad <- FALSE
  
  gradients <- torch::autograd_grad(
    outputs = d_interpolates,
    inputs = interpolates,
    grad_outputs = fake,
    create_graph = TRUE,
    retain_graph = TRUE
  )[[1]]
  
  # Reshape gradients to group the pac samples together
  gradients <- gradients$reshape(c(-1, pac$item() * size$item()))
  gradient_penalty <- torch_mean((torch_norm(gradients, p = 2, dim = 2) - 1) ^ 2)
  
  return (gradient_penalty)
}
