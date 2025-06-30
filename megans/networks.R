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
  initialize = function(ncols, cat_inds, token_dim = 8, n_unique) {
    self$cat_inds <- cat_inds
    self$token_dim <- token_dim
    category_offsets <- cumsum(c(1, n_unique[-length(n_unique)]))
    
    self$register_buffer("category_offsets", torch_tensor(category_offsets, dtype = torch_long()))
    self$category_embeddings <- nn_embedding(sum(n_unique) + 1, token_dim)
    self$category_embeddings$weight$requires_grad <- T
    nn_init_kaiming_uniform_(self$category_embeddings$weight, a = sqrt(5))
    
    self$weight <- nn_parameter(torch_empty(ncols - length(cat_inds) + 1, token_dim))
    self$weight$requires_grad <- T
    
    nn_init_kaiming_uniform_(self$weight, a = sqrt(5))
    
  },
  
  n_tokens = function() {
    if (length(self$cat_inds) <= 1){
      return (self$weight$size(1))
    }else{
      return (self$category_offsets$size(1) + self$weight$size(1))
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
      inp <- x_some$to(dtype = torch_long()) + 
        self$category_offsets$unsqueeze(1)
      x <- torch_cat(list(x, self$category_embeddings(inp)), dim = 2)
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
    
    self$ff <- torch::nn_sequential(
      nn_linear(embed_dim, embed_dim),
      nn_gelu(),
      nn_linear(embed_dim, embed_dim)
    )
  },
  forward = function(input){
    input <- input$unsqueeze(2)
    attn_out <- self$attn(input, input, input)[[1]]
    attn_out <- self$norm1(input + self$dropout1(attn_out))
    out <- self$norm2(attn_out + self$dropout2(self$ff(attn_out)))
    out <- out$squeeze(2)
    return (out)
  }
)

m_net <- nn_module(
  initialize = function(input_dim, params, hidden = 64){
    self$pacdim <- input_dim * params$pac
    self$seq <- nn_sequential(
      nn_linear(self$pacdim, hidden),
      nn_relu(),
      nn_linear(hidden, 1)
    )
  },
  forward = function(input){
    input <- input$reshape(c(-1, self$pacdim))
    self$seq(input)
  }
)

gradient_penalty <- function(D, real_samples, fake_samples, pac, device) {
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
