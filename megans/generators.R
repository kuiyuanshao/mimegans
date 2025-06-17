generator.mlp <- torch::nn_module(
  "Generator",
  initialize = function(n_g_layers, params, ncols, nphase2, output_dim, cat_inds){
    dim1 <- params$g_dim + ncols - nphase2
    dim2 <- params$g_dim
    self$seq <- torch::nn_sequential()
    for (i in 1:n_g_layers){
      self$seq$add_module(paste0("Residual_", i), Residual(dim1, dim2))
      dim1 <- dim1 + dim2
    }
    self$seq$add_module("Linear", nn_linear(dim1, output_dim))
  },
  forward = function(input){
    out <- self$seq(input)
    return (out)
  }
)

generator.attn <- torch::nn_module(
  "Generator",
  initialize = function(n_g_layers, params, ncols, nphase2, output_dim, cat_inds){
    self$cat_inds <- (cat_inds - nphase2)[(cat_inds - nphase2) > 0]
    self$num_inds <- which(!(1:(ncols - nphase2) %in% self$cat_inds))
    self$tokenizer <- Tokenizer((ncols - nphase2), self$cat_inds,
                                params$token_dim, params$token_learn)
    
    self$noise_dim <- params$g_dim
    dim1 <- params$g_dim + (ncols - nphase2 + 1) * self$tokenizer$token_dim
    dim2 <- params$g_dim
    
    self$proj_layer <- torch::nn_sequential(
      nn_linear(dim1, dim2),
      nn_layer_norm(dim2),
      nn_relu()
    )
    
    self$seq <- torch::nn_sequential()
    for (i in 1:n_g_layers){
      self$seq$add_module(paste0("Encoder_", i), Encoder(dim2, 8))
    }
    
    self$output_layer <- torch::nn_sequential(
      nn_layer_norm(dim2),
      nn_relu(),
      nn_linear(dim2, output_dim)
    )
  },
  forward = function(input){
    conditions <- input[, (self$noise_dim + 1):(input$size(2))]
    conditions <- self$tokenizer(conditions[, self$num_inds, drop = F], 
                                 conditions[, self$cat_inds, drop = F])
    conditions <- conditions$reshape(c(conditions$size(1), 
                                       conditions$size(2) * conditions$size(3)))
    input <- torch_cat(list(input[, 1:self$noise_dim], conditions), dim = 2)
    out <- input %>% 
      self$proj_layer() %>% 
      self$seq() %>%
      self$output_layer()
    return (out)
  }
)