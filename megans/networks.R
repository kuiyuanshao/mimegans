Residual <- torch::nn_module(
  "Residual",
  initialize = function(dim1, dim2, rate){
    self$linear <- nn_linear(dim1, dim2)
    self$norm <- nn_batch_norm1d(dim2)
    self$act <- nn_elu()
    self$dropout <- nn_dropout(rate)
  },
  forward = function(input){
    output <- input %>% 
      self$linear() %>% 
      self$norm() %>%
      self$act() %>%
      self$dropout()
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

ScaleNorm <- torch::nn_module(
  "ScaleNorm",
  initialize = function(scale = 1, eps = 1e-5){
    self$scale <- nn_parameter(torch_tensor(scale))
    self$eps <- eps
  },
  forward = function(x){
    norm <- self$scale / (torch_norm(x, dim = -1, keepdim = T) + self$eps)
    return (x * norm)
  }
)

RMSNorm <- nn_module(
  "RMSNorm",
  initialize = function(d, p = -1, eps = 1e-8, bias = FALSE) {
    self$eps <- eps
    self$d <- d
    self$p <- p
    self$bias <- bias
    
    self$scale <- nn_parameter(torch_ones(d))   # γ vector
    if (bias) {
      self$offset <- nn_parameter(torch_zeros(d))
    }
  },
  
  forward = function(x) {
    if (self$p < 0 || self$p > 1) {
      norm_x <- x$norm(p = 2, dim = -1, keepdim = TRUE)
      d_x    <- self$d
    } else {
      partial_size <- as.integer(self$d * self$p)
      split <- torch_split(x, c(partial_size, self$d - partial_size), dim = -1)
      partial_x <- split[[1]]
      norm_x <- partial_x$norm(p = 2, dim = -1, keepdim = TRUE)
      d_x <- partial_size
    }
    
    rms_x <- norm_x * d_x^(-0.5)
    x_normed <- x / (rms_x + self$eps)
    
    if (self$bias) {
      return(self$scale * x_normed + self$offset)
    } else {
      return(self$scale * x_normed)
    }
  }
)

Encoder <- torch::nn_module(
  "Encoder",
  initialize = function(embed_dim, num_heads){
    self$attn <- nn_multihead_attention(embed_dim, num_heads, 
                                        batch_first = T)
    self$dropout1 <- nn_dropout(0.5)
    self$dropout2 <- nn_dropout(0.5)
    
    self$norm1 <- nn_layer_norm(embed_dim)
    self$norm2 <- nn_layer_norm(embed_dim)
    
    self$ff <- torch::nn_sequential(
      nn_linear(embed_dim, embed_dim),
      nn_gelu(),
      nn_linear(embed_dim, embed_dim)
    )
  },
  forward = function(input1, input2){
    input1 <- input1$unsqueeze(2)
    input2 <- input2$unsqueeze(2)
    attn_out <- self$attn(input1, input2, input2)[[1]]
    attn_out <- self$norm1(input1 + self$dropout1(attn_out))
    out <- self$norm2(attn_out + self$dropout1(self$ff(attn_out)))
    out <- out$squeeze(2)
    return (out)
  }
)


SpectralNorm <- nn_module(
  classname = "SpectralNorm",
  
  initialize = function(module, name = "weight",
                        n_power_iterations = 1L, eps = 1e-12) {
    
    self$module <- module
    self$name <- name
    self$niters <- as.integer(n_power_iterations)
    self$eps <- eps
    
    w <- module$parameters[[name]]
    stopifnot(!is.null(w))
    
    # init power‑iter vectors as buffers (unit length)
    w_mat  <- w$view(c(w$size(1), -1))
    make_unit <- function(dim) {
      v <- torch_randn(dim, device = w$device, dtype = w$dtype)
      v / (v$norm(p = 2) + eps)
    }
    self$u <- nn_buffer(make_unit(w_mat$size(1)))
    self$v <- nn_buffer(make_unit(w_mat$size(2)))
  },
  
  .power_iteration = function(w_mat) {
    u <- self$u$detach()
    v <- self$v$detach()
    eps <- self$eps
    
    for (i in seq_len(self$niters)) {
      v <- torch_matmul(w_mat$t(), u)
      v <- v / (v$norm(p = 2) + eps)
      u <- torch_matmul(w_mat, v)
      u <- u / (u$norm(p = 2) + eps)
    }
    # update buffers *outside* autograd
    with_no_grad({
      self$u$copy_(u)
      self$v$copy_(v)
    })
    list(u = u, v = v)
  },
  
  forward = function(x) {
    w_orig <- self$module$parameters[[self$name]]
    w_mat  <- w_orig$view(c(w_orig$size(1), -1))
    
    uv   <- self$.power_iteration(w_mat)
    sigma <- torch_dot(uv$u, torch_matmul(w_mat, uv$v))$detach()
    w_bar <- w_orig / (sigma)         # fresh tensor, no graph history

    return(nnf_linear(x, w_bar, bias = self$module$bias))
  }
)

spectral_norm <- function(module, name = "weight", n_power_iterations = 1L) {
  SpectralNorm(module, name, n_power_iterations)
}



