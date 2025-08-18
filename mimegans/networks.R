Residual <- torch::nn_module(
  "Residual",
  initialize = function(dim1, dim2, rate, resid = "concat", ...){
    self$resid <- resid
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
    if (self$resid == "concat"){
      return (torch_cat(list(output, input), dim = 2))
    }else{
      return (output + input)
    }
  }
)

SpectralNorm <- nn_module(
  classname = "SpectralNorm",
  
  initialize = function(module, name = "weight",
                        n_power_iterations = 1L, eps = 1e-12) {
    
    self$register_module("module", module)
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



