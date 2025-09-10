Residual <- torch::nn_module(
  "Residual",
  initialize = function(dim1, dim2, rate, resid = "concat", ...){
    self$rate <- rate
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
      self$act() 
    if (self$rate > 0){
      output <- self$dropout(output)
    }
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
                        n_power_iterations = 1L, eps = 1e-8) {
    
    self$register_module("module", module)
    self$name <- name
    self$niters <- as.integer(n_power_iterations)
    self$eps <- eps
    
    w <- module$parameters[[name]]
    stopifnot(!is.null(w))
    
    # init power‑iter vectors as buffers (unit length)
    w_mat  <- w$view(c(w$size(1), -1))
    make_unit <- function(dim) {
      v <- torch_randn(dim, device = w$device, dtype = torch_float32())
      v / (v$norm(p = 2) + eps)
    }
    self$u <- nn_buffer(make_unit(w_mat$size(1)))
    self$v <- nn_buffer(make_unit(w_mat$size(2)))
    self$sigma_prev <- nn_buffer(torch_tensor(1.0, device=w$device, dtype=torch_float32()))
  },
  
  .power_iteration = function(w_mat) {
    u <- self$u$detach()
    v <- self$v$detach()
    eps <- self$eps
    
    for (i in seq_len(self$niters)) {
      v <- torch_matmul(w_mat$t(), u)
      v_norm <- v$norm(p = 2)
      v <- v / torch_clamp(v_norm, eps)
      u <- torch_matmul(w_mat, v)
      u_norm <- u$norm(p = 2)
      u <- u / torch_clamp(u_norm, eps)
    }
    # update buffers *outside* autograd
    with_no_grad({
      self$u$copy_(u)
      self$v$copy_(v)
    })
    list(u = u, v = v)
  },
  
  forward = function(x) {
    w_orig <- self$module$parameters[[self$name]]$to(dtype = torch_float32())
    w_mat  <- w_orig$view(c(w_orig$size(1), -1))
    
    uv   <- self$.power_iteration(w_mat)
    sigma <- torch_dot(uv$u, torch_matmul(w_mat, uv$v))$detach()
    if (!as.logical(torch_isfinite(sigma)$item())){
      sigma <- self$sigma_prev
    }
    w_bar <- w_orig / (sigma + self$eps)         # fresh tensor, no graph history
    self$sigma_prev <- sigma
    # if (!as.logical(torch_isfinite(sigma)$item())) {
    #   sigma <- torch_tensor(1.0, device = w_mat$device, dtype = w_mat$dtype)
    # }
    # sigma <- torch_clamp(sigma, min = 1e-4)
    # inv_scale <- 1.0 / sigma
    # inv_scale <- torch_clamp(inv_scale, max = 1e4)
    # 
    # w_bar <- w_mat * inv_scale
    
    return(nnf_linear(x, w_bar, bias = self$module$bias))
  }
)

spectral_norm <- function(module, name = "weight", n_power_iterations = 1L) {
  SpectralNorm(module, name, n_power_iterations)
}



