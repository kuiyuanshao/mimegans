Residual <- torch::nn_module(
  "Residual",
  initialize = function(dim1, dim2, rate, ...){
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
    return (torch_cat(list(output, input), dim = 2))
  }
)


