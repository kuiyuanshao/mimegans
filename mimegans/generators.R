Residual <- torch::nn_module(
  "Residual",
  initialize = function(dim1, dim2, rate, ...){
    self$rate <- rate
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
    return (torch_cat(list(input, output), dim = 2))
  }
)

forwardA.mlp <- torch::nn_module(
  "ForwardA",
  initialize = function(params, ncols, nphase1){
    dim1 <- ncols - nphase1
    self$seq <- torch::nn_sequential()
    for (i in 1:length(params$g_dim)){
      self$seq$add_module(paste0("Residual_", i), Residual(dim1, params$g_dim[i], params$g_dropout / 2))
      dim1 <- dim1 + params$g_dim[i]
    }
    self$seq$add_module("Linear", nn_linear(dim1, nphase1))
  },
  forward = function(input){
    fake <- self$seq(input)
    return (fake)
  }
)

generator.mlp <- torch::nn_module(
  "Generator",
  initialize = function(params, ncols, nphase2, nphase1, ...){
    self$nphase2 <- nphase2
    self$params <- params
    
    if (params$unconditional){
      self$dim1 <- ncols
      dim1 <- ncols
    }else{
      self$dim1 <- params$noise_dim + ncols - nphase2
      dim1 <- params$noise_dim + ncols - nphase2
    }
    self$dropout <- nn_dropout(params$g_dropout * 0.4)
    self$seq <- torch::nn_sequential()
    for (i in 1:length(params$g_dim)){
      self$seq$add_module(paste0("Residual_", i), Residual(dim1, params$g_dim[i], params$g_dropout))
      dim1 <- dim1 + params$g_dim[i]
    }
    if (params$component == "gen_loss"){
      self$seq$add_module("Linear", nn_linear(dim1, ncols))
    }else{
      self$seq$add_module("Linear", nn_linear(dim1, nphase2))
    }
    
    if (params$component == "match_p1"){
      dim1 <- nphase2
      self$seqA <- torch::nn_sequential()
      for (i in 1:length(params$g_dim)){
        self$seqA$add_module(paste0("Residual_", i), Residual(dim1, params$g_dim[i], 0))
        dim1 <- dim1 + params$g_dim[i]
      }
      self$seqA$add_module("Linear", nn_linear(dim1, nphase1))
    }
  },
  forward = function(N, A, C, ...){
    if (self$params$g_dropout > 0){
      if (self$params$unconditional){
        input <- torch_cat(list(N, A, C), dim = 2)
        input <- self$dropout(input)
      }else{
        cond <- torch_cat(list(A, C), dim = 2)
        cond <- self$dropout(cond)
        input <- torch_cat(list(N, cond), dim = 2)
      }
    }
    
    X_fake <- self$seq(input)
    
    if (self$params$component == "gen_loss"){
      Full_fake <- X_fake
      X_fake <- X_fake[, 1:self$nphase2, drop = F]
      return (list(X_fake, Full_fake))
    }else{
      return (list(X_fake))
    }
  }
)