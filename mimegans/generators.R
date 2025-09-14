generator.mlp <- torch::nn_module(
  "Generator",
  initialize = function(params, ncols, nphase2, nphase1, rate, ...){
    self$nphase2 <- nphase2
    dim1 <- params$noise_dim + ncols - nphase2
    dim2 <- params$g_dim
    self$params <- params
    self$seq <- torch::nn_sequential(nn_dropout(0.25))
    for (i in 1:params$n_g_layers){
      self$seq$add_module(paste0("Residual_", i), Residual(dim1, dim2, rate))
      dim1 <- dim1 + dim2
    }
    
    if (params$component == "gen_loss"){
      self$seq$add_module("Linear", nn_linear(dim1, ncols))
    }else{
      self$seq$add_module("Linear", nn_linear(dim1, nphase2))
    }
    
    if (params$component == "match_p1"){
      dim1 <- nphase2
      self$seqA <- torch::nn_sequential(nn_dropout(0.25))
      for (i in 1:2){
        self$seqA$add_module(paste0("Residual_", i), Residual(dim1, dim2, rate))
        dim1 <- dim1 + dim2
      }
      self$seqA$add_module("Linear", nn_linear(dim1, nphase1))
    }
  },
  forward = function(input, ...){
    X_fake <- self$seq(input)
    if (self$params$component == "match_p1"){
      A_fake <- self$seqA(X_fake$clone())
      return (list(X_fake, A_fake))
    }else if (self$params$component == "gen_loss"){
      Full_fake <- X_fake
      X_fake <- X_fake[, 1:self$nphase2, drop = F]
      return (list(X_fake, Full_fake))
    }else{
      return (list(X_fake))
    }
  }
)