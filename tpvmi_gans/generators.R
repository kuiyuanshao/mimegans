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
    output <- self$act(self$norm(self$linear(input)))
    if (self$rate > 0){
      output <- self$dropout(output)
    }
    return (torch_cat(list(output, input), dim = 2))
  }
)

generator.mlp <- nn_module(
  "Generator",
  initialize = function(params, ...){
    self$nphase2 <- params$nphase2
    self$params <- params

    dim1 <- params$noise_dim + params$cond_dim

    self$dropout <- nn_dropout(params$g_dropout / 2)

    self$cond_encoder <- nn_sequential(
      nn_linear(params$ncols - params$nphase2, params$cond_dim),
      nn_batch_norm1d(params$cond_dim),
      nn_leaky_relu(0.2)
    )

    self$seq <- nn_sequential()
    for (i in 1:length(params$g_dim)){
      self$seq$add_module(paste0("Residual_", i), Residual(dim1, params$g_dim[i], params$g_dropout))
      dim1 <- dim1 + params$g_dim[i]
    }
    self$seq$add_module("Linear", nn_linear(dim1, params$nphase2))
  },
  forward = function(N, A, C, ...){
    cond <- torch_cat(list(A, C), dim = 2)
    cond <- self$cond_encoder(cond)
    if (self$params$g_dropout > 0){
      cond <- self$dropout(cond)
    }
    input <- torch_cat(list(N, cond), dim = 2)
    X_fake <- self$seq(input)
    
    # X_fake[, self$params$cat_inds] <- X_fake[, self$params$cat_inds] + 5 * A[, self$params$cat_inds]
    return (X_fake)
  }
)
