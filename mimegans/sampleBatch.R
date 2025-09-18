initset <- dataset(
  name = "initData",
  initialize = function(data_training, rows, 
                        phase1_vars_encode, phase2_vars_encode, conditions_vars_encode) {  
    subset <- data_training[rows, ]
    self$A_t <- subset[, phase1_vars_encode] %>% as.matrix()
    self$X_t <- subset[, phase2_vars_encode] %>% as.matrix()
    self$C_t <- subset[, conditions_vars_encode] %>% as.matrix()
    self$M_t <- 1 - is.na(self$X_t)
    self$X_t[is.na(self$X_t)] <- 0 
  },
  .getitem = function(i) {
    list(M = torch_tensor(self$M_t[i, ], dtype = torch_long()), 
         C = torch_tensor(self$C_t[i, ]), 
         X = torch_tensor(self$X_t[i, ]), 
         A = torch_tensor(self$A_t[i, ]))
  },
  .length = function() {
    self$A_t$size()[[1]]
  }
)

BalancedSampler <- sampler(
  "BalancedSampler",
  initialize = function(x, bin_cols, batch_size, epochs) {
    self$x <- x
    self$bin_cols <- bin_cols
    self$N <- dim(x)[1]
    self$bs <- batch_size
    self$L <- length(bin_cols)
    self$epochs <- epochs
    self$k1 <- floor(self$bs / 2)
    self$k0 <- self$bs - self$k1
  },
  .iter = function() {
    cursor <- 0L
    function() {
      col <- self$bin_cols[(cursor %% self$L) + 1L]
      v <- self$x[, col]
      idx1 <- which(v == 1L)
      idx0 <- which(v == 0L)
      
      rep1 <- length(idx1) < self$k1
      rep0 <- length(idx0) < self$k0
      take1 <- sample(idx1, self$k1, replace = rep1)
      take0 <- sample(idx0, self$k0, replace = rep0)
      idx <- c(take1, take0)
      if (length(idx) > 1) idx <- sample(idx)
      cursor <<- cursor + 1L
      idx
    }
  },
  .length = function(){
    self$epochs
  }
)

SRSSampler <- sampler(
  name = "SRSSampler",
  initialize = function(n, batch_size, epochs) {
    self$n <- n
    self$bs <- batch_size
    self$replace <- n < batch_size
    self$epochs <- epochs
  },
  .iter = function() {
    function() {
      sample.int(self$n, self$bs, replace = self$replace)
    }
  },
  .length = function() {
    self$epochs
  }
)