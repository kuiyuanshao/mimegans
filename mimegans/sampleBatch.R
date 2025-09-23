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

alloc_even <- function(total, levl) {
  k <- length(levl)
  if (k == 0L) return(integer(0))
  base <- total %/% k
  rem  <- total - base * k
  counts <- rep.int(base, k)
  if (rem > 0) counts[seq_len(rem)] <- counts[seq_len(rem)] + 1
  names(counts) <- levl
  counts
}

sampleFun <- function(indices, elements, total){
  samp_idx <- NULL
  for (i in unique(elements)){
    curr_rows <- indices[elements == i]
    sampled <- sample(1:length(curr_rows), total[[as.character(i)]],
                      replace = total[[as.character(i)]] > length(curr_rows))
    samp_idx <- c(samp_idx, curr_rows[sampled])
  }
  return (samp_idx)
}

BalancedSampler <- sampler(
  "BalancedSampler",
  initialize = function(x, bin_cols, batch_size, epochs) {
    self$x <- x
    self$bin_cols <- bin_cols
    self$N <- dim(x)[1]
    self$bs <- batch_size
    self$L <- length(bin_cols)
    self$epochs <- epochs
    
  },
  .iter = function() {
    cursor <- 0L
    function() {
      col <- self$bin_cols[(cursor %% self$L) + 1L]
      v <- self$x[, col]
      total <- alloc_even(self$bs, unique(v))
      idx <- sampleFun(1:self$N, v, total)
      idx <- sample(idx)
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