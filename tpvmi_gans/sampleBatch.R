initset <- dataset(
  name = "initData",
  initialize = function(data_training, rows, 
                        phase1_vars_encode, 
                        phase2_vars_encode, 
                        conditions_vars_encode,
                        device) {  # Added device argument
    
    subset <- data_training[rows, ]
    
    # 1. Phase 1 Vars: Matrix -> Tensor -> Device
    self$A_t <- torch_tensor(as.matrix(subset[, phase1_vars_encode]), device = device)
    
    # 2. Phase 2 Vars & Mask
    X_mat <- as.matrix(subset[, phase2_vars_encode])
    
    # Generate Mask directly on Device
    # Note: 1 - is.na() results in numeric, convert to long for index/mask usage
    self$M_t <- torch_tensor(1 - is.na(X_mat), dtype = torch_long(), device = device)
    
    # Handle NA in X and move to Device
    X_mat[is.na(X_mat)] <- 0 
    self$X_t <- torch_tensor(X_mat, device = device)
    
    # 3. Conditions Vars: Matrix -> Tensor -> Device
    self$C_t <- torch_tensor(as.matrix(subset[, conditions_vars_encode]), device = device)
  },
  
  .getitem = function(i) {
    # Directly slice the GPU-resident tensors. Zero overhead.
    list(M = self$M_t[i, ], 
         C = self$C_t[i, ], 
         X = self$X_t[i, ], 
         A = self$A_t[i, ])
  },
  
  .length = function() {
    self$A_t$size(1)
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

# Time Complexity: O(Batch_Size)
sample_from_cache <- function(grouped_indices, counts_vec) {
  total_req <- sum(counts_vec)
  res <- integer(total_req) # Pre-allocate memory to avoid c() vector growth overhead.
  cursor <- 1L
  
  for (grp in names(counts_vec)) {
    n_req <- counts_vec[[grp]]
    
    # Only proceed if we actually need samples and the group exists
    if (n_req > 0) {
      candidates <- grouped_indices[[grp]]
      n_cand <- length(candidates)
      
      # FIX: Critical Safety Check
      # 1. Check if candidates exist (n_cand > 0) to prevent "invalid first argument"
      # 2. Use sample.int on LENGTH to avoid sample() scalar ambiguity behavior
      if (n_cand > 0) {
        # Safe sampling: get indices positions, then subset candidates
        s_idx <- sample.int(n_cand, n_req, replace = n_req > n_cand)
        s <- candidates[s_idx]
        
        # Direct assignment
        end <- cursor + n_req - 1L
        res[cursor:end] <- s
        cursor <- cursor + n_req
      } else {
        # This branch implies alloc_even assigned counts to an empty group.
        # With drop=TRUE in BalancedSampler, this should theoretically not happen,
        # but this check prevents the crash if it does.
        warning(paste("Skipping empty group:", grp))
      }
    }
  }
  
  # Trim result if we skipped any empty groups (safety)
  if (cursor <= total_req) {
    res <- res[1:(cursor - 1L)]
  }
  return(res)
}

BalancedSampler <- sampler(
  "BalancedSampler",
  initialize = function(x, bin_cols, batch_size, epochs) {
    self$bs <- batch_size
    self$L <- length(bin_cols)
    self$epochs <- epochs
    self$bin_cols <- bin_cols
    
    self$indices_cache <- list()
    for (col in bin_cols) {
      # FIX: add `drop = TRUE`
      # This ensures that if a factor level has 0 rows in this specific dataset subset,
      # it is removed from the list. alloc_even will then not assign samples to it.
      self$indices_cache[[col]] <- split(seq_len(nrow(x)), as.factor(x[, col]), drop = TRUE)
    }
  },
  
  .iter = function() {
    cursor <- 0L
    function() {
      # 1. Get the column name for the current iteration.
      col_name <- self$bin_cols[(cursor %% self$L) + 1L]
      
      # 2. O(1) retrieval of pre-computed grouped indices for this column.
      groups <- self$indices_cache[[col_name]]
      
      # 3. Calculate the number of samples needed for each group.
      count_dist <- alloc_even(self$bs, names(groups))
      
      # 4. Execute fast sampling using the cached groups.
      idx <- sample_from_cache(groups, count_dist)
      
      # 5. Shuffle the internal order of the batch.
      # FIX: Use safe shuffle (sample.int) to handle cases where batch size might be 1.
      if (length(idx) > 0) {
        idx <- idx[sample.int(length(idx))]
      }
      
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