samplebatches <- function(data_original, data_training, tensor_list, 
                          phase1_rows, phase2_rows, phase1_vars, phase2_vars, 
                          num_vars, new_col_names, batch_size, at_least_p = 0.2){
  # Remove non-informative columns for phase1 and phase2 based on the training data
  cat_names <- unlist(new_col_names)
  phase1_bins <- cat_names[!(cat_names %in% phase2_vars)] 
  phase2_bins <- phase2_vars[!(phase2_vars %in% num_vars)]
  
  phase1_bins <- if (length(phase1_bins) > 0) {
    phase1_bins[sapply(phase1_bins, function(col) {
      length(unique(data_training[phase1_rows, col])) > 1
    })]
  } else {
    character(0)
  }
  
  phase2_bins <- if (length(phase2_bins) > 0) {
    phase2_bins[sapply(phase2_bins, function(col) {
      length(unique(data_training[phase2_rows, col])) > 1
    })]
  } else {
    character(0)
  }
  
  # Determine the total number of control (phase1) and case (phase2) samples needed
  n_control_total <- batch_size - as.integer(at_least_p * batch_size)
  n_case_total <- as.integer(at_least_p * batch_size)
  
  # For binary variables, we assume exactly 2 groups
  n_control <- c(floor(n_control_total/2), ceiling(n_control_total/2))
  n_case <- c(floor(n_case_total/2), ceiling(n_case_total/2))
  
  sampled_ind <- c()
  
  # Helper function: given a column for phase1 and phase2, try to sample indices using a binary split.
  sample_by_level <- function(phase_rows, col, desired_counts) {
    # Get the values for the provided column in the phase
    values <- data_training[phase_rows, col]
    unique_levels <- sort(unique(values))
    
    # Get indices for each binary level.
    group1 <- phase_rows[which(values == unique_levels[1])]
    group2 <- phase_rows[which(values == unique_levels[2])]
    
    desired1 <- desired_counts[1]
    desired2 <- desired_counts[2]
    
    # Sample from group1.
    if (length(group1) >= desired1) {
      samp1 <- sample(group1, size = desired1)
    } else {
      samp1 <- group1
    }
    
    # Sample from group2.
    if (length(group2) >= desired2) {
      samp2 <- sample(group2, size = desired2)
    } else {
      samp2 <- group2
    }
    
    total_sampled <- c(samp1, samp2)
    # Calculate total shortage for this phase.
    shortage <- (desired1 - length(samp1)) + (desired2 - length(samp2))
    
    if (shortage > 0) {
      # From the remaining indices (in both groups) not yet sampled, fill the gap.
      remaining <- setdiff(phase_rows, total_sampled)
      if (length(remaining) > 0) {
        additional <- sample(remaining, size = min(shortage, length(remaining)))
        total_sampled <- c(total_sampled, additional)
      }
    }
    
    # Return the sampled indices for the phase.
    return(total_sampled)
  }
  
  # Helper function that performs binary sampling for phase1 and phase2 using separate variables.
  sample_by_binary <- function(col_phase1, col_phase2) {
    # Phase 1 sampling using its own candidate variable.
    samp_phase1 <- sample_by_level(phase1_rows, col_phase1, n_control)
    # Phase 2 sampling using its own candidate variable.
    samp_phase2 <- sample_by_level(phase2_rows, col_phase2, n_case)
    
    sampled_ind <<- c(samp_phase1, samp_phase2)
    return(TRUE)
  }
  
  success <- FALSE
  # First try: if we have both valid phase1 and phase2 variables, try to pair one from each.
  if(length(phase1_bins) > 0 && length(phase2_bins) > 0) {
    # Randomize order in each pool to avoid systematic bias.
    for(var1 in sample(phase1_bins)) {
      for(var2 in sample(phase2_bins)) {
        if(!is.null(sample_by_binary(var1, var2))) {
          success <- TRUE
          break
        }
      }
      if(success) break
    }
  }
  
  # Fallback: if no valid binary pairing was found, perform simple random sampling.
  if(!success) {
    sampphase1 <- sample(phase1_rows, size = sum(n_control))
    sampphase2 <- sample(phase2_rows, size = sum(n_case))
    sampled_ind <- c(sampphase1, sampphase2)
  }
  
  # Create the batches by subsetting each tensor in the tensor_list.
  batches <- lapply(tensor_list, function(tensor) tensor[sampled_ind, , drop = FALSE])
  return(batches)
}