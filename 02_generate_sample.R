#################################### Generate Sample ####################################
lapply(c("BalancedSampling", "dplyr"), require, character.only = T)
source("00_utils_functions.R")
generateSample <- function(data, proportion, seed){
  set.seed(seed)
  nRow <- N <- nrow(data)
  n_phase2 <- n <- round(nRow * proportion) 
  p2vars <- c("SMOKE", "ALC", "EXER", "INCOME", "EDU", 
              "Na_INTAKE", "K_INTAKE", "KCAL_INTAKE", "PROTEIN_INTAKE", 
              "rs10811661", "rs7756992", "rs11708067", "rs17036101", "rs17584499",
              "rs1111875", "rs4402960", "rs4607103", "rs7754840", "rs9300039",
              "rs5015480", "rs9465871", "rs4506565", "rs5219", "rs358806",
              "GLUCOSE", "F_GLUCOSE", "HbA1c", "INSULIN", "T_I", "EVENT", "C")
  # Simple Random Sampling
  srs_ind <- sample(nRow, n_phase2)
  samp_srs <- data %>%
    dplyr::mutate(R = ifelse(1:nRow %in% srs_ind, 1, 0),
                  W = 1,
                  across(all_of(p2vars), ~ ifelse(R == 0, NA, .)))
  # Balanced Sampling
  time_cut <- cut(data$T_I_STAR, breaks = c(-Inf, seq(3, 24, by = 3), Inf), 
                  labels = 1:8) # Cut by every 3 months
  hba1c_cut <- as.numeric(cut(data$HbA1c_STAR, breaks = c(-Inf, 64, 75, Inf), 
                   labels = 1:3))
  hba1c_cut <- as.numeric(data$URBAN)
  strata <- interaction(data$EVENT_STAR, time_cut, hba1c_cut, drop = TRUE)
  k <- nlevels(strata)
  per_strat <- floor(n_phase2 / k)
  ids_by_str <- split(seq_len(nRow), strata)
  balanced_ind <- unlist(lapply(names(ids_by_str), function(i){
    if (table(strata)[i] < per_strat){
      return (ids_by_str[[i]]) # Sample everyone if insufficient 
    }else{
      return (sample(ids_by_str[[i]], per_strat))
    }
  }))
  openStrata <- names(table(strata)[table(strata) > per_strat])
  remaining_per_strat <- ceiling((n_phase2 - length(balanced_ind)) / length(openStrata))
  remaining_ind <- unlist(lapply(openStrata, function(i){
    sample(ids_by_str[[i]][!(ids_by_str[[i]] %in% balanced_ind)], remaining_per_strat)
  }))[1:(n_phase2 - length(balanced_ind))]
  balanced_ind <- c(balanced_ind, remaining_ind)
  balanced_weights <- table(strata) / table(strata[balanced_ind])
  samp_balance <- data %>%
    dplyr::mutate(R = ifelse(1:nRow %in% balanced_ind, 1, 0),
                  STRATA = strata, 
                  W = case_when(!!!lapply(names(balanced_weights), function(value){
                    expr(STRATA == !!value ~ !!balanced_weights[[value]])
                  })),
                  across(all_of(p2vars), ~ ifelse(R == 0, NA, .)))
  # Stratified Sampling with Neyman Allocation
  ### Getting Influence Function by auxiliary variables
  mod.aux <- coxph(Surv(T_I_STAR, EVENT_STAR) ~ I((HbA1c_STAR - 50) / 5) + 
                     rs4506565_STAR + I((AGE - 50) / 5) + SEX + INSURANCE + 
                     RACE + I(BMI / 5) + EXER_STAR, 
                   data = data)
  
  
  return (list(samp_srs = samp_srs,
               samp_balance = samp_balance,
               samp_neyman = samp_neyman))
}

####### STARTING SIMULATION.  SAVING FILES ########
if(!dir.exists('./data/Sample')){system('mkdir ./data/Sample')}
if(!dir.exists('./data/Sample/SRS')){system('mkdir ./data/Sample/SRS')}
if(!dir.exists('./data/Sample/Balance')){system('mkdir ./data/Sample/Balance')}
if(!dir.exists('./data/Sample/Neyman')){system('mkdir ./data/Sample/Neyman')}
replicate <- 1
seed <- 1
for (i in 1:replicate){
  digit <- stringr::str_pad(i, 4, pad = 0)
  cat("Current:", digit, "\n")
  load(paste0("./data/Complete/", digit, ".RData"))
  samp_result <- generateSample(data, 0.05, seed)
  write.csv(samp_result$samp_balance, 
            file = paste0("./data/Sample/Balance/", digit, ".csv"))
  write.csv(samp_result$samp_neyman, 
            file = paste0("./data/Sample/Neyman/", digit, ".csv"))
  seed <- seed + 1
}


