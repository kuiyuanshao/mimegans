#################################### Generate Sample ####################################
lapply(c("BalancedSampling", "dplyr"), require, character.only = T)
source("00_utils_functions.R")
balanceSample <- function(obsT, obsE, obsZ, ){
  
}
generateSample <- function(data, proportion, seed){
  set.seed(seed)
  nRow <- nrow(data)
  n_phase2 <- round(nRow * proportion) 
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
                  labels = 1:9) # Cut by every 3 months
  hba1c_cut <- as.numeric(cut(data$HbA1c_STAR, breaks = c(-Inf, 64, 75, Inf), 
                   labels = 1:3))
  strata <- interaction(data$EVENT_STAR, time_cut, hba1c_cut, drop = TRUE)
  k <- nlevels(strata)
  per_strat <- floor(n_phase2 / k)
  ids_by_str <- split(seq_len(nRow), strata)
  balanced_ind <- unlist(lapply(names(ids_by_str), function(i){
    if (table(strata)[i] < per_strat){
      return (ids_by_str[[i]])
    }else{
      return (sample(ids_by_str[[i]], per_strat))
    }
  }))
  openStrata <- table(strata)[table(strata) > per_strat]
  remaining <- (n_phase2 - length(balanced_ind)) / length(openStrata)
  
  balanced_weights <- table(strata) / per_strat
  samp_balance <- data %>%
    dplyr::mutate(R = ifelse(1:nRow %in% balanced_ind, 1, 0),
                  STRATA = strata, 
                  W = case_when(!!!lapply(names(balanced_weights), function(value){
                    expr(STRATA == !!value ~ !!balanced_weights[[value]])
                  })),
                  across(all_of(p2vars), ~ ifelse(R == 0, NA, .)))
  # Adaptive Sampling
  
  
  
  
  neyman_alloc <- exactAllocation(data = data, stratum_variable = "stratum",
                                  target_variable = "HbA1c_STAR",
                                  sample_size = ceiling(proportion * nRow))
  neyman_ind <- lapply(1:length(table(data[["stratum"]])), 
                       function(j){sample((1:nRow)[data[["stratum"]] == j], neyman_alloc[j])})
  neyman_ind <- unlist(neyman_ind)
  neyman_weights <- table(stratum) / neyman_alloc
  samp_neyman <- data %>%
    dplyr::mutate(R = ifelse(1:nRow %in% neyman_ind, 1, 0),
                  W = case_when(!!!lapply(names(neyman_weights), function(value){
                    expr(stratum == !!value ~ !!neyman_weights[[value]])
                  })),
                  across(all_of(p2vars), ~ ifelse(R == 0, NA, .)))
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


