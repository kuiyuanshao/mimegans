#################################### Generate Sample ####################################
lapply(c("BalancedSampling", "dplyr"), require, character.only = T)
source("00_utils_functions.R")
generateSample <- function(data, proportion, seed){
  set.seed(seed)
  nRow <- nrow(data)
  p2vars <- c("SMOKE", "ALC", "EXER", "EXER", "INCOME", "EDU", 
              "Na_INTAKE", "K_INTAKE", "KCAL_INTAKE", "PROTEIN_INTAKE", 
              "rs10811661", "rs7756992", "rs11708067", "rs17036101", "rs17584499",
              "rs1111875", "rs4402960", "rs4607103", "rs7754840", "rs9300039",
              "rs5015480", "rs9465871", "rs4506565", "rs5219", "rs358806",
              "GLUCOSE", "F_GLUCOSE", "HbA1c", "INSULIN", "T_I", "EVENT")
  # Balanced Sampling
  prob <- rep(proportion, nRow)
  time_cut <- cut(data$T_I_STAR, breaks = c(-Inf, 12, 18, 23.999, Inf), 
                  labels = 1:4)
  hba1c_cut <- as.numeric(cut(data$HbA1c_STAR, breaks = c(-Inf, 64, 75, Inf), 
                   labels = 1:3))
  stratum <- as.numeric(as.factor(paste0(time_cut, "_", data$EVENT_STAR)))
  data$stratum <- stratum
  balanced_ind <- cubestratified(prob, as.matrix(hba1c_cut), stratum)
  balanced_weights <- table(stratum) / table(stratum[balanced_ind])
  samp_balance <- data %>%
    dplyr::mutate(R = ifelse(1:nRow %in% balanced_ind, 1, 0),
                  W = case_when(!!!lapply(names(balanced_weights), function(value){
                    expr(stratum == !!value ~ !!balanced_weights[[value]])
                  })),
                  across(all_of(p2vars), ~ ifelse(R == 0, NA, .)))
  # Neyman Allocation
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
  return (list(samp_balance = samp_balance,
               samp_neyman = samp_neyman))
}

####### STARTING SIMULATION.  SAVING FILES ########
if(!dir.exists('./data/Sample')){system('mkdir ./data/Sample')}
if(!dir.exists('./data/Sample/Balance')){system('mkdir ./data/Sample/Balance')}
if(!dir.exists('./data/Sample/Neyman')){system('mkdir ./data/Sample/Neyman')}
replicate <- 1000
seed <- 1
for (i in 1:replicate){
  digit <- stringr::str_pad(i, 4, pad = 0)
  cat("Current:", digit, "\n")
  load(paste0("./data/TRUE/", digit, ".RData"))
  samp_result <- generateSample(data, 0.05, seed)
  write.csv(samp_result$samp_balance, 
            file = paste0("./data/Sample/Balance/", digit, ".csv"))
  write.csv(samp_result$samp_neyman, 
            file = paste0("./data/Sample/Neyman/", digit, ".csv"))
  seed <- seed + 1
}


