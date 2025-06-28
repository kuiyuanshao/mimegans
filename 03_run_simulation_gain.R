lapply(c("dplyr", "stringr"), require, character.only = T)
lapply(paste0("./comparisons/gain/", list.files("./comparisons/gain/")), source)
source("00_utils_functions.R")

if(!dir.exists('./simulations')){dir.create('./simulations')}
if(!dir.exists('./simulations/SRS')){dir.create('./simulations/SRS')}
if(!dir.exists('./simulations/Balance')){dir.create('./simulations/Balance')}
if(!dir.exists('./simulations/Neyman')){dir.create('./simulations/Neyman')}

if(!dir.exists('./simulations/SRS/gain')){dir.create('./simulations/SRS/gain')}
if(!dir.exists('./simulations/Balance/gain')){dir.create('./simulations/Balance/gain')}
if(!dir.exists('./simulations/Neyman/gain')){dir.create('./simulations/Neyman/gain')}


replicate <- 1000
for (i in 1:replicate){
  digit <- stringr::str_pad(i, 4, pad = 0)
  cat("Current:", digit, "\n")
  load(paste0("./data/Complete/", digit, ".RData"))
  samp_srs <- read.csv(paste0("./data/Sample/SRS/", digit, ".csv"))
  samp_balance <- read.csv(paste0("./data/Sample/Balance/", digit, ".csv"))
  samp_neyman <- read.csv(paste0("./data/Sample/Neyman/", digit, ".csv"))
  
  samp_srs <- match_types(samp_srs, data) %>% 
    mutate(across(all_of(data_info_srs$cat_vars), as.factor, .names = "{.col}"),
           across(all_of(data_info_srs$num_vars), as.numeric, .names = "{.col}"))
  samp_balance <- match_types(samp_balance, data) %>% 
    mutate(across(all_of(data_info_balance$cat_vars), as.factor, .names = "{.col}"),
           across(all_of(data_info_balance$num_vars), as.numeric, .names = "{.col}"))
  samp_neyman <- match_types(samp_neyman, data) %>% 
    mutate(across(all_of(data_info_neyman$cat_vars), as.factor, .names = "{.col}"),
           across(all_of(data_info_neyman$num_vars), as.numeric, .names = "{.col}"))
  
  # GAIN:
  gain_imp.srs <- gain(samp_srs, device = "cpu", batch_size = 128, hint_rate = 0.9, 
                       alpha = 10, beta = 1, n = 10000)
  save(mice_imp.srs, file = paste0("./simulations/SRS/gain/", digit, ".RData"))
  gain_imp.balance <- gain(samp_balance, device = "cpu", batch_size = 128, hint_rate = 0.9, 
                           alpha = 10, beta = 1, n = 10000)
  save(gain_imp.balance, file = paste0("./simulations/Balance/gain/", digit, ".RData"))
  gain_imp.neyman <- gain(samp_neyman, device = "cpu", batch_size = 128, hint_rate = 0.9, 
                          alpha = 10, beta = 1, n = 10000)
  save(gain_imp.neyman, file = paste0("./simulations/Neyman/gain/", digit, ".RData"))
  
  
}