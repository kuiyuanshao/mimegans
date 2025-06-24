lapply(c("mixgb", "dplyr", "stringr"), require, character.only = T)
lapply(paste0("./comparisons/mixgb/", list.files("./comparisons/mixgb/")), source)
source("00_utils_functions.R")

if(!dir.exists('./simulations')){system('mkdir ./simulations')}
if(!dir.exists('./simulations/SRS')){system('mkdir ./simulations/SRS')}
if(!dir.exists('./simulations/Balance')){system('mkdir ./simulations/Balance')}
if(!dir.exists('./simulations/Neyman')){system('mkdir ./simulations/Neyman')}

if(!dir.exists('./simulations/SRS/mixgb')){system('mkdir ./simulations/SRS/mixgb')}
if(!dir.exists('./simulations/Balance/mixgb')){system('mkdir ./simulations/Balance/mixgb')}
if(!dir.exists('./simulations/Neyman/mixgb')){system('mkdir ./simulations/Neyman/mixgb')}


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
  
  
  
}



