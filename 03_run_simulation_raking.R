lapply(c("survey", "svyVGAM", "dplyr", "stringr"), require, character.only = T)
lapply(paste0("./comparisons/raking/", list.files("./comparisons/raking/")), source)
source("00_utils_functions.R")

if(!dir.exists('./simulations')){dir.create('./simulations')}
if(!dir.exists('./simulations/SRS')){dir.create('./simulations/SRS')}
if(!dir.exists('./simulations/Balance')){dir.create('./simulations/Balance')}
if(!dir.exists('./simulations/Neyman')){dir.create('./simulations/Neyman')}

if(!dir.exists('./simulations/Balance/raking')){dir.create('./simulations/Balance/raking')}
if(!dir.exists('./simulations/Neyman/raking')){dir.create('./simulations/Neyman/raking')}


replicate <- 1000
for (i in 1:replicate){
  digit <- stringr::str_pad(i, 4, pad = 0)
  cat("Current:", digit, "\n")
  load(paste0("./data/Complete/", digit, ".RData"))
  samp_balance <- read.csv(paste0("./data/Sample/Balance/", digit, ".csv"))
  samp_neyman <- read.csv(paste0("./data/Sample/Neyman/", digit, ".csv"))
  
  samp_balance <- match_types(samp_balance, data) %>% 
    mutate(across(all_of(data_info_balance$cat_vars), as.factor, .names = "{.col}"),
           across(all_of(data_info_balance$num_vars), as.numeric, .names = "{.col}"))
  samp_neyman <- match_types(samp_neyman, data) %>% 
    mutate(across(all_of(data_info_neyman$cat_vars), as.factor, .names = "{.col}"),
           across(all_of(data_info_neyman$num_vars), as.numeric, .names = "{.col}"))
  
  rakingest <- calibrateFun(samp_balance)
  save(rakingest, file = paste0("./simulations/Balance/raking/", digit, ".RData"))
  rakingest <- calibrateFun(samp_neyman)
  save(rakingest, file = paste0("./simulations/Neyman/raking/", digit, ".RData"))
}




