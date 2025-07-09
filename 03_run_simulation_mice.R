lapply(c("mice", "dplyr", "stringr"), require, character.only = T)
source("00_utils_functions.R")

if(!dir.exists('./simulations')){dir.create('./simulations')}
if(!dir.exists('./simulations/SRS')){dir.create('./simulations/SRS')}
if(!dir.exists('./simulations/Balance')){dir.create('./simulations/Balance')}
if(!dir.exists('./simulations/Neyman')){dir.create('./simulations/Neyman')}

if(!dir.exists('./simulations/SRS/mice')){dir.create('./simulations/SRS/mice')}
if(!dir.exists('./simulations/Balance/mice')){dir.create('./simulations/Balance/mice')}
if(!dir.exists('./simulations/Neyman/mice')){dir.create('./simulations/Neyman/mice')}

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
  
  # MICE:
  mice_imp <- mice(samp_srs, m = 20, print = F, maxit = 50, 
                   maxcor = 1.0001, ls.meth = "ridge", ridge = 0.01, 
                   predictorMatrix = quickpred(samp_srs, mincor = 0.15))
  save(mice_imp, file = paste0("./simulations/SRS/mice/", digit, ".RData"))
  mice_imp <- mice(samp_balance, m = 20, print = F, maxit = 50, 
                   maxcor = 1.0001, ls.meth = "ridge", ridge = 0.01, 
                   predictorMatrix = quickpred(samp_balance, mincor = 0.15))
  save(mice_imp, file = paste0("./simulations/Balance/mice/", digit, ".RData"))
  mice_imp <- mice(samp_neyman, m = 20, print = F, maxit = 50, 
                   maxcor = 1.0001, ls.meth = "ridge", ridge = 0.01, 
                   predictorMatrix = quickpred(samp_neyman, mincor = 0.15))
  save(mice_imp, file = paste0("./simulations/Neyman/mice/", digit, ".RData"))
  
}


library(survival)
load("./data/Complete/0001.RData")
megans_imp$imputation <- lapply(megans_imp$imputation, function(i){
  match_types(i, data)
})
imp.mids <- as.mids(megans_imp$imputation)
fit.cox <- with(data = imp.mids, 
                exp = coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                              rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                              RACE + I(BMI / 5) + SMOKE))
pooled.cox <- mice::pool(fit.cox)
sumry.lm <- summary(pooled.cox, conf.int = TRUE)
mod.true <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                    rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                    RACE + I(BMI / 5) + SMOKE, data = data)
exp(coef(mod.true)) - exp(sumry.lm$estimate)