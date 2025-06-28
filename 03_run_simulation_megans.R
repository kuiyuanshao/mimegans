lapply(c("mice", "dplyr", "stringr"), require, character.only = T)
lapply(paste0("./megans/", list.files("./megans")), source)
source("00_utils_functions.R")

if(!dir.exists('./simulations')){dir.create('./simulations')}
if(!dir.exists('./simulations/SRS')){dir.create('./simulations/SRS')}
if(!dir.exists('./simulations/Balance')){dir.create('./simulations/Balance')}
if(!dir.exists('./simulations/Neyman')){dir.create('./simulations/Neyman')}

if(!dir.exists('./simulations/SRS/megans')){dir.create('./simulations/SRS/megans')}
if(!dir.exists('./simulations/Balance/megans')){dir.create('./simulations/Balance/megans')}
if(!dir.exists('./simulations/Neyman/megans')){dir.create('./simulations/Neyman/megans')}


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
  
  # MEGANs:
  megans_imp.srs <- mmer.impute.cwgangp(samp_srs, m = 20, 
                                        num.normalizing = "mode", 
                                        cat.encoding = "onehot", 
                                        device = "cpu", epochs = 5000, , 
                                        data_info = data_info_srs, save.step = 1000)
  save(megans_imp.srs, file = paste0("./simulations/SRS/megans/", digit, ".RData"))
  
  megans_imp.balance <- mmer.impute.cwgangp(samp_balance, m = 5, 
                                            num.normalizing = "mode", 
                                            cat.encoding = "token", 
                                            device = "cpu", epochs = 2500,
                                            params = list(beta = 1),
                                            data_info = data_info_balance, save.step = 500)
  save(megans_imp.balance, file = paste0("./simulations/Balance/megans/", digit, ".RData"))
  
  megans_imp.neyman <- mmer.impute.cwgangp(samp_neyman, m = 20, 
                                           num.normalizing = "mode", 
                                           cat.encoding = "onehot", 
                                           device = "cpu", epochs = 5000, 
                                           data_info = data_info_balance, save.step = 1000)
  save(megans_imp.neyman, file = paste0("./simulations/Neyman/megans/", digit, ".RData"))
}


library(survival)
load("./data/Complete/0001.RData")
cox_mat <- NULL
for (i in 1:10){
  for (j in 1:5){
    mod.imp <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                       rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                       RACE + I(BMI / 5) + SMOKE, 
                     data = match_types(output_list_surv_10000[[i]]$imputation[[j]], data))
    cox_mat <- rbind(cox_mat, exp(coef(mod.imp)))
  }
}
colMeans(cox_mat) - exp(coef(mod.true))
mod.true <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                    rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                    RACE + I(BMI / 5) + SMOKE, data = data)
exp(coef(mod.imp)) - exp(coef(mod.true))

table(megans_imp.balance$imputation[[5]]$EVENT)

library(ggplot2)
ggplot(samp_balance) + 
  geom_density(aes(x = HbA1c)) + 
  geom_density(data = megans_imp.balance$gsample[[1]],
               aes(x = HbA1c), colour = "red") +
  geom_density(data = data, 
               aes(x = HbA1c), colour = "blue") +
  geom_vline(xintercept = 77.08)
ggplot(samp_balance) + 
  geom_density(aes(x = T_I)) + 
  geom_density(data = megans_imp.balance$imputation[[1]],
               aes(x = T_I), colour = "red") +
  geom_density(data = data, 
               aes(x = T_I), colour = "blue")




ggplot(samp_balance) + 
  geom_point(data = megans_imp.balance$step_result[[17]][[1]],
               aes(x = HbA1c, y = T_I), colour = "red", alpha = 0.05)

ggplot() +
  geom_line(aes(x = 1:dim(megans_imp.balance$loss)[1], y = megans_imp.balance$loss$`G Loss`), colour = "red") +
  geom_line(aes(x = 1:dim(megans_imp.balance$loss)[1], y = megans_imp.balance$loss$`D Loss`), colour = "blue")




