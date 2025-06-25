lapply(c("mice", "dplyr", "stringr"), require, character.only = T)
lapply(paste0("./megans/", list.files("./megans")), source)
source("00_utils_functions.R")

if(!dir.exists('./simulations')){system('mkdir ./simulations')}
if(!dir.exists('./simulations/SRS')){system('mkdir ./simulations/SRS')}
if(!dir.exists('./simulations/Balance')){system('mkdir ./simulations/Balance')}
if(!dir.exists('./simulations/Neyman')){system('mkdir ./simulations/Neyman')}

if(!dir.exists('./simulations/SRS/megans')){system('mkdir ./simulations/SRS/megans')}
if(!dir.exists('./simulations/Balance/megans')){system('mkdir ./simulations/Balance/megans')}
if(!dir.exists('./simulations/Neyman/megans')){system('mkdir ./simulations/Neyman/megans')}


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
  output_list_surv <- list()
  for (i in 1:10){
    output_list_surv[[i]] <- mmer.impute.cwgangp(samp_balance, m = 5, 
                                            num.normalizing = "mode", 
                                            cat.encoding = "token", 
                                            device = "cpu", epochs = 10000,
                                            params = list(beta = 1),
                                            data_info = data_info_balance, save.step = 500)
  }
  save(megans_imp.balance, file = paste0("./simulations/Balance/megans/", digit, ".RData"))
  megans_imp.neyman <- mmer.impute.cwgangp(samp_neyman, m = 20, 
                                           num.normalizing = "mode", 
                                           cat.encoding = "onehot", 
                                           device = "cpu", epochs = 5000, 
                                           data_info = data_info_balance, save.step = 1000)
  save(megans_imp.neyman, file = paste0("./simulations/Neyman/megans/", digit, ".RData"))
}

load("NutritionalData_0001.RData")
nutri <- read.csv("SRS_0001.csv")
data_info <- list(weight_var = "W",
                  cat_vars = c("usborn", "high_chol", "female", "bkg_pr", 
                               "bkg_o", "hypertension", "R", "W", "idx"),
                  num_vars = names(nutri)[!names(nutri) %in% c("W", "usborn", "high_chol", "female", "bkg_pr", 
                                                               "bkg_o", "hypertension", "R", "idx")])

output_list_2500 <- list()
for (i in 1:10){
  output_list_2500[[i]] <- mmer.impute.cwgangp(nutri, m = 5, 
                             num.normalizing = "mode", 
                             cat.encoding = "token", 
                             device = "cpu", epochs = 2500, 
                             params = list(lr_d = 1e-4, lr_g = 5e-4, 
                                           pac = 5, n_g_layers = 1, 
                                           n_d_layers = 3, alpha = 1,
                                           discriminator_steps = 1), 
                             data_info = data_info, save.step = 500)
}
output_list_surv_2500 <- list()
for (i in 1:10){
  output_list_surv_2500[[i]] <- mmer.impute.cwgangp(samp_balance, m = 5, 
                                               num.normalizing = "mode", 
                                               cat.encoding = "token", 
                                               device = "cpu", epochs = 2500,
                                               params = list(lr_d = 1e-4, lr_g = 5e-4, 
                                                             pac = 5, n_g_layers = 1, 
                                                             n_d_layers = 3, alpha = 1,
                                                             discriminator_steps = 1),
                                               data_info = data_info_balance, save.step = 500)
}

output_list_10000 <- list()
for (i in 1:10){
  output_list_10000[[i]] <- mmer.impute.cwgangp(nutri, m = 5, 
                                          num.normalizing = "mode", 
                                          cat.encoding = "token", 
                                          device = "cpu", epochs = 10000, 
                                          params = list(lr_d = 1e-4, lr_g = 5e-4, 
                                                        pac = 5, n_g_layers = 1, 
                                                        n_d_layers = 3, alpha = 1,
                                                        discriminator_steps = 1), 
                                          data_info = data_info, save.step = 500)
}
output_list_surv_10000 <- list()
for (i in 1:10){
  output_list_surv_10000[[i]] <- mmer.impute.cwgangp(samp_balance, m = 5, 
                                               num.normalizing = "mode", 
                                               cat.encoding = "token", 
                                               device = "cpu", epochs = 10000,
                                               params = list(lr_d = 1e-4, lr_g = 5e-4, 
                                                             pac = 5, n_g_layers = 1, 
                                                             n_d_layers = 3, alpha = 1,
                                                             discriminator_steps = 1),
                                               data_info = data_info_balance, save.step = 500)
}

binomial_mat <- linear_mat <- NULL
for (i in 1:10){
  for (j in 1:5){
    mod.1 <- glm(hypertension ~ c_ln_na_true + c_age + c_bmi + high_chol + 
                   usborn + female + bkg_o + bkg_pr, output_list[[i]]$imputation[[j]], family = binomial())
    mod.2 <- glm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + 
                   usborn + female + bkg_o + bkg_pr, output_list[[i]]$imputation[[j]], family = gaussian())
    binomial_mat <- rbind(binomial_mat, coef(mod.1))
    linear_mat <- rbind(linear_mat, coef(mod.2))
  }
}
colMeans(binomial_mat) - coef(mod.3)
colMeans(linear_mat) - coef(mod.4)

mod.1 <- glm(hypertension ~ c_ln_na_true + c_age + c_bmi + high_chol + 
             usborn + female + bkg_o + bkg_pr, out$imputation[[1]], family = binomial())
mod.2 <- glm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + 
               usborn + female + bkg_o + bkg_pr, out$imputation[[1]], family = gaussian())
mod.3 <- glm(hypertension ~ c_ln_na_true + c_age + c_bmi + high_chol + 
               usborn + female + bkg_o + bkg_pr, pop, family = binomial())
mod.4 <- glm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + 
               usborn + female + bkg_o + bkg_pr, pop, family = gaussian())
coef(mod.1) - coef(mod.3)
coef(mod.2) - coef(mod.4)
ggplot(nutri) + 
  geom_density(aes(x = c_ln_na_true)) + 
  geom_density(data = out$step_result[[5]][[1]],
               aes(x = c_ln_na_true), colour = "red") +
  geom_density(data = pop,
               aes(x = c_ln_na_true), colour = "blue") 

ggplot(data = pop) + 
  geom_point(aes(x = c_ln_na_true, y = sbp), colour = "blue", alpha = 0.2) + 
  geom_point(data = out$step_result[[5]][[1]], 
             aes(x = c_ln_na_true, y = sbp), colour = "red", alpha = 0.2)

lapply(1:10, function(i){sd(out$step_result[[i]][[1]]$c_ln_na_true)})
lm(sbp ~ c_ln_na_true, data = pop)
lm(sbp ~ c_ln_na_true, data = out$step_result[[9]][[1]])
library(survival)
load("./data/Complete/0001.RData")
cox_mat <- NULL
for (i in 1:10){
  for (j in 1:5){
    mod.imp <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                       rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                       RACE + I(BMI / 5) + SMOKE, 
                     data = match_types(output_list_surv[[i]]$imputation[[j]], data))
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

### Once we set the start to zero, everyone would have different censoring time.
### Neyman allocation: influence function A ~ Z, to do neyman allocation.
### Tong's github.



