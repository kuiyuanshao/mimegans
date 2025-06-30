lapply(c("mice", "dplyr", "stringr", "torch"), require, character.only = T)
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
  megans_imp <- mmer.impute.cwgangp(samp_srs, m = 20, 
                                        num.normalizing = "mode", 
                                        cat.encoding = "onehot", 
                                        device = "cpu", epochs = 5000,
                                        data_info = data_info_srs, save.step = 1000)
  save(megans_imp, file = paste0("./simulations/SRS/megans/", digit, ".RData"))
  
  megans_imp <- mmer.impute.cwgangp(samp_balance, m = 5, 
                                            num.normalizing = "mode", 
                                            cat.encoding = "onehot", 
                                            device = "cpu", epochs = 3000,
                                            params = list(alpha = 1, beta = 1, 
                                                          n_g_layers = 3, lr_g = 1e-4, lr_d = 5e-4,
                                                          type_g = "mmer"),
                                            data_info = data_info_balance, save.step = 500)
  save(megans_imp, file = paste0("./simulations/Balance/megans/", digit, ".RData"))
  
  megans_imp <- mmer.impute.cwgangp(samp_neyman, m = 20, 
                                           num.normalizing = "mode", 
                                           cat.encoding = "onehot", 
                                           device = "cpu", epochs = 5000, 
                                           data_info = data_info_balance, save.step = 1000)
  save(megans_imp, file = paste0("./simulations/Neyman/megans/", digit, ".RData"))
}


library(survival)
load("./data/Complete/0001.RData")
mod.imp <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                    rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                    RACE + I(BMI / 5) + SMOKE, data = match_types(megans_imp$imputation[[1]], data))

mod.true <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                   rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                   RACE + I(BMI / 5) + SMOKE, data = data)
exp(coef(mod.true)) - exp(coef(mod.imp))

nutri_samp <- read.csv("SRS_0001.csv")
nutri_samp$W <- 4
data_info <- list(weight_var = "W", 
                  cat_vars = c("idx", "usborn", "high_chol", "bkg_pr", "bkg_o", "hypertension", "R", "female"),
                  num_vars = names(nutri_samp)[!(names(nutri_samp) %in% c("idx", "usborn", "high_chol", 
                                                                          "female", "bkg_pr", "bkg_o", 
                                                                          "hypertension", "R", "W"))],
                  phase2_vars = c("c_ln_na_true", "c_ln_k_true", "c_ln_kcal_true", "c_ln_protein_true"),
                  phase1_vars = c("c_ln_na_bio1", "c_ln_k_bio1", "c_ln_kcal_bio1", "c_ln_protein_bio1"))
load("NutritionalData_0001.RData")
true.lm <- glm(sbp ~ c_age + c_bmi + c_ln_na_true + high_chol + usborn +
                 female + bkg_pr + bkg_o, family = gaussian(), pop)
true.bn <- glm(hypertension ~ c_age + c_bmi + c_ln_na_true + high_chol + usborn +
                 female + bkg_pr + bkg_o, family = binomial(), pop)
megans_nutri <- mmer.impute.cwgangp(nutri_samp, m = 20, 
                                    num.normalizing = "mode", 
                                    cat.encoding = "onehot", 
                                    device = "cpu", epochs = 5000,
                                    params = list(alpha = 0.01, beta = 1, 
                                                  n_g_layers = 3, lr_g = 2e-4, lr_d = 2e-4,
                                                  type_g = "mmer", pac = 10, lambda = 20),
                                    AIPW = F, 
                                    data_info = data_info, save.step = 500)
imp.mids <- as.mids(megans_nutri$imputation)
fit.lm <- with(data = imp.mids, 
               exp = glm(sbp ~ c_age + c_bmi + c_ln_na_true + high_chol + usborn +
                           female + bkg_pr + bkg_o, family = gaussian()))
fit.bn <- with(data = imp.mids, 
               exp = glm(hypertension ~ c_age + c_bmi + c_ln_na_true + high_chol + usborn +
                           female + bkg_pr + bkg_o, family = binomial()))
pooled.lm <- mice::pool(fit.lm)
pooled.bn <- mice::pool(fit.bn)
sumry.lm <- summary(pooled.lm, conf.int = TRUE)
sumry.bn <- summary(pooled.bn, conf.int = TRUE)
sum(abs(sumry.lm$std.error - sqrt(diag(vcov(true.lm)))))
sum(abs(sumry.bn$std.error - sqrt(diag(vcov(true.bn)))))
sumry.lm$estimate - coef(true.lm)
sumry.bn$estimate - coef(true.bn)

library(ggplot2)
ggplot(nutri_samp) + 
  geom_density(aes(x = c_ln_na_true)) + 
  geom_density(data = megans_nutri$imputation[[1]],
               aes(x = c_ln_na_true), colour = "red") +
  geom_density(data = pop, 
               aes(x = c_ln_na_true), colour = "blue")
ggplot(samp_balance) + 
  geom_density(aes(x = T_I)) + 
  geom_density(data = megans_imp$imputation[[1]],
               aes(x = T_I), colour = "red") +
  geom_density(data = data, 
               aes(x = T_I), colour = "blue")





ggplot(samp_balance) + 
  geom_point(data = megans_imp.balance$step_result[[17]][[1]],
               aes(x = HbA1c, y = T_I), colour = "red", alpha = 0.05)

ggplot() +
  geom_line(aes(x = 1:dim(megans_nutri$loss)[1], y = megans_nutri$loss$`G Loss`), colour = "red") +
  geom_line(aes(x = 1:dim(megans_nutri$loss)[1], y = megans_nutri$loss$`D Loss`), colour = "blue")




