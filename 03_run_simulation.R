lapply(c("mice", "mixgb", "dplyr", "stringr"), require, character.only = T)
lapply(paste0("./megans/", list.files("./megans")), source)
lapply(paste0("./comparisons/gain/", list.files("./comparisons/gain/")), source)
lapply(paste0("./comparisons/mice/", list.files("./comparisons/mice/")), source)
lapply(paste0("./comparisons/mixgb/", list.files("./comparisons/mixgb/")), source)
lapply(paste0("./comparisons/raking/", list.files("./comparisons/raking/")), source)

if(!dir.exists('./simulations')){system('mkdir ./simulations')}
if(!dir.exists('./simulations/Balance')){system('mkdir ./simulations/Balance')}
if(!dir.exists('./simulations/Neyman')){system('mkdir ./simulations/Neyman')}

if(!dir.exists('./simulations/Balance/megans')){system('mkdir ./simulations/Balance/megans')}
if(!dir.exists('./simulations/Neyman/megans')){system('mkdir ./simulations/Neyman/megans')}

if(!dir.exists('./simulations/Balance/gain')){system('mkdir ./simulations/Balance/gain')}
if(!dir.exists('./simulations/Neyman/gain')){system('mkdir ./simulations/Neyman/gain')}

if(!dir.exists('./simulations/Balance/mice')){system('mkdir ./simulations/Balance/mice')}
if(!dir.exists('./simulations/Neyman/mice')){system('mkdir ./simulations/Neyman/mice')}

if(!dir.exists('./simulations/Balance/mixgb')){system('mkdir ./simulations/Balance/mixgb')}
if(!dir.exists('./simulations/Neyman/mixgb')){system('mkdir ./simulations/Neyman/mixgb')}

if(!dir.exists('./simulations/Balance/raking')){system('mkdir ./simulations/Balance/raking')}
if(!dir.exists('./simulations/Neyman/raking')){system('mkdir ./simulations/Neyman/raking')}

data_info <- list(weight_var = "W",
                  cat_vars = c("SEX", "RACE", "SMOKE", "EXER", "ALC", "INSURANCE", "REGION", 
                               "URBAN", "INCOME", "MARRIAGE", 
                               "rs10811661", "rs7756992", "rs11708067", "rs17036101", "rs17584499",
                               "rs1111875", "rs4402960", "rs4607103", "rs7754840", "rs9300039",
                               "rs5015480", "rs9465871", "rs4506565", "rs5219", "rs358806", 
                               "HYPERTENSION", 
                               "SMOKE_STAR", "ALC_STAR", "EXER_STAR", "INCOME_STAR",
                               "rs10811661_STAR", "rs7756992_STAR", "rs11708067_STAR", "rs17036101_STAR", "rs17584499_STAR",
                               "rs1111875_STAR", "rs4402960_STAR", "rs4607103_STAR", "rs7754840_STAR", "rs9300039_STAR",
                               "rs5015480_STAR", "rs9465871_STAR", "rs4506565_STAR", "rs5219_STAR", "rs358806_STAR",
                               "EVENT", "EVENT_STAR", "stratum", "R", "W"),
                  num_vars = c("X", "ID", "AGE", "EDU", "HEIGHT", "BMI", "WEIGHT", "CREATININE",
                               "BUN", "URIC_ACID", "HDL", "LDL", "TG", "WBC",
                               "RBC", "Hb", "HCT", "PLATELET", "PT", "Na_INTAKE",          
                               "K_INTAKE", "KCAL_INTAKE", "PROTEIN_INTAKE", "ALT", "AST", "ALP",                
                               "GGT", "BILIRUBIN", "GLUCOSE", "F_GLUCOSE", "HbA1c", "INSULIN",            
                               "ALBUMIN", "GLOBULIN", "FERRITIN", "CRP", "SBP", "DBP",                
                               "PULSE", "PP", "EDU_STAR", "Na_INTAKE_STAR", "K_INTAKE_STAR", "KCAL_INTAKE_STAR",    
                               "PROTEIN_INTAKE_STAR", "GLUCOSE_STAR", "F_GLUCOSE_STAR", "HbA1c_STAR", "INSULIN_STAR", "T_I",                
                               "T_I_STAR", "C", "C_STAR"))
replicate <- 1000
for (i in 1:replicate){
  digit <- stringr::str_pad(i, 4, pad = 0)
  cat("Current:", digit, "\n")
  samp_balance <- read.csv(paste0("./data/Sample/Balance/", digit, ".csv"))
  samp_neyman <- read.csv(paste0("./data/Sample/Neyman/", digit, ".csv"))
  
  samp_balance <- samp_balance %>% 
    mutate(across(all_of(data_info$cat_vars), as.factor, .names = "{.col}"))
  samp_neyman <- samp_neyman %>% 
    mutate(across(all_of(data_info$cat_vars), as.factor, .names = "{.col}"))
  
  # MEGANs:
  megans_imp.balance <- mmer.impute.cwgangp(samp_balance, m = 1, 
                                            num.normalizing = "mode", 
                                            cat.encoding = "onehot", 
                                            device = "cpu", epochs = 5000, 
                                            params = list(batch_size = 500, 
                                                          alpha = 0, beta = 1, 
                                                          discriminator_steps = 3, 
                                                          zeta = 0, at_least_p = 0.5, 
                                                          n_g_layers = 1, n_d_layers = 2,
                                                          type_g = "attn", type_d = "mlp"), 
                                            data_info = data_info, save.step = 1000)
  save(megans_imp.balance, file = paste0("./simulations/Balance/megans/", digit, ".RData"))
  megans_imp.neyman <- mmer.impute.cwgangp(samp_neyman, m = 20, num.normalizing = "mode", cat.encoding = "onehot", 
                                           device = "cpu", epochs = 10000, 
                                           params = list(n_g_layers = 3, n_d_layers = 2, 
                                                         type_g = "mlp", type_d = "mlp"), 
                                           data_info = data_info, save.step = 1000)
  save(megans_imp.neyman, file = paste0("./simulations/Neyman/megans/", digit, ".RData"))
  
  # GAIN:
  gain_imp.balance <- gain(samp_balance, device = "cpu", batch_size = 128, hint_rate = 0.9, 
                           alpha = 10, beta = 1, n = 10000)
  
  gain_imp.neyman <- gain(samp_neyman, device = "cpu", batch_size = 128, hint_rate = 0.9, 
                          alpha = 10, beta = 1, n = 10000)
  # MICE:
  
}
megans_imp.balance <- mmer.impute.cwgangp(samp_balance, m = 1, 
                                          num.normalizing = "mode", 
                                          cat.encoding = "onehot", 
                                          device = "cpu", epochs = 3000, 
                                          params = list(batch_size = 500, 
                                                        alpha = 0, beta = 1, 
                                                        zeta = 0, at_least_p = 0.5, 
                                                        n_g_layers = 1, n_d_layers = 2,
                                                        type_g = "attn", type_d = "mlp"), 
                                          data_info = data_info, save.step = 1000)
library(survival)
samp_srs <- read.csv("./data/SRS_0001.csv")
data_info <- list(weight_var = "W",
                 cat_vars = c("hypertension", "bkg_pr", "bkg_o", "female", 
                              "high_chol", "usborn", "idx", "W", "R"),
                 num_vars = names(samp_srs)[!names(samp_srs) %in% c("hypertension", 
                                                                    "bkg_pr", "bkg_o", "female", 
                                                                    "high_chol", "usborn", "idx", "W", "R")])
megans_imp.srs <- mmer.impute.cwgangp(samp_srs, m = 1, 
                                      num.normalizing = "mode", 
                                      cat.encoding = "onehot", 
                                      device = "cpu", epochs = 3000, 
                                      params = list(batch_size = 500, 
                                                          alpha = 0, beta = 1, 
                                                          discriminator_steps = 3, 
                                                          zeta = 0, at_least_p = 0.5, pac = 5,
                                                          n_g_layers = 1, n_d_layers = 2,
                                                          type_g = "attn", type_d = "mlp",
                                                          token_learn = F), 
                                      data_info = data_info, save.step = 1000)
load("./data/NutritionalData_0001.RData")
imp_mod.1 <- glm(hypertension ~ c_ln_na_true + c_age + c_bmi + 
                   high_chol + usborn + female + bkg_o + bkg_pr, 
                 megans_imp.srs$imputation[[1]], family = binomial())
imp_mod.2 <- glm(sbp ~ c_ln_na_true + c_age + c_bmi + 
                   high_chol + usborn + female + bkg_o + bkg_pr, 
                 megans_imp.srs$imputation[[1]], family = gaussian())
true_mod.1 <- glm(hypertension ~ c_ln_na_true + c_age + c_bmi + 
                   high_chol + usborn + female + bkg_o + bkg_pr, 
                 pop, family = binomial())
true_mod.2 <- glm(sbp ~ c_ln_na_true + c_age + c_bmi + 
                   high_chol + usborn + female + bkg_o + bkg_pr, 
                 pop, family = gaussian())
coef(true_mod.1) - coef(imp_mod.1)
coef(true_mod.2) - coef(imp_mod.2)


load("./data/Complete/0001.RData")
mod.imp <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                   rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                   RACE + I(BMI / 5) + EXER, 
                 data = match_types(megans_imp.balance$imputation[[1]], data))
mod.true <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                    RACE + I(BMI / 5) + EXER, data = data)
exp(coef(mod.imp)) - exp(coef(mod.true))


ggplot(samp_balance) + 
  geom_density(aes(x = HbA1c)) + 
  geom_density(data = megans_imp.balance$step_result[[5]][[1]],
               aes(x = HbA1c), colour = "red") +
  geom_density(data = data, 
               aes(x = HbA1c), colour = "blue") +
  geom_vline(xintercept = 77.08)
