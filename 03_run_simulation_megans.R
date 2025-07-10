lapply(c("dplyr", "stringr", "torch"), require, character.only = T)
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
                                    device = "cpu", epochs = 7500,
                                    params = list(lambda = 50), 
                                    data_info = data_info_srs, save.step = 1000)
  save(megans_imp, file = paste0("./simulations/SRS/megans/", digit, ".RData"))
  
  megans_imp <- mmer.impute.cwgangp(samp_balance, m = 20, 
                                    num.normalizing = "mode", 
                                    cat.encoding = "onehot", 
                                    device = "cpu", epochs = 7500,
                                    params = list(lambda = 50),
                                    data_info = data_info_balance, save.step = 1000)
  save(megans_imp, file = paste0("./simulations/Balance/megans/", digit, ".RData"))
  
  megans_imp <- mmer.impute.cwgangp(samp_neyman, m = 20, 
                                    num.normalizing = "mode", 
                                    cat.encoding = "onehot", 
                                    device = "cpu", epochs = 7500, 
                                    params = list(lambda = 100),
                                    data_info = data_info_neyman, save.step = 1000)
  save(megans_imp, file = paste0("./simulations/Neyman/megans/", digit, ".RData"))
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
round(exp(coef(mod.true)) - exp(sumry.lm$estimate), 2)

data_original <- samp_balance
gsamples <- megans_imp$gsample[[1]]
imp <- megans_imp$imputation[[1]]
vars_to_pmm <- "T_I"
if (!is.null(vars_to_pmm)){
  for (i in vars_to_pmm){
      pmm_matched <- pmm(gsamples[data_original$R == 1, i],
                         gsamples[data_original$R == 0, i],
                         data_original[data_original$R == 1, i], 5)
      imp[data_original$R == 0, i] <- pmm_matched
  }
}
mod.gsamp <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                    rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                    RACE + I(BMI / 5) + SMOKE, data = match_types(imp, data))
exp(coef(mod.gsamp)) - exp(coef(mod.true))
ggplot(samp_balance) + 
  geom_density(aes(x = HbA1c_STAR - HbA1c)) + 
  geom_density(data = megans_imp$imputation[[2]],
               aes(x = HbA1c_STAR - HbA1c), colour = "red") +
  geom_density(data = data, 
               aes(x = HbA1c_STAR - HbA1c), colour = "blue")

ggplot(samp_balance) + 
  geom_density(aes(x = T_I)) + 
  geom_density(data = megans_imp$imputation[[1]],
               aes(x = T_I), colour = "red") +
  geom_density(data = data, 
               aes(x = T_I), colour = "blue")

ggplot(samp_balance) + 
  geom_density(aes(x = HbA1c)) + 
  geom_density(data = megans_imp$imputation[[1]],
               aes(x = HbA1c), colour = "red") +
  geom_density(data = data, 
               aes(x = HbA1c), colour = "blue")

ggplot(samp_balance) + 
  geom_point(aes(x = T_I, y = HbA1c)) + 
  geom_point(data = megans_imp$imputation[[1]],
               aes(x = T_I, y = HbA1c), colour = "red") +
  geom_point(data = data, 
               aes(x = T_I, y = HbA1c), colour = "blue", alpha = 0.5)

