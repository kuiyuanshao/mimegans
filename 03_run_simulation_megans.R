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

args <- commandArgs(trailingOnly = TRUE)
task_id <- as.integer(ifelse(length(args) >= 1,
                             args[1],
                             Sys.getenv("SLURM_ARRAY_TASK_ID", "1")))

replicate <- 500
n_chunks <- 20
chunk_size <- ceiling(replicate / n_chunks)
first_rep <- (task_id - 1) * chunk_size + 1
last_rep <- min(task_id * chunk_size, replicate)

for (i in first_rep:last_rep){
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
  # if (!file.exists(paste0("./simulations/SRS/megans/", digit, ".RData"))){
  #   megans_imp <- mmer.impute.cwgangp(samp_srs, m = 20, 
  #                                     num.normalizing = "mode", 
  #                                     cat.encoding = "onehot", 
  #                                     device = "cpu", epochs = 5000,
  #                                     params = list(lambda = 50), 
  #                                     data_info = data_info_srs, save.step = 20000)
  #   save(megans_imp, file = paste0("./simulations/SRS/megans/", digit, ".RData"))
  # }
  # if (!file.exists(paste0("./simulations/Neyman/megans/", digit, ".RData"))){
    megans_imp <- mmer.impute.cwgangp(samp_balance, m = 20, 
                                      num.normalizing = "mode", 
                                      cat.encoding = "onehot", 
                                      device = "cpu", epochs = 5000,
                                      params = list(batch_size = 500, pac = 10,
                                                    lambda = 10, lr_g = 2e-4, lr_d = 2e-4, 
                                                    n_g_layers = 5, n_d_layers = 3, noise_dim = 128,
                                                    discriminator_steps = 1, type_d = "attn",
                                                    g_dim = 512, d_dim = 512), 
                                      type = "mmer",
                                      data_info = data_info_balance, save.step = 1000)
     save(megans_imp, file = paste0("./simulations/Balance/megans/", digit, ".RData"))
  # }
  # if (!file.exists(paste0("./simulations/Neyman/megans/", digit, ".RData"))){
  #   megans_imp <- mmer.impute.cwgangp(samp_neyman, m = 20, 
  #                                     num.normalizing = "mode", 
  #                                     cat.encoding = "onehot", 
  #                                     device = "cpu", epochs = 5000, 
  #                                     params = list(lambda = 50),
  #                                     data_info = data_info_neyman, save.step = 20000)
  #   save(megans_imp, file = paste0("./simulations/Neyman/megans/", digit, ".RData"))
  # }
}

load(paste0("./data/Complete/", digit, ".RData"))
cox.true <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                    rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                    RACE + I(BMI / 5) + SMOKE, data = data)
imp <- lapply(megans_imp$imputation, function(dat){
  match_types(dat, data)
})
imp.mids <- as.mids(imp)
fit <- with(data = imp.mids, 
            exp = coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                          rs4506565 + I((AGE - 50) / 5) + 
                          SEX + INSURANCE + 
                          RACE + I(BMI / 5) + SMOKE))
pooled <- mice::pool(fit)
sumry <- summary(pooled, conf.int = TRUE)
round(exp(sumry$estimate) - exp(coef(cox.true)), 4)

ggplot(megans_imp$imputation[[2]]) + 
  geom_density(aes(x = T_I), colour = "red") +
  geom_density(aes(x = T_I), data = data)

ggplot(megans_imp$imputation[[2]]) + 
  geom_density(aes(x = HbA1c), colour = "red") +
  geom_density(aes(x = HbA1c), data = data)

ggplot(megans_imp$imputation[[1]]) + 
  geom_density(aes(x = T_I_STAR - T_I), colour = "red") + 
  geom_density(aes(x = T_I_STAR - T_I), data = data)

ggplot(megans_imp$imputation[[1]]) + 
  geom_density(aes(x = HbA1c_STAR - HbA1c), colour = "red") + 
  geom_density(aes(x = HbA1c_STAR - HbA1c), data = data)

coeffs <- bind_rows(lapply(fit$analyses, function(i){sqrt(diag(vcov(i)))}))
vars <- bind_rows(lapply(fit$analyses, function(i){coef(i)}))
21 * apply(vars, 2, var) / 20

coefres <- NULL
for (i in 1:100){
  digit <- stringr::str_pad(i, 4, pad = 0)
  samp_balance <- read.csv(paste0("./data/Sample/Debug/", digit, ".csv"))
  samp_balance <- match_types(samp_balance, data)
  design <- svydesign(ids = ~1, strata = ~STRATA, weights = ~W, 
                      data = samp_balance)
  cox.comp <- svycoxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                         rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                         RACE + I(BMI / 5) + SMOKE, design)
  coefres <- rbind(coefres, coef(cox.comp))
}
101 * apply(coefres, 2, var) / 100
load(paste0("./simulations/Balance/megans/0001.RData"))
imp <- lapply(megans_imp$imputation, function(dat){
  match_types(dat, data)
})
imp.mids <- as.mids(imp)
fit <- with(data = imp.mids, 
            exp = coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                          rs4506565 + I((AGE - 50) / 5) + 
                          SEX + INSURANCE + 
                          RACE + I(BMI / 5) + SMOKE))
apply(bind_rows(lapply(fit$analyses, function(i){coef(i)})), 2, var)

load(paste0("./simulations/Balance/mice/0001.RData"))
mice_imp <- mice::complete(mice_imp, "all")

imp <- lapply(mice_imp, function(dat){
  match_types(dat, data)
})
imp.mids <- as.mids(imp)
fit <- with(data = imp.mids, 
            exp = coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                          rs4506565 + I((AGE - 50) / 5) + 
                          SEX + INSURANCE + 
                          RACE + I(BMI / 5) + SMOKE))
21 * apply(bind_rows(lapply(fit$analyses, function(i){coef(i)})), 2, var) / 20
