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
                                      device = "cpu", epochs = 10000,
                                      params = list(batch_size = 500, pac = 10,
                                                    lambda = 10, lr_g = 2e-4, lr_d = 2e-4, 
                                                    n_g_layers = 5, n_d_layers = 1, noise_dim = 128,
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

cox.step <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                    rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                    RACE + I(BMI / 5) + SMOKE, data = match_types(megans_imp$step_result[[1]], data))
megans_imp$imputation <- lapply(megans_imp$imputation, function(dat){
  match_types(dat, data)
})
imp.mids <- as.mids(megans_imp$imputation)
fit <- with(data = imp.mids, 
            exp = coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                          rs4506565 + I((AGE - 50) / 5) + 
                          SEX + INSURANCE + 
                          RACE + I(BMI / 5) + SMOKE))
pooled <- mice::pool(fit)
sumry <- summary(pooled, conf.int = TRUE)
exp(sumry$estimate) - exp(coef(cox.true))
exp(coef(cox.step)) - exp(coef(cox.true))

digit = "0008"
load(paste0("./data/Sample/ATTN/", digit, ".RData"))
ggplot(megans_imp$imputation[[1]]) + 
  geom_density(aes(x = T_I), colour = "red") +
  geom_density(aes(x = T_I), data = data)

ggplot(megans_imp$imputation[[1]]) + 
  geom_density(aes(x = HbA1c), colour = "red") +
  geom_density(aes(x = HbA1c), data = data)

ggplot(megans_imp$imputation[[1]]) + 
  geom_density(aes(x = T_I_STAR - T_I), colour = "red") + 
  geom_density(aes(x = T_I_STAR - T_I), data = data)


ggplot(megans_imp$imputation[[1]]) + 
  geom_density(aes(x = HbA1c_STAR - HbA1c), colour = "red") + 
  geom_density(aes(x = HbA1c_STAR - HbA1c), data = data)


coefres <- NULL

lapply(c("dplyr", "stringr", "torch", "survival", "ggplot2"), require, character.only = T)
digit <- "0001"
load(paste0("./data/Complete/", digit, ".RData"))
cox.true <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                    rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                    RACE + I(BMI / 5) + SMOKE, data = data)
lapply(paste0("./megans/", list.files("./megans")), source)
source("00_utils_functions.R")
i <- 1
for (i in 1:20){
  digit <- stringr::str_pad(i, 4, pad = 0)
  samp_balance <- read.csv(paste0("./data/Sample/Debug/", digit, ".csv"))
  samp_balance <- match_types(samp_balance, data) %>% 
    mutate(across(all_of(data_info_balance$cat_vars), as.factor, .names = "{.col}"),
           across(all_of(data_info_balance$num_vars), as.numeric, .names = "{.col}"))
  megans_imp <- mmer.impute.cwgangp(samp_balance, m = 20, 
                                    num.normalizing = "mode", 
                                    cat.encoding = "onehot", 
                                    device = "cpu", epochs = 10000,
                                    params = list(batch_size = 500, pac = 10, 
                                                  lambda = 10, lr_g = 2e-4, lr_d = 2e-4, 
                                                  n_g_layers = 5, n_d_layers = 3, noise_dim = 128,
                                                  discriminator_steps = 1, 
                                                  type_g = "mlp", type_d = "saencoder",
                                                  g_dim = 256, d_dim = 256, sn_g = F, sn_d = F), 
                                    type = "mmer",
                                    data_info = data_info_balance, save.step = 5000)
  save(megans_imp, file = paste0("./data/Sample/ATTN/", digit, ".RData"))
  
  megans_imp$imputation <- lapply(megans_imp$imputation, function(dat){
    match_types(dat, data)
  })
  imp.mids <- as.mids(megans_imp$imputation)
  fit <- with(data = imp.mids, 
              exp = coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                            rs4506565 + I((AGE - 50) / 5) + 
                            SEX + INSURANCE + 
                            RACE + I(BMI / 5) + SMOKE))
  pooled <- mice::pool(fit)
  sumry <- summary(pooled, conf.int = TRUE)
  print(exp(sumry$estimate) - exp(coef(cox.true)))
  coeff <- bind_rows(lapply(fit$analyses, function(i){coef(i)}))
  print(apply(exp(coeff), 2, var))
}
a

show_var(imputation.list = megans_imp$imputation, var.name = "T_I",
         original.data = samp_balance)

plot_box(imputation.list = megans_imp$imputation[1:5], var.name = "T_I",
          original.data = samp_balance)

plot_2num(imputation.list = megans_imp$imputation[1:5], var.x = "HbA1c",
          var.y = "T_I", original.data = samp_balance, shape = TRUE)

coeffs <- NULL
vars <- NULL
for (i in 1:16){
  digit <- stringr::str_pad(i, 4, pad = 0)
  load(paste0("./data/Sample/ATTN/", digit, ".RData"))
  
  megans_imp$imputation <- lapply(megans_imp$imputation, function(dat){
    match_types(dat, data)
  })
  imp.mids <- as.mids(megans_imp$imputation)
  fit <- with(data = imp.mids, 
              exp = coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                            rs4506565 + I((AGE - 50) / 5) + 
                            SEX + INSURANCE + 
                            RACE + I(BMI / 5) + SMOKE))
  pooled <- mice::pool(fit)
  sumry <- summary(pooled, conf.int = TRUE)
  coeffs <- rbind(coeffs, exp(sumry$estimate))
  vars <- rbind(vars, apply(bind_rows(lapply(fit$analyses, function(i){exp(coef(i))})), 2, var))
  
}

apply(coeffs, 2, var) / apply(vars, 2, mean) 
vars <- bind_rows(lapply(fit$analyses, function(i){exp(coef(i))}))
apply(vars, 2, var)


load("NutritionalData_0001.RData")
for (i in 1:20){
  digit <- stringr::str_pad(i, 4, pad = 0)
  idx <- sample(1:4000, 3500)
  data <- pop
  data$c_ln_na_true[idx] <- NA
  data$c_ln_k_true[idx] <- NA
  data$c_ln_kcal_true[idx] <- NA
  data$c_ln_protein_true[idx] <- NA
  data$W <- 1
  save(data, file = paste0("./debug/", digit, ".RData"))
}
lapply(c("dplyr", "stringr", "torch", "survival", "ggplot2"), require, character.only = T)
lapply(paste0("./megans/", list.files("./megans")), source)
source("00_utils_functions.R")
cox.true <- glm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + 
                  female + bkg_o + bkg_pr, family = gaussian(), pop)
for (i in 1:20){
  digit <- stringr::str_pad(i, 4, pad = 0)
  load(paste0("./debug/", digit, ".RData"))
  data_info <- list(weight_var = "W", 
                    cat_vars = c("idx", "usborn", "high_chol", "female", "bkg_pr", "bkg_o", "hypertension"),
                    num_vars = c("id", "age", "bmi", "c_age", "c_bmi",
                                 "c_ln_na_true", "c_ln_k_true", "c_ln_kcal_true", "c_ln_protein_true",
                                 "c_ln_na_bio1", "c_ln_k_bio1", "c_ln_kcal_bio1", "c_ln_protein_bio1",
                                 "sbp", "W"),
                    phase2_vars = c("c_ln_na_true", "c_ln_k_true", "c_ln_kcal_true", "c_ln_protein_true"),
                    phase1_vars = c("c_ln_na_bio1", "c_ln_k_bio1", "c_ln_kcal_bio1", "c_ln_protein_bio1"))
  data <- match_types(data, pop) %>% 
    mutate(across(all_of(data_info$cat_vars), as.factor, .names = "{.col}"),
           across(all_of(data_info$num_vars), as.numeric, .names = "{.col}"))
  megans_imp <- mmer.impute.cwgangp(data, m = 20, 
                                    num.normalizing = "mode", 
                                    cat.encoding = "onehot", 
                                    device = "cpu", epochs = 10000,
                                    params = list(batch_size = 500, pac = 10, 
                                                  lambda = 10, lr_g = 2e-4, lr_d = 2e-4, 
                                                  n_g_layers = 3, n_d_layers = 2, noise_dim = 128,
                                                  discriminator_steps = 1, 
                                                  type_g = "mlp", type_d = "encoder",
                                                  g_dim = 256, d_dim = 256, sn_g = F, sn_d = F), 
                                    type = "mmer",
                                    data_info = data_info, save.step = 5000)
  save(megans_imp, file = paste0("./debug/imp_", digit, ".RData"))
  megans_imp$imputation <- lapply(megans_imp$imputation, function(dat){
    match_types(dat, pop)
  })
  imp.mids <- as.mids(megans_imp$imputation)
  fit <- with(data = imp.mids, 
              exp = glm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + 
                          female + bkg_o + bkg_pr))
  pooled <- mice::pool(fit)
  sumry <- summary(pooled, conf.int = TRUE)
  print(sumry$estimate - coef(cox.true))
  coeff <- bind_rows(lapply(fit$analyses, function(i){coef(i)}))
  print(apply(coeff, 2, var))
}


ggplot() + 
  geom_density(aes(x = megans_imp$imputation[[1]]$c_ln_na_true), colour = "blue") + 
  geom_density(aes(x = pop$c_ln_na_true), colour = "red")
