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
                                                  type_g = "mlp", type_d = "sattn",
                                                  g_dim = 256, d_dim = 256), 
                                    type = "mmer",
                                    data_info = data_info_balance, save.step = 10001)
  save(megans_imp, file = paste0("./data/Sample/ScaleATTN/", digit, ".RData"))
  
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
load(paste0("./data/Sample/ScaleATTN/", "0001", ".RData"))
ggplot() +
  geom_line(aes(x = 1:10000, y = megans_imp$loss[, 2]))
show_var(imputation.list = megans_imp$imputation, var.name = "T_I",
         original.data = samp_balance)

plot_box(imputation.list = megans_imp$imputation[1:5], var.name = "T_I",
          original.data = samp_balance)

plot_2num(imputation.list = megans_imp$imputation[1:5], var.x = "HbA1c",
          var.y = "T_I", original.data = samp_balance, shape = TRUE)

coeffs <- NULL
vars <- NULL
for (i in 1:10){
  digit <- stringr::str_pad(i, 4, pad = 0)
  load(paste0("./data/Sample/ScaleATTN/", digit, ".RData"))
  
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
  idx <- sample(1:4000, 3000)
  data <- pop
  data$c_ln_na_true[idx] <- NA
  data$c_ln_k_true[idx] <- NA
  data$c_ln_kcal_true[idx] <- NA
  data$c_ln_protein_true[idx] <- NA
  data$W <- 1
  data$R <- 1
  data$R[idx] <- 0
  save(data, file = paste0("./debug/", digit, ".RData"))
}
lapply(c("dplyr", "stringr", "torch", "ggplot2", "mice"), require, character.only = T)
lapply(paste0("./megans/", list.files("./megans")), source)
source("00_utils_functions.R")
cox.true.lm <- glm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + 
                     female + bkg_o + bkg_pr, family = gaussian(), pop)
cox.true.bn <- glm(hypertension ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + 
                     female + bkg_o + bkg_pr, family = binomial(), pop)
for (i in 1:20){
  digit <- stringr::str_pad(i, 4, pad = 0)
  load(paste0("./debug/", digit, ".RData"))
  data_info <- list(weight_var = "W", 
                    cat_vars = c("idx", "usborn", "high_chol", "female", "bkg_pr", "bkg_o", "hypertension", "R"),
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
                                    device = "cpu", epochs = 5000,
                                    params = list(batch_size = 500, pac = 10, 
                                                  lambda = 10, lr_g = 1e-4, lr_d = 1e-4,
                                                  n_g_layers = 3, n_d_layers = 1, noise_dim = 64,
                                                  discriminator_steps = 1,
                                                  type_g = "mlp", type_d = "sattn",
                                                  g_dim = 256, d_dim = 256),
                                    type = "mmer",
                                    data_info = data_info, save.step = 10001)
  save(megans_imp, file = paste0("./debug/imp_", digit, ".RData"))
  inc <- c("c_age", "c_bmi", "c_ln_na_bio1",
           "high_chol", "usborn",
           "female", "bkg_pr", "bkg_o", "sbp", "hypertension")
  mice_imp <- mice(data, m = 20, print = T, maxit = 5, remove.collinear = F, 
                   maxcor = 1.0001, ls.meth = "ridge", ridge = 0.001,
                   predictorMatrix = quickpred(data, include = inc))
  save(mice_imp, file = paste0("./debug/MICEimp_", digit, ".RData"))
  imp.mids <- as.mids(megans_imp$imputation)
  fit.lm <- with(data = imp.mids, 
                 exp = glm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + 
                             female + bkg_o + bkg_pr, family = gaussian()))
  pooled.lm <- mice::pool(fit.lm)
  sumry.lm <- summary(pooled.lm, conf.int = TRUE)
  print(sumry.lm$estimate - coef(cox.true.lm))
  
  fit.bn <- with(data = imp.mids, 
                 exp = glm(hypertension ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + 
                             female + bkg_o + bkg_pr, family = binomial()))
  pooled.bn <- mice::pool(fit.bn)
  sumry.bn <- summary(pooled.bn, conf.int = TRUE)
  print(sumry.bn$estimate - coef(cox.true.bn))
  
  coeff.lm <- bind_rows(lapply(fit.lm$analyses, function(i){coef(i)}))
  print(apply(coeff.lm, 2, var))
  coeff.bn <- bind_rows(lapply(fit.bn$analyses, function(i){coef(i)}))
  print(apply(coeff.bn, 2, var))
}

pacman::p_load("ggplot2", "tidyr", "dplyr", "RColorBrewer", "ggh4x")
coeffs.lm <- NULL
coeffs.bn <- NULL
vars.lm <- NULL
vars.bn <- NULL
combined_df.1 <- combined_df.2 <- NULL
for (i in 1:7){
  digit <- stringr::str_pad(i, 4, pad = 0)
  load(paste0("./debug/imp_", digit, ".RData"))
  load(paste0("./debug/MICEimp_", digit, ".RData"))
  
  imp.mids <- as.mids(megans_imp$imputation)
  fit.lm <- with(data = imp.mids, 
                 exp = glm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + 
                             female + bkg_o + bkg_pr, family = gaussian()))
  pooled.lm <- mice::pool(fit.lm)
  sumry.lm <- summary(pooled.lm, conf.int = TRUE)
  fit.bn <- with(data = imp.mids, 
                 exp = glm(hypertension ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + 
                             female + bkg_o + bkg_pr, family = binomial()))
  pooled.bn <- mice::pool(fit.bn)
  sumry.bn <- summary(pooled.bn, conf.int = TRUE)
  
  
  coeffs.lm <- rbind(coeffs.lm, sumry.lm$estimate)
  coeffs.bn <- rbind(coeffs.bn, sumry.bn$estimate)
  vars.lm <- rbind(vars.lm, apply(bind_rows(lapply(fit.lm$analyses, function(i){coef(i)})), 2, var))
  vars.bn <- rbind(vars.bn, apply(bind_rows(lapply(fit.bn$analyses, function(i){coef(i)})), 2, var))
  
  combined_df.1 <- rbind(combined_df.1, c("/gans", digit, "GANs", "Est", sumry.bn$estimate[2]))
  combined_df.1 <- rbind(combined_df.1, c("/gans", digit, "GANs", "Var", (sumry.bn$std.error^2)[2]))
  combined_df.2 <- rbind(combined_df.2, c("/gans", digit, "GANs", "Est", sumry.lm$estimate[2]))
  combined_df.2 <- rbind(combined_df.2, c("/gans", digit, "GANs", "Var", (sumry.lm$std.error^2)[2]))
  
  fit.lm <- with(data = mice_imp, 
                 exp = glm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + 
                             female + bkg_o + bkg_pr, family = gaussian()))
  pooled.lm <- mice::pool(fit.lm)
  sumry.lm <- summary(pooled.lm, conf.int = TRUE)
  fit.bn <- with(data = mice_imp, 
                 exp = glm(hypertension ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + 
                             female + bkg_o + bkg_pr, family = binomial()))
  pooled.bn <- mice::pool(fit.bn)
  sumry.bn <- summary(pooled.bn, conf.int = TRUE)
  combined_df.1 <- rbind(combined_df.1, c("/pmm", digit, "MICE", "Est", sumry.bn$estimate[2]))
  combined_df.1 <- rbind(combined_df.1, c("/pmm", digit, "MICE", "Var", (sumry.bn$std.error^2)[2]))
  combined_df.2 <- rbind(combined_df.2, c("/pmm", digit, "MICE", "Est", sumry.lm$estimate[2]))
  combined_df.2 <- rbind(combined_df.2, c("/pmm", digit, "MICE", "Var", (sumry.lm$std.error^2)[2]))
}
colMeans(coeffs.lm) -  coef(cox.true.lm)
colMeans(coeffs.bn) - coef(cox.true.bn)
apply(coeffs.lm, 2, var) / apply(vars.lm, 2, mean) 
apply(coeffs.bn, 2, var) / apply(vars.bn, 2, mean) 

combined_df.1 <- as.data.frame(combined_df.1)
names(combined_df.1) <- c("TYPE", "DIGIT", "METHOD", "ESTIMATE", "value")
combined_df.2 <- as.data.frame(combined_df.2)
names(combined_df.2) <- c("TYPE", "DIGIT", "METHOD", "ESTIMATE", "value")
ggplot(combined_df.1) + 
  geom_boxplot(aes(x = METHOD, 
                   y = as.numeric(value), colour = METHOD)) + 
  geom_hline(data = means.1, aes(yintercept = 1.38), linetype = "dashed", color = "black") + 
  facet_wrap(~ESTIMATE, scales = "free", ncol = 1,
             labeller = labeller(ESTIMATE = c(Est = "Coefficient", Var = "Variance"))) + 
  theme_minimal() +
  facetted_pos_scales(y = list(ESTIMATE == "Est" ~ scale_y_continuous(limits = c(0, 2.5)),
                               ESTIMATE == "Var" ~ scale_y_continuous(limits = c(0, 0.09))))
ggplot(combined_df.2) + 
  geom_boxplot(aes(x = METHOD, 
                   y = as.numeric(value), colour = METHOD)) + 
  geom_hline(data = means.1, aes(yintercept = 27.5), linetype = "dashed", color = "black") + 
  facet_wrap(~ESTIMATE, scales = "free", ncol = 1,
             labeller = labeller(ESTIMATE = c(Est = "Coefficient", Var = "Variance"))) + 
  theme_minimal()+ 
  facetted_pos_scales(y = list(ESTIMATE == "Est" ~ scale_y_continuous(limits = c(20, 33)),
                               ESTIMATE == "Var" ~ scale_y_continuous(limits = c(0, 2))))




load("result_miceTestRej_imputation.RData")
library(tidyr)
combined_df.1 <- bind_rows(result_df.1) %>% 
  filter(grepl("^c_ln_na_true", rownames(.))) %>%
  pivot_longer(
    cols = 1:6,
    names_to = c("METHOD", "ESTIMATE"),
    names_pattern = "^(.*)\\.(Est|Var)$"
  ) %>% 
  mutate(METHOD = case_when(
    METHOD == "MICE.imp" & TYPE == "/pmm"  ~ "MICE.IMP.PMM",
    METHOD == "MICE.imp" & TYPE == "/norm"  ~ "MICE.IMP.NORM",
    METHOD == "MICE.imp" & TYPE == "/wnorm" ~ "MICE.IMP.PWLS",
    METHOD == "MICE.imp" & TYPE == "/cml" ~ "MICE.IMP.CML",
    METHOD == "MICE.imp" & TYPE == "/cml_rejsamp" ~ "MICE.IMP.CML_REJ",
    TRUE ~ METHOD
  ))

combined_df.2 <- bind_rows(result_df.2) %>% 
  filter(grepl("^c_ln_na_true", rownames(.))) %>%
  pivot_longer(
    cols = 1:6,
    names_to = c("METHOD", "ESTIMATE"),
    names_pattern = "^(.*)\\.(Est|Var)$"
  ) %>% 
  mutate(METHOD = case_when(
    METHOD == "MICE.imp" & TYPE == "/pmm"  ~ "MICE.IMP.PMM",
    METHOD == "MICE.imp" & TYPE == "/norm"  ~ "MICE.IMP.NORM",
    METHOD == "MICE.imp" & TYPE == "/wnorm" ~ "MICE.IMP.PWLS",
    METHOD == "MICE.imp" & TYPE == "/cml" ~ "MICE.IMP.CML",
    METHOD == "MICE.imp" & TYPE == "/cml_rejsamp" ~ "MICE.IMP.CML_REJ",
    TRUE ~ METHOD
  ))

means.1 <- combined_df.1 %>% 
  dplyr::filter(METHOD == "TRUE") %>%
  aggregate(as.numeric(value) ~ ESTIMATE, data = ., FUN = mean)

means.2 <- combined_df.2 %>% 
  dplyr::filter(METHOD == "TRUE") %>%
  aggregate(as.numeric(value) ~ ESTIMATE, data = ., FUN = mean)

pacman::p_load("ggplot2", "tidyr", "dplyr", "RColorBrewer", "ggh4x")
ggplot(combined_df.1) + 
  geom_boxplot(aes(x = factor(METHOD, levels = c("TRUE", "COMPL", "MICE.IMP.PMM", "MICE.IMP.NORM", "MICE.IMP.PWLS", "MICE.IMP.CML", "MICE.IMP.CML_REJ", "GANs")), 
                   y = as.numeric(value), colour = factor(METHOD, levels = c("TRUE", "COMPL", "MICE.IMP.PMM", "MICE.IMP.NORM", "MICE.IMP.PWLS", "MICE.IMP.CML", "MICE.IMP.CML_REJ", "GANs")))) + 
  geom_hline(data = means.1, aes(yintercept = `as.numeric(value)`), linetype = "dashed", color = "black") + 
  facet_wrap(~ESTIMATE, scales = "free", ncol = 1,
             labeller = labeller(ESTIMATE = c(Est = "Coefficient", Var = "Variance"))) + 
  theme_minimal() + 
  labs(x = "Methods", y = "Estimate", colour = "Methods") + 
  theme(axis.title.x = element_text(family = "Georgia"),
        axis.title.y = element_text(family = "Georgia"),
        axis.text.x = element_text(family = "Georgia"),
        axis.text.y = element_text(family = "Georgia"),
        legend.title = element_text(family = "Georgia"),
        legend.text = element_text(family = "Georgia"),
        strip.text = element_text(family = "Georgia")) +
  scale_x_discrete(labels = c("TRUE" = "True",  "COMPL" = "Complete-Case",
                              "MICE.IMP.PMM" = "PMM", 
                              "MICE.IMP.NORM" = "NORM", 
                              "MICE.IMP.PWLS" = "PWLS",
                              "MICE.IMP.CML" = "CML",
                              "MICE.IMP.CML_REJ" = "CML_REJ",
                              "GANs" = "GANs")) +
  scale_color_brewer(palette = "Paired") +
  facetted_pos_scales(y = list(ESTIMATE == "Est" ~ scale_y_continuous(limits = c(0, 2.5)),
                               ESTIMATE == "Var" ~ scale_y_continuous(limits = c(0, 0.09))))


ggplot(combined_df.2) + 
  geom_boxplot(aes(x = factor(METHOD, levels = c("TRUE", "COMPL", "MICE.IMP.PMM", "MICE.IMP.NORM", "MICE.IMP.PWLS", "MICE.IMP.CML", "MICE.IMP.CML_REJ", "GANs")), 
                   y = as.numeric(value), colour = factor(METHOD, levels = c("TRUE", "COMPL", "MICE.IMP.PMM", "MICE.IMP.NORM", "MICE.IMP.PWLS", "MICE.IMP.CML", "MICE.IMP.CML_REJ", "GANs")))) + 
  geom_hline(data = means.2, aes(yintercept = `as.numeric(value)`), linetype = "dashed", color = "black") + 
  facet_wrap(~ESTIMATE, scales = "free", ncol = 1,
             labeller = labeller(ESTIMATE = c(Est = "Coefficient", Var = "Variance"))) + 
  theme_minimal() + 
  labs(x = "Methods", y = "Estimate", colour = "Methods") + 
  theme(axis.title.x = element_text(family = "Georgia"),
        axis.title.y = element_text(family = "Georgia"),
        axis.text.x = element_text(family = "Georgia"),
        axis.text.y = element_text(family = "Georgia"),
        legend.title = element_text(family = "Georgia"),
        legend.text = element_text(family = "Georgia"),
        strip.text = element_text(family = "Georgia")) +
  scale_x_discrete(labels = c("TRUE" = "True",  "COMPL" = "Complete-Case",
                              "MICE.IMP.PMM" = "PMM", 
                              "MICE.IMP.NORM" = "NORM", 
                              "MICE.IMP.PWLS" = "PWLS",
                              "MICE.IMP.CML" = "CML",
                              "MICE.IMP.CML_REJ" = "CML_REJ",
                              "GANs" = "GANs")) +
  scale_color_brewer(palette = "Paired") +
  facetted_pos_scales(y = list(ESTIMATE == "Est" ~ scale_y_continuous(limits = c(20, 33)),
                               ESTIMATE == "Var" ~ scale_y_continuous(limits = c(0, 2))))
