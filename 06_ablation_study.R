lapply(c("dplyr", "stringr", "torch", "survival", "mclust", "RColorBrewer", "ggh4x", "tidyr"), require, character.only = T)
files <- list.files("./mimegans", full.names = TRUE, recursive = FALSE)
files <- files[!grepl("tests", files)]
lapply(files, source)
source("00_utils_functions.R")

if(!dir.exists('./simulations')){dir.create('./simulations')}
if(!dir.exists('./simulations/Ablation')){dir.create('./simulations/Ablation')}
if(!dir.exists('./simulations/Ablation/Base')){dir.create('./simulations/Ablation/Base')}
if(!dir.exists('./simulations/Ablation/CGAN')){dir.create('./simulations/Ablation/CGAN')}
if(!dir.exists('./simulations/Ablation/PacGAN')){dir.create('./simulations/Ablation/PacGAN')}
if(!dir.exists('./simulations/Ablation/BalSamp')){dir.create('./simulations/Ablation/BalSamp')}
if(!dir.exists('./simulations/Ablation/InfoLoss')){dir.create('./simulations/Ablation/InfoLoss')}
if(!dir.exists('./simulations/Ablation/CELoss')){dir.create('./simulations/Ablation/CELoss')}
if(!dir.exists('./simulations/Ablation/CatProj')){dir.create('./simulations/Ablation/CatProj')}
if(!dir.exists('./simulations/Ablation/MMER')){dir.create('./simulations/Ablation/MMER')}
if(!dir.exists('./simulations/Ablation/CondLossV1')){dir.create('./simulations/Ablation/CondLossV1')}
if(!dir.exists('./simulations/Ablation/CondLossV2')){dir.create('./simulations/Ablation/CondLossV2')}

args <- commandArgs(trailingOnly = TRUE)
task_id <- as.integer(ifelse(length(args) >= 1,
                             args[1],
                             Sys.getenv("SLURM_ARRAY_TASK_ID", "1")))
start_rep <- 1
end_rep   <- 100
n_chunks  <- 100
task_id   <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID"))

n_in_window <- end_rep - start_rep + 1L
chunk_size  <- ceiling(n_in_window / n_chunks)

first_rep <- start_rep + (task_id - 1L) * chunk_size
last_rep  <- min(start_rep + task_id * chunk_size - 1L, end_rep)

### Used the Neyman Allocation Set to Perform
info <- data_info_neyman

show_bias <- function(mimegans_imp){
  mimegans_imp$imputation <- lapply(mimegans_imp$imputation, function(dat){
    match_types(dat, data)
  })
  imp.mids <- as.mids(mimegans_imp$imputation)
  cox.fit <- with(data = imp.mids, 
                  exp = coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) +
                                rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                                SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE))
  pooled <- mice::pool(cox.fit)
  sumry <- summary(pooled, conf.int = TRUE)
  cat("Bias: \n")
  cat(exp(sumry$estimate) - exp(coef(cox.true)), "\n")
  cat("Variance: \n")
  cat(apply(bind_rows(lapply(cox.fit$analyses, function(i){exp(coef(i))})), 2, var), "\n")
}

for (i in 1:100){
  digit <- stringr::str_pad(i, 4, pad = 0)
  cat("Current:", digit, "\n")
  load(paste0("./data/Complete/", digit, ".RData"))
  
  cox.true <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                      rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) + 
                      SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE, data = data)
  
  samp <- read.csv(paste0("./data/Sample/Neyman/", digit, ".csv"))
  samp <- match_types(samp, data) %>% 
    mutate(across(all_of(data_info_neyman$cat_vars), as.factor, .names = "{.col}"),
           across(all_of(data_info_neyman$num_vars), as.numeric, .names = "{.col}"))
  #### Base
  cat("Base \n")
  if (!file.exists(paste0("./simulations/Ablation/Base/", digit, ".RData"))){
    mimegans_imp <- mimegans(samp, m = 20, epochs = 10000,
                             data_info = info, params = list(unconditional = T, balancebatch = F, 
                                                             pac = 1, info_loss = F, beta = 0, 
                                                             num = "general", cat = "general"),
                             device = "cpu")
    save(mimegans_imp, file = paste0("./simulations/Ablation/Base/", digit, ".RData"))
    show_bias(mimegans_imp)
  }
  #### Conditional
  cat("CGAN \n")
  if (!file.exists(paste0("./simulations/Ablation/CGAN/", digit, ".RData"))){
    mimegans_imp <- mimegans(samp, m = 20, epochs = 10000,
                             data_info = info, params = list(unconditional = F, balancebatch = F, 
                                                             pac = 1, info_loss = F, beta = 0, 
                                                             num = "general", cat = "general"),
                             device = "cpu")
    save(mimegans_imp, file = paste0("./simulations/Ablation/CGAN/", digit, ".RData"))
    show_bias(mimegans_imp)
  }
  #### BalSamp
  cat("BalSamp \n")
  if (!file.exists(paste0("./simulations/Ablation/BalSamp/", digit, ".RData"))){
    mimegans_imp <- mimegans(samp, m = 20, epochs = 10000,
                             data_info = info, params = list(unconditional = T, balancebatch = T,
                                                             pac = 1, info_loss = F, beta = 0, 
                                                             num = "general", cat = "general"),
                             device = "cpu")
    save(mimegans_imp, file = paste0("./simulations/Ablation/BalSamp/", digit, ".RData"))
    show_bias(mimegans_imp)
  }
  #### PacGAN 
  cat("PacGAN \n")
  if (!file.exists(paste0("./simulations/Ablation/PacGAN/", digit, ".RData"))){
    mimegans_imp <- mimegans(samp, m = 20, epochs = 10000,
                             data_info = info, params = list(unconditional = T, balancebatch = F,
                                                             pac = 5, info_loss = F, beta = 0, 
                                                             num = "general", cat = "general"),
                             device = "cpu")
    save(mimegans_imp, file = paste0("./simulations/Ablation/PacGAN/", digit, ".RData"))
    show_bias(mimegans_imp)
  }
  #### Information Loss
  cat("Info Loss \n")
  if (!file.exists(paste0("./simulations/Ablation/InfoLoss/", digit, ".RData"))){
    mimegans_imp <- mimegans(samp, m = 20, epochs = 10000,
                             data_info = info, params = list(unconditional = T, balancebatch = F, 
                                                             pac = 1, info_loss = T, beta = 0, 
                                                             num = "general", cat = "general"),
                             device = "cpu")
    save(mimegans_imp, file = paste0("./simulations/Ablation/InfoLoss/", digit, ".RData"))
    show_bias(mimegans_imp)
  }
  cat("CE Loss \n")
  #### Loss on Categorical Variables
  if (!file.exists(paste0("./simulations/Ablation/CELoss/", digit, ".RData"))){
    mimegans_imp <- mimegans(samp, m = 20, epochs = 10000,
                             data_info = info, params = list(unconditional = T, balancebatch = F, 
                                                             pac = 1, info_loss = F, beta = 1, 
                                                             num = "general", cat = "general"),
                             device = "cpu")
    save(mimegans_imp, file = paste0("./simulations/Ablation/CELoss/", digit, ".RData"))
    show_bias(mimegans_imp)
  }
  cat("Cat Proj \n")
  #### Category Projection Loss
  if (!file.exists(paste0("./simulations/Ablation/CatProj/", digit, ".RData"))){
    mimegans_imp <- mimegans(samp, m = 20, epochs = 10000,
                             data_info = info, params = list(unconditional = T, balancebatch = F, 
                                                             pac = 1, info_loss = F, beta = 1,
                                                             num = "general", cat = "projp1"),
                             device = "cpu")
    save(mimegans_imp, file = paste0("./simulations/Ablation/CatProj/", digit, ".RData"))
    show_bias(mimegans_imp)
  }
  cat("MMER \n")
  #### Prediction without Measurement Errors
  if (!file.exists(paste0("./simulations/Ablation/MMER/", digit, ".RData"))){
    mimegans_imp <- mimegans(samp, m = 20, epochs = 10000,
                             data_info = info, params = list(unconditional = T, balancebatch = F, 
                                                             pac = 1, info_loss = F, beta = 0,
                                                             num = "mmer", cat = "general"),
                             device = "cpu")
    save(mimegans_imp, file = paste0("./simulations/Ablation/MMER/", digit, ".RData"))
    show_bias(mimegans_imp)
  }
  cat("CondLoss \n")
  #### Conditioning Loss V1
  if (!file.exists(paste0("./simulations/Ablation/CondLossV1/", digit, ".RData"))){
    mimegans_imp <- mimegans(samp, m = 20, epochs = 10000,
                             data_info = info, params = list(unconditional = F, balancebatch = T, 
                                                             pac = 1, info_loss = F, beta = 0,
                                                             num = "general", cat = "general",
                                                             component = "cond_lossv1"),
                             device = "cpu")
    save(mimegans_imp, file = paste0("./simulations/Ablation/CondLossV1/", digit, ".RData"))
    show_bias(mimegans_imp)
  }
  cat("CondLoss V2 \n")
  #### Conditioning Loss V2
  if (!file.exists(paste0("./simulations/Ablation/CondLossV2/", digit, ".RData"))){
    mimegans_imp <- mimegans(samp, m = 20, epochs = 10000,
                             data_info = info, params = list(unconditional = F, balancebatch = T, 
                                                             pac = 1, info_loss = F, beta = 0,
                                                             num = "general", cat = "general",
                                                             component = "cond_lossv2"),
                             device = "cpu")
    save(mimegans_imp, file = paste0("./simulations/Ablation/CondLossV2/", digit, ".RData"))
    show_bias(mimegans_imp)
  }
}



options(survey.lonely.psu = "certainty")
methods <- c("Base", "CGAN", "BalSamp", "PacGAN", "InfoLoss", 
             "CELoss", "CatProj", "MMER", "CondLossV1", "CondLossV2")
#resultCoeff <- resultStdError <- resultCI <- NULL

for (i in 1:100){
  digit <- stringr::str_pad(i, 4, pad = 0)
  cat("Current:", digit, "\n")
  load(paste0("./data/Complete/", digit, ".RData"))
  cox.true <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) +
                      rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                      SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE, data = data)
  resultCoeff <- rbind(resultCoeff, c(exp(coef(cox.true)), "TRUE", digit))
  resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(cox.true))), "TRUE", digit))
  resultCI <- rbind(resultCI, c(exp(confint(cox.true)[, 1]), exp(confint(cox.true)[, 2]), "TRUE", digit))
  
  for (k in methods){
    if (!file.exists(paste0("./simulations/Ablation/", k, "/", digit, ".RData"))){
      next
    }
    load(paste0("./simulations/Ablation/", k, "/", digit, ".RData"))
    mimegans_imp$imputation <- lapply(mimegans_imp$imputation, function(dat){
      match_types(dat, data)
    })
    imp.mids <- as.mids(mimegans_imp$imputation)
    cox.fit <- with(data = imp.mids, 
                    exp = coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) +
                                  rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                                  SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE))
    pooled <- mice::pool(cox.fit)
    sumry <- summary(pooled, conf.int = TRUE)
    resultCoeff <- rbind(resultCoeff, c(exp(sumry$estimate), toupper(k), digit))
    resultStdError <- rbind(resultStdError, c(sumry$std.error, toupper(k), digit))
    resultCI <- rbind(resultCI, c(exp(sumry$`2.5 %`), exp(sumry$`97.5 %`), toupper(k), digit))
  }
}

resultCoeff <- as.data.frame(resultCoeff)
names(resultCoeff) <- c(names(coef(cox.true)), "Method", "ID")
resultStdError <- as.data.frame(resultStdError)
names(resultStdError) <- c(names(coef(cox.true)), "Method", "ID")
resultCI <- as.data.frame(resultCI)
names(resultCI) <- c(paste0(names(coef(cox.true)), ".lower"), 
                     paste0(names(coef(cox.true)), ".upper"), "Method", "ID")
save(resultCoeff, resultStdError, resultCI, file = "./simulations/results_ablation.RData")

load("./simulations/results_ablation.RData")

resultCoeff_long <- resultCoeff %>% 
  pivot_longer(
    cols = 1:14,
    names_to = "Covariate", 
    values_to = "Coefficient"
  ) %>%
  mutate(Coefficient = as.numeric(Coefficient), 
         Method = factor(Method, levels = c("TRUE", "BASE", "CGAN", "BALSAMP", "PACGAN", "INFOLOSS", "CELOSS", 
                                            "CATPROJ", "MMER", "CONDLOSSV1", "CONDLOSSV2")),
         Covariate = factor(Covariate, levels = names(resultCoeff)[1:14], labels = 
                              c("HbA1c", "rs4506565 1", "rs4506565 2", "AGE", "eGFR", "SEX TRUE", "INSURANCE TRUE", 
                                "RACE AFR", "RACE AMR", "RACE SAS", "RACE EAS", "BMI", "SMOKE 2", "SMOKE 3")))

means.coef <- resultCoeff_long %>% 
  filter(Method == "TRUE") %>%
  select(-c("Method", "ID")) %>% 
  group_by(Covariate) %>%
  summarise(mean = mean(Coefficient))

true.coeff <- resultCoeff %>% filter(Method == "TRUE") %>%
  select(-c("Method", "ID")) %>%
  mutate(across(everything(), as.numeric))
cols <- intersect(names(true.coeff), names(resultCoeff))
rmse_result <- resultCoeff %>%
  filter(!Method %in% c("TRUE")) %>%
  select(Method, any_of("ID"), all_of(cols)) %>%
  mutate(across(all_of(cols), as.numeric)) %>%
  group_by(Method) %>%
  summarise(across(
    all_of(cols),
    ~ {
      t <- true.coeff[[cur_column()]]
      sqrt(mean((.x - t)^2, na.rm = TRUE))
    }
  ), .groups = "drop")

diffCoeff <- resultCoeff %>%
  filter(!Method %in% c("TRUE")) %>%
  select(Method, any_of("ID"), all_of(cols)) %>%
  mutate(across(all_of(cols), as.numeric)) %>%
  group_by(Method) %>%
  summarise(across(
    all_of(cols),
    ~ {
      t <- true.coeff[[cur_column()]]
      (.x - t)
    }
  ), .groups = "drop")

rmse_result_long <- rmse_result %>% 
  pivot_longer(
    cols = 2:15,
    names_to = "Covariate", 
    values_to = "Coefficient"
  ) %>%
  mutate(Coefficient = as.numeric(Coefficient), 
         Method = factor(Method, levels = c("BASE", "CGAN", "BALSAMP", "PACGAN", "INFOLOSS", "CELOSS", 
                                            "CATPROJ", "MMER", "CONDLOSSV1", "CONDLOSSV2")),
         Covariate = factor(Covariate, levels = names(rmse_result)[2:15], labels = 
                              c("HbA1c", "rs4506565 1", "rs4506565 2", "AGE", "eGFR", "SEX TRUE", "INSURANCE TRUE", 
                                "RACE AFR", "RACE AMR", "RACE SAS", "RACE EAS", "BMI", "SMOKE 2", "SMOKE 3")))

truth <- colMeans(true.coeff)
CIcoverage <- NULL
for (method in unique(resultCI$Method)){
  ind <- which(resultCI$Method == method)
  curr_g <- NULL
  for (i in ind){
    curr.lower <- resultCI[i, 1:14]
    curr.upper <- resultCI[i, 15:28]
    curr_g <- rbind(curr_g, c(calcCICover(truth, curr.lower, curr.upper), method))
  }
  CIcoverage <- rbind(CIcoverage, curr_g)
}
CIcoverage <- as.data.frame(CIcoverage)
names(CIcoverage) <- c(names(resultCoeff)[1:14], "Method")

CIcoverage <- CIcoverage %>%
  select(Method, all_of(cols)) %>%
  mutate(across(all_of(names(.)[2:15]), as.logical)) %>%
  group_by(Method) %>%
  summarise(across(all_of(cols), ~ mean(.x)), .groups = "drop")

CIcoverage_long <- CIcoverage %>%
  pivot_longer(
    cols = 2:15,
    names_to = "Covariate", 
    values_to = "Coverage"
  ) %>%
  mutate(Coverage = as.numeric(Coverage), 
         Method = factor(Method, levels = c("TRUE", "BASE", "CGAN", "BALSAMP", "PACGAN", "INFOLOSS", "CELOSS", 
                                            "CATPROJ", "MMER", "CONDLOSSV1", "CONDLOSSV2")),
         Covariate = factor(Covariate, levels = names(rmse_result)[2:15], labels = 
                              c("HbA1c", "rs4506565 1", "rs4506565 2", "AGE", "eGFR", "SEX TRUE", "INSURANCE TRUE", 
                                "RACE AFR", "RACE AMR", "RACE SAS", "RACE EAS", "BMI", "SMOKE 2", "SMOKE 3")))
range_coef <- list(Covariate == "HbA1c" ~ scale_y_continuous(limits = c(means.coef$mean[1] - 0.15, means.coef$mean[1] + 0.15)),
                   Covariate == "rs4506565 1" ~ scale_y_continuous(limits = c(means.coef$mean[2] - 0.25, means.coef$mean[2] + 0.25)),
                   Covariate == "rs4506565 2" ~ scale_y_continuous(limits = c(means.coef$mean[3] - 0.25, means.coef$mean[3] + 0.25)),
                   Covariate == "AGE" ~ scale_y_continuous(limits = c(means.coef$mean[4] - 0.1, means.coef$mean[4] + 0.1)),
                   Covariate == "eGFR" ~ scale_y_continuous(limits = c(means.coef$mean[5] - 0.1, means.coef$mean[5] + 0.1)),
                   Covariate == "SEX TRUE" ~ scale_y_continuous(limits = c(means.coef$mean[6] - 0.2, means.coef$mean[6] + 0.2)),
                   Covariate == "INSURANCE TRUE" ~ scale_y_continuous(limits = c(means.coef$mean[7] - 0.2, means.coef$mean[7] + 0.2)),
                   Covariate == "RACE AFR" ~ scale_y_continuous(limits = c(means.coef$mean[8] - 0.25, means.coef$mean[8] + 0.25)),
                   Covariate == "RACE AMR" ~ scale_y_continuous(limits = c(means.coef$mean[9] - 0.25, means.coef$mean[9] + 0.25)),
                   Covariate == "RACE SAS" ~ scale_y_continuous(limits = c(means.coef$mean[10] - 0.25, means.coef$mean[10] + 0.25)),
                   Covariate == "RACE EAS" ~ scale_y_continuous(limits = c(means.coef$mean[11] - 0.25, means.coef$mean[11] + 0.25)),
                   Covariate == "BMI" ~ scale_y_continuous(limits = c(means.coef$mean[12] - 0.25, means.coef$mean[12] + 0.25)),
                   Covariate == "SMOKE 2" ~ scale_y_continuous(limits = c(means.coef$mean[13] - 0.25, means.coef$mean[13] + 0.25)),
                   Covariate == "SMOKE 3" ~ scale_y_continuous(limits = c(means.coef$mean[14] - 0.25, means.coef$mean[14] + 0.25)))
ggplot(CIcoverage_long) + 
  geom_col(aes(x = Method, 
               y = Coverage), position = "dodge") + 
  geom_hline(aes(yintercept = 0.95), lty = 2) + 
  facet_wrap(~ Covariate, scales = "free") + 
  theme_minimal() + 
  theme(axis.title.x = element_text(family = "Georgia"),
        axis.title.y = element_text(family = "Georgia"),
        axis.text.x = element_text(family = "Georgia"),
        axis.text.y = element_text(family = "Georgia"),
        legend.title = element_text(family = "Georgia"),
        legend.text = element_text(family = "Georgia"),
        strip.text = element_text(family = "Georgia")) + 
  ylim(0, 1.25)

ggsave("./simulations/Ablation_Imputation_Coverage_Barchart.png", width = 35, height = 10, limitsize = F)

ggplot(rmse_result_long) + 
  geom_col(aes(x = Method, 
               y = Coefficient), position = "dodge") + 
  facet_wrap(~ Covariate, scales = "free") + 
  theme_minimal() + 
  theme(axis.title.x = element_text(family = "Georgia"),
        axis.title.y = element_text(family = "Georgia"),
        axis.text.x = element_text(family = "Georgia"),
        axis.text.y = element_text(family = "Georgia"),
        legend.title = element_text(family = "Georgia"),
        legend.text = element_text(family = "Georgia"),
        strip.text = element_text(family = "Georgia"))

ggsave("./simulations/Ablation_Imputation_Coeff_Barchart.png", width = 35, height = 10, limitsize = F)


ggplot(resultCoeff_long) + 
  geom_boxplot(aes(x = Method, 
                   y = Coefficient)) + 
  geom_hline(data = means.coef, aes(yintercept = mean), lty = 2) + 
  facet_wrap(~ Covariate, scales = "free") + 
  theme_minimal() + 
  theme(axis.title.x = element_text(family = "Georgia"),
        axis.title.y = element_text(family = "Georgia"),
        axis.text.x = element_text(family = "Georgia"),
        axis.text.y = element_text(family = "Georgia"),
        legend.title = element_text(family = "Georgia"),
        legend.text = element_text(family = "Georgia"),
        strip.text = element_text(family = "Georgia")) + 
  facetted_pos_scales(y = range_coef)

ggsave("./simulations/Ablation_Imputation_Coeff_Boxplot.png", width = 35, height = 10, limitsize = F)

ggplot(resultStdError) + 
  geom_boxplot(aes(x = factor(Method, levels = c("TRUE", "BASE", "CGAN", "BALSAMP", "PACGAN", "INFOLOSS", 
                                                 "CELOSS", "CATPROJ", "MMER", "CONDLOSSV1", "CONDLOSSV2")), 
                   y = as.numeric(`I((HbA1c - 50)/5)`))) + 
  theme_minimal() + 
  labs(x = "Methods", y = "Standard Errors") + 
  theme(axis.title.x = element_text(family = "Georgia"),
        axis.title.y = element_text(family = "Georgia"),
        axis.text.x = element_text(family = "Georgia"),
        axis.text.y = element_text(family = "Georgia"),
        legend.title = element_text(family = "Georgia"),
        legend.text = element_text(family = "Georgia"),
        strip.text = element_text(family = "Georgia"))

ggsave("./simulations/Ablation_Imputation_StdError_Boxplot.png", width = 15, height = 10, limitsize = F)

Bias <- cbind(rmse_result[, 1], rowSums(rmse_result[, 7:12]), rowSums(rmse_result[, c(5, 6, 13)]),
              rowSums(rmse_result[, c(2, 3, 14, 15)]), rmse_result[, 2],
              rowSums(rmse_result[, 2:dim(rmse_result)[2]]))
names(Bias) <- c("Method", "Phase-1 Cat", "Phase-1 Num", "Validated Cat", "Validated Num", "Total")
Bias <- Bias[order(Bias$Total), ]
lapply(1:nrow(Bias), function(i) {Bias[i, 2:6] - Bias[Bias$Method == "BASE", 2:6]})


