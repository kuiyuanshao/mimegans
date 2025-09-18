lapply(c("dplyr", "stringr", "torch", "survival", "mclust"), require, character.only = T)
files <- list.files("./mimegans", full.names = TRUE, recursive = FALSE)
files <- files[!grepl("tests", files)]
lapply(files, source)
source("00_utils_functions.R")

if(!dir.exists('./simulations')){dir.create('./simulations')}
if(!dir.exists('./simulations/Ablation')){dir.create('./simulations/Ablation')}
if(!dir.exists('./simulations/Ablation/PacGAN')){dir.create('./simulations/Ablation/PacGAN')}
if(!dir.exists('./simulations/Ablation/InfoLoss')){dir.create('./simulations/Ablation/InfoLoss')}
if(!dir.exists('./simulations/Ablation/DropoutG')){dir.create('./simulations/Ablation/DropoutG')}
if(!dir.exists('./simulations/Ablation/CELoss')){dir.create('./simulations/Ablation/CELoss')}
if(!dir.exists('./simulations/Ablation/CatProj')){dir.create('./simulations/Ablation/CatProj')}
if(!dir.exists('./simulations/Ablation/MMER')){dir.create('./simulations/Ablation/MMER')}
if(!dir.exists('./simulations/Ablation/GenLoss')){dir.create('./simulations/Ablation/GenLoss')}

args <- commandArgs(trailingOnly = TRUE)
task_id <- as.integer(ifelse(length(args) >= 1,
                             args[1],
                             Sys.getenv("SLURM_ARRAY_TASK_ID", "1")))
start_rep <- 1
end_rep   <- 500
n_chunks  <- 20
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
for (i in 1:500){
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
  #### PacGAN discriminator
  cat("PacGAN \n")
  if (!file.exists(paste0("simulations/Ablation/PacGAN/", digit, ".RData"))){
    mimegans_imp <- mimegans(samp, m = 20, epochs = 10000,
                             data_info = info, params = list(pac = 1),
                             device = "cpu")
    save(mimegans_imp, file = paste0("simulations/Ablation/PacGAN/", digit, ".RData"))
    show_bias(mimegans_imp)
  }
  #### Information Loss
  cat("Info Loss \n")
  if (!file.exists(paste0("simulations/Ablation/InfoLoss/", digit, ".RData"))){
    mimegans_imp <- mimegans(samp, m = 20, epochs = 10000,
                             data_info = info, params = list(info_loss = F),
                             device = "cpu")
    save(mimegans_imp, file = paste0("simulations/Ablation/InfoLoss/", digit, ".RData"))
    show_bias(mimegans_imp)
  }
  cat("Dropout G \n")
  #### Dropout in Generator
  if (!file.exists(paste0("simulations/Ablation/DropoutG/", digit, ".RData"))){
    mimegans_imp <- mimegans(samp, m = 20, epochs = 10000,
                             data_info = info, params = list(g_dropout = 0),
                             device = "cpu")
    save(mimegans_imp, file = paste0("simulations/Ablation/DropoutG/", digit, ".RData"))
    show_bias(mimegans_imp)
  }
  cat("CE Loss \n")
  #### Loss on Categorical Variables
  if (!file.exists(paste0("simulations/Ablation/CELoss/", digit, ".RData"))){
    mimegans_imp <- mimegans(samp, m = 20, epochs = 10000,
                             data_info = info, params = list(beta = 0),
                             device = "cpu")
    save(mimegans_imp, file = paste0("simulations/Ablation/CELoss/", digit, ".RData"))
    show_bias(mimegans_imp)
  }
  cat("Cat Proj \n")
  #### Category Projection Loss
  if (!file.exists(paste0("simulations/Ablation/CatProj/", digit, ".RData"))){
    mimegans_imp <- mimegans(samp, m = 20, epochs = 10000,
                             data_info = info, params = list(cat = "general"),
                             device = "cpu")
    save(mimegans_imp, file = paste0("simulations/Ablation/CatProj/", digit, ".RData"))
    show_bias(mimegans_imp)
  }
  cat("MMER \n")
  #### Prediction with Measurement Errors
  if (!file.exists(paste0("simulations/Ablation/MMER/", digit, ".RData"))){
    mimegans_imp <- mimegans(samp, m = 20, epochs = 10000,
                             data_info = info, params = list(num = "general"),
                             device = "cpu")
    save(mimegans_imp, file = paste0("simulations/Ablation/MMER/", digit, ".RData"))
    show_bias(mimegans_imp)
  }
  cat("Gen Loss \n")
  #### Generator Loss
  if (!file.exists(paste0("simulations/Ablation/GenLoss/", digit, ".RData"))){
    mimegans_imp <- mimegans(samp, m = 20, epochs = 10000,
                             data_info = info, params = list(component = "gen_loss"),
                             device = "cpu")
    save(mimegans_imp, file = paste0("simulations/Ablation/GenLoss/", digit, ".RData"))
    show_bias(mimegans_imp)
  }
}
