lapply(c("dplyr", "stringr", "torch", "survival", "mclust"), require, character.only = T)
files <- list.files("./mimegans", full.names = TRUE, recursive = FALSE)
files <- files[!grepl("tests", files)]
lapply(files, source)
source("00_utils_functions.R")
if(!dir.exists('./simulations')){dir.create('./simulations')}
if(!dir.exists('./simulations/SRS')){dir.create('./simulations/SRS')}
if(!dir.exists('./simulations/Balance')){dir.create('./simulations/Balance')}
if(!dir.exists('./simulations/Neyman')){dir.create('./simulations/Neyman')}

if(!dir.exists('./simulations/SRS/mimegans')){dir.create('./simulations/SRS/mimegans')}
if(!dir.exists('./simulations/Balance/mimegans')){dir.create('./simulations/Balance/mimegans')}
if(!dir.exists('./simulations/Neyman/mimegans')){dir.create('./simulations/Neyman/mimegans')}

args <- commandArgs(trailingOnly = TRUE)
task_id <- as.integer(ifelse(length(args) >= 1,
                             args[1],
                             Sys.getenv("SLURM_ARRAY_TASK_ID", "1")))
sampling_design <- ifelse(length(args) >= 2, 
                          args[2], Sys.getenv("SAMP", "All"))
start_rep <- 1
end_rep   <- 500
n_chunks  <- 20
task_id   <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID"))

n_in_window <- end_rep - start_rep + 1L
chunk_size  <- ceiling(n_in_window / n_chunks)

first_rep <- start_rep + (task_id - 1L) * chunk_size
last_rep  <- min(start_rep + task_id * chunk_size - 1L, end_rep)


do_mimegans <- function(samp, info, nm, digit) {
  tm <- system.time({
    mimegans_imp <- mimegans(samp, m = 20, epochs = 10000,
                             params = list(batch_size = 500, 
                                           n_g_layers = 5, n_d_layers = 3),
                             data_info = info,
                             device = "cpu")
  })
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
  
  save(mimegans_imp, tm, file = file.path("simulations", nm, "mimegans",
                                          paste0(digit, ".RData")))
}

for (i in 1:500){
  digit <- stringr::str_pad(i, 4, pad = 0)
  cat("Current:", digit, "\n")
  load(paste0("./data/Complete/", digit, ".RData"))
  samp_srs <- read.csv(paste0("./data/Sample/SRS/", digit, ".csv"))
  samp_balance <- read.csv(paste0("./data/Sample/Balance/", digit, ".csv"))
  samp_neyman <- read.csv(paste0("./data/Sample/Neyman/", digit, ".csv"))
  
  samp_srs$W <- 20
  samp_srs <- match_types(samp_srs, data) %>% 
    mutate(across(all_of(data_info_srs$cat_vars), as.factor, .names = "{.col}"),
           across(all_of(data_info_srs$num_vars), as.numeric, .names = "{.col}"))
  samp_balance <- match_types(samp_balance, data) %>% 
    mutate(across(all_of(data_info_balance$cat_vars), as.factor, .names = "{.col}"),
           across(all_of(data_info_balance$num_vars), as.numeric, .names = "{.col}"))
  samp_neyman <- match_types(samp_neyman, data) %>% 
    mutate(across(all_of(data_info_neyman$cat_vars), as.factor, .names = "{.col}"),
           across(all_of(data_info_neyman$num_vars), as.numeric, .names = "{.col}"))
  
  cox.true <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                      rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) + 
                      SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE, data = data)
  
  if (!file.exists(paste0("./simulations/SRS/mimegans/", digit, ".RData"))){
    do_mimegans(samp_srs, data_info_srs, "SRS", digit)
  }
  if (!file.exists(paste0("./simulations/Balance/mimegans/", digit, ".RData"))){
    do_mimegans(samp_balance, data_info_balance, "Balance", digit)
  }
  if (!file.exists(paste0("./simulations/Neyman/mimegans/", digit, ".RData"))){
    do_mimegans(samp_neyman, data_info_neyman, "Neyman", digit)
  }
}

