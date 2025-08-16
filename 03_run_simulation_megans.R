lapply(c("dplyr", "stringr", "torch", "survival"), require, character.only = T)
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
sampling_design <- ifelse(length(args) >= 2, 
                          args[2], Sys.getenv("SAMP", "All"))
replicate <- 500
n_chunks <- 20
chunk_size <- ceiling(replicate / n_chunks)
first_rep <- (task_id - 1) * chunk_size + 1
last_rep <- min(task_id * chunk_size, replicate)


do_megans <- function(samp, info, nm, digit) {
  tm <- system.time({
    megans_imp <- mmer.impute.cwgangp(samp, m = 20, 
                                      params = list(n_g_layers = 5, n_d_layers = 3, type_d = "mlp", lambda = 0),
                                      data_info = info)
  })
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
  cat("Current: ", nm, "\n")
  
  cat(sprintf("[system.time] user=%.3fs sys=%.3fs elapsed=%.3fs\n",
              tm[["user.self"]], tm[["sys.self"]], tm[["elapsed"]]))
  cat("Bias: \n")
  cat(exp(sumry$estimate) - exp(coef(cox.true)))
  cat("Variance: \n")
  cat(apply(bind_rows(lapply(fit$analyses, function(i){coef(i)})), 2, var))
  
  save(megans_imp, tm, file = file.path("simulations", nm, "megans",
                                        paste0(digit, ".RData")))
}


for (i in 1:10){ 
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
  
  cox.true <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                      rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                      RACE + I(BMI / 5) + SMOKE, data = data)
  
  if (!file.exists(paste0("./simulations/SRS/megans/", digit, ".RData"))){
    do_megans(samp_srs, data_info_srs, "SRS", digit)
  }
  if (!file.exists(paste0("./simulations/Balance/megans/", digit, ".RData"))){
    do_megans(samp_balance, data_info_balance, "Balance", digit)
  }
  if (!file.exists(paste0("./simulations/Neyman/megans/", digit, ".RData"))){
    do_megans(samp_neyman, data_info_neyman, "Neyman", digit)
  }
}
