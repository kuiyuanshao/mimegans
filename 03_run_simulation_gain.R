lapply(c("dplyr", "stringr", "survival"), require, character.only = T)
lapply(paste0("./comparisons/gain/", list.files("./comparisons/gain/")), source)
source("00_utils_functions.R")

if(!dir.exists('./simulations')){dir.create('./simulations')}
if(!dir.exists('./simulations/SRS')){dir.create('./simulations/SRS')}
if(!dir.exists('./simulations/Balance')){dir.create('./simulations/Balance')}
if(!dir.exists('./simulations/Neyman')){dir.create('./simulations/Neyman')}

if(!dir.exists('./simulations/SRS/gain')){dir.create('./simulations/SRS/gain')}
if(!dir.exists('./simulations/Balance/gain')){dir.create('./simulations/Balance/gain')}
if(!dir.exists('./simulations/Neyman/gain')){dir.create('./simulations/Neyman/gain')}

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


do_gain <- function(samp, info, nm, digit) {
  tm <- system.time({
    gain_imp <- gain(samp_srs, m = 20, data_info_srs, device = "cpu", batch_size = 128, hint_rate = 0.9, 
                     alpha = 100, beta = 10, n = 10000)
  })
  
  save(gain_imp, tm, file = file.path("simulations", nm, "gain",
                                      paste0(digit, ".RData")))
}

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
  
  # GAIN:
  if (!file.exists(paste0("./simulations/SRS/gain/", digit, ".RData"))){
    do_gain(samp_srs, data_info_srs, "SRS", digit)
  }
  if (!file.exists(paste0("./simulations/Balance/gain/", digit, ".RData"))){
    do_gain(samp_balance, data_info_balance, "Balance", digit)
  }
  if (!file.exists(paste0("./simulations/Neyman/gain/", digit, ".RData"))){
    do_gain(samp_neyman, data_info_neyman, "Neyman", digit)
  }
}


