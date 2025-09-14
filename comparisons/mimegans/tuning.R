lapply(c("dplyr", "stringr", "torch", "survival", "mclust"), require, character.only = T)
files <- list.files("./mimegans", full.names = TRUE, recursive = FALSE)
files <- files[!grepl("tests", files)]
lapply(files, source)
source("00_utils_functions.R")

grid <- tidyr::expand_grid(pac = c(2, 5, 8, 10), 
                           lr_g = c(1e-4, 2e-4, 3e-4), lr_d = c(1e-4, 2e-4, 3e-4),
                           n_g_layers = 3:5, n_d_layers = 2:4,
                           beta = c(0, 1, 5, 10)) %>%
  filter(n_d_layers <= n_g_layers,
         lr_g       <= lr_d)

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

grid <- grid[first_rep:last_rep, ]

digit <- stringr::str_pad(1, 4, pad = 0)
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

result_srs <- mimegans.cv(samp_srs, fold = 5, data_info_srs, grid, seed = 1)
save(result_srs, file = paste0("./data/miemgans_srs_", first_rep, "-", last_rep, ".RData"))
result_balance <- mimegans.cv(samp_balance, fold = 5, data_info_balance, grid, seed = 1)
save(result_balance, file = paste0("./data/miemgans_balance_", first_rep, "-", last_rep, ".RData"))
result_neyman <- mimegans.cv(samp_neyman, fold = 5, data_info_neyman, grid, seed = 1)
save(result_neyman, file = paste0("./data/miemgans_neyman_", first_rep, "-", last_rep, ".RData"))
