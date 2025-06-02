#################################### Generate Sample ####################################
lapply(c("BalancedSampling", "dplyr"), require, character.only = T)
generateSample <- function(data, proportion, seed, design = "balanced"){
  n <- round(proportion * nrow(data))
  strat_srs <- function(ids, strata, frac){
    split(ids, strata) |>
      lapply(function(ix) sample(ix, ceiling(frac * length(ix)))) |>
      unlist()
  }
  cases <- data$EVENT_STAR == 1
  cohort <- data$EVENT_STAR == 0
  time_cut <- cut(data$T_I_STAR, breaks = c(-Inf, 12, 18, 23.999, Inf), 
                  labels = 1:4)
  hba1c_cut <- cut(data$HbA1c_STAR, breaks = c(-Inf, 64, 75, Inf), 
                   labels = 1:3)
  stratum <- interaction(time_cut, hba1c_cut, data$EVENT_STAR)
  subcohort <- strat_srs(1:nrow(data), stratum, 0.05)
  
  phase2_cases <- union(cases, subcohort)
  phase2_cohort <- union(cohort, subcohort)
  # 2.
  
  cases <- data$EVENT_STAR == 1
  paste0(timecut[cases], "_", 1)
}


load("./data/TRUE/0001.RData")
