lapply(c("survival", "dplyr", "stringr"), require, character.only = T)
source("00_utils_functions.R")

replicate <- 1000
sampling_designs <- c("SRS", "Balance", "Neyman")
methods <- c("megans", "gain", "mice", "mixgb", "raking")

result_mat <- matrix(0, nrow = replicate * 4 * 3 + replicate * 2,
                     ncol = 1)
for (i in 1:replicate){
  digit <- stringr::str_pad(i, 4, pad = 0)
  cat("Current:", digit, "\n")
  for (j in sampling_designs){
    for (k in methods){
      if (!dir.exists(paste0("./simulations/", j, "/", k, "/", i, ".RData"))){
        next
      }
      load(paste0("./simulations/", j, "/", k, "/", i, ".RData"))
      
      if (k != "raking"){
        if (k == "megans"){
          imp.mids <- as.mids(megans_imp$imputation)
        }else if (k == "gain"){
          imp.mids <- as.mids(gain_imp$imputation)
        }else if (k == "mice"){
          imp.mids <- mice_imp
        }else if (k == "mixgb"){
          imp.mids <- as.mids(mixgb_imp$imputation)
        }
        fit <- with(data = imp.mids, 
                    exp = coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                                  rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                                  RACE + I(BMI / 5) + SMOKE))
        pooled <- mice::pool(fit)
        sumry <- summary(pooled, conf.int = TRUE)
        result_mat[m, ] <- c(exp(sumry$estimate), j, k, digit)
      }else{
        sumry <- summary(rakingest)
      }
      
    }
  }
}