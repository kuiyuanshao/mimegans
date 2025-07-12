lapply(c("survival", "dplyr", "stringr"), require, character.only = T)
source("00_utils_functions.R")

replicate <- 500
sampling_designs <- c("SRS", "Balance", "Neyman")
methods <- c("megans", "gain", "mice", "mixgb", "raking")

result_mat <- matrix(0, nrow = (replicate * 4 * 3 + replicate * 2) * 2, ncol = 17)
m <- 1
for (i in 1:replicate){
  digit <- stringr::str_pad(i, 4, pad = 0)
  cat("Current:", digit, "\n")
  load(paste0("./data/Complete/", digit, ".RData"))
  for (j in sampling_designs){
    for (k in methods){
      if (!file.exists(paste0("./simulations/", j, "/", k, "/", digit, ".RData"))){
        next
      }
      load(paste0("./simulations/", j, "/", k, "/", digit, ".RData"))
      
      cox.true <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                          rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                          RACE + I(BMI / 5) + SMOKE, data = data)
      
      
      if (k != "raking"){
        if (k == "megans"){
          megans_imp$imputation <- lapply(megans_imp$imputation, function(dat){
            match_types(dat, data)
          })
          imp.mids <- as.mids(megans_imp$imputation)
        }else if (k == "gain"){
          gain_imp$imputation <- lapply(gain_imp$imputation, function(dat){
            match_types(dat, data)
          })
          imp.mids <- as.mids(gain_imp$imputation)
        }else if (k == "mice"){
          mice_imp <- lapply(complete(mice_imp), function(dat){
            match_types(dat, data)
          })
          imp.mids <- as.mids(mice_imp)
        }else if (k == "mixgb"){
          mixgb_imp <- lapply(mixgb_imp, function(dat){
            match_types(dat, data)
          })
          imp.mids <- as.mids(mixgb_imp)
        }
        fit <- with(data = imp.mids, 
                    exp = coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                                  rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                                  RACE + I(BMI / 5) + SMOKE))
        pooled <- mice::pool(fit)
        sumry <- summary(pooled, conf.int = TRUE)
        result_mat[m, ] <- c(exp(sumry$estimate) - exp(coef(cox.true)), j, k, digit, "Coeff")
        m <- m + 1
        result_mat[m, ] <- c(sumry$std.error, j, k, digit, "StdError")
        m <- m + 1
        result_mat[m, ] <- c(sqrt(diag(vcov(cox.true))), "True", "true", digit, "StdError")
        m <- m + 1
      }else{
        sumry <- summary(rakingest)
      }
    }
  }
}


result_mat <- as.data.frame(result_mat)
names(result_mat) <- c(names(coef(cox.true)), "Design", "Method", "ID", "Type")
result_mat[, 1:length(coef(cox.true))] <- as.matrix(bind_cols(lapply(1:length(coef(cox.true)), 
                                                                     function(i) as.numeric(result_mat[, i]))))

ggplot(result_mat %>% filter(Type == "Coeff", Method == "megans")) + 
  geom_boxplot(aes(x = `rs45065651`))

result_mat %>% filter(Type == "Coeff", Method == "megans")
