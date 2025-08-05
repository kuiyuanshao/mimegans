lapply(c("survival", "dplyr", "stringr", "survey", "mice"), require, character.only = T)
source("00_utils_functions.R")
options(survey.lonely.psu = "certainty")

replicate <- 21
sampling_designs <- "Balance" # c("SRS", "Balance", "Neyman")
methods <- c("megans", "mice") # c("megans", "gain", "mice", "mixgb", "raking")

resultCoeff <- NULL
resultStdError <- NULL
resultCI <- NULL
for (i in 1:replicate){
  digit <- stringr::str_pad(i, 4, pad = 0)
  cat("Current:", digit, "\n")
  load(paste0("./data/Complete/", digit, ".RData"))
  cox.true <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                      rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                      RACE + I(BMI / 5) + SMOKE, data = data)
  cox.me <- coxph(Surv(T_I_STAR, EVENT_STAR) ~ I((HbA1c_STAR - 50) / 5) + 
                    rs4506565_STAR + I((AGE - 50) / 5) + SEX + INSURANCE + 
                    RACE + I(BMI / 5) + SMOKE_STAR, data = data)
  resultCoeff <- rbind(resultCoeff, c(exp(coef(cox.me)) - exp(coef(cox.true)), "ME", "me", digit))
  resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(cox.me))), "ME", "me", digit))
  resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(cox.true))), "True", "true", digit))
  for (j in sampling_designs){
    samp <- read.csv(paste0("./data/Sample/", j, "/", digit, ".csv"))
    samp <- match_types(samp, data)
    if (j %in% c("Balance", "Neyman")){
      # samp <- reallocate(samp)
      design <- svydesign(ids = ~1, strata = ~STRATA, weights = ~W, 
                          data = samp)
      cox.comp <- svycoxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                             rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                             RACE + I(BMI / 5) + SMOKE, design)
    }else{
      cox.comp <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                          rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                          RACE + I(BMI / 5) + SMOKE, data = samp)
    }
    resultCoeff <- rbind(resultCoeff, c(exp(coef(cox.comp)) - exp(coef(cox.true)), j, "complete", digit))
    resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(cox.comp))), "COMPLETE", "complete", digit))
    for (k in methods){
      if (!file.exists(paste0("./simulations/", j, "/", k, "/", digit, ".RData"))){
        next
      }
      imp <- load(paste0("./simulations/", j, "/", k, "/", digit, ".RData"))
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
          if (imp != "mice_imp"){
            mice_imp <- fit
          }
          if (class(mice_imp) != "list"){
            mice_imp <- mice::complete(mice_imp, "all")
          }
          mice_imp <- lapply(mice_imp, function(dat){
            match_types(dat, data)
          })
          imp.mids <- as.mids(mice_imp)
        }else if (k == "mixgb"){
          mixgb_imp <- lapply(mixgb_imp, function(dat){
            match_types(dat, data)
          })
          imp.mids <- as.mids(mixgb_imp)
        }
        cox.fit <- with(data = imp.mids, 
                    exp = coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                                  rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                                  RACE + I(BMI / 5) + SMOKE))
        pooled <- mice::pool(cox.fit)
        sumry <- summary(pooled, conf.int = TRUE)
        resultCoeff <- rbind(resultCoeff, c(exp(sumry$estimate) - exp(coef(cox.true)), j, k, digit))
        resultStdError <- rbind(resultStdError, c(sumry$std.error, j, k, digit))
        resultCI <- rbind(resultCI, c(exp(sumry$`2.5 %`), exp(sumry$`97.5 %`), j, k, digit))
      }else{
        sumry <- summary(rakingest)
      }
    }
  }
}

resultCoeff <- as.data.frame(resultCoeff)
names(resultCoeff) <- c(names(coef(cox.true)), "Design", "Method", "ID")
resultStdError <- as.data.frame(resultStdError)
names(resultStdError) <- c(names(coef(cox.true)), "Design", "Method", "ID")
resultCI <- as.data.frame(resultCI)
names(resultCI) <- c(paste0(names(coef(cox.true)), ".lower"), 
                     paste0(names(coef(cox.true)), ".upper"),
                     "Design", "Method", "ID")

save(resultCoeff, resultStdError, resultCI, file = "./simulations/results.RData")


ggplot(megans_imp$imputation[[1]]) + 
  geom_density(aes(x = T_I), colour = "red") +
  geom_density(aes(x = T_I), data = data)


ggplot(megans_imp$imputation[[1]]) + 
  geom_density(aes(x = HbA1c), colour = "red") +
  geom_density(aes(x = HbA1c), data = data)


apply(bind_cols(lapply((resultCoeff %>% filter(Method == "megans"))[, 1:13], as.numeric)) + matrix(rep(exp(coef(cox.true)), 17), byrow = T,  ncol = 13), 2, var)
apply(bind_cols(lapply((resultStdError %>% filter(Method == "megans"))[, 1:13], as.numeric)) ^ 2, 2, mean)

apply(bind_cols(lapply((resultCoeff %>% filter(Method == "megans"))[, 1:13], as.numeric)) + 
        matrix(rep(exp(coef(cox.true)), 17), byrow = T,  ncol = 13), 2, var) 

coeffs <- NULL
for (i in 1:20){
  digit <- stringr::str_pad(i, 4, pad = 0)
  load(paste0("./data/Sample/Debug/", digit, ".RData"))
  
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
}

apply(coeffs, 2, var) / apply(vars, 2, var) 
vars <- bind_rows(lapply(fit$analyses, function(i){exp(coef(i))}))
apply(vars, 2, var)