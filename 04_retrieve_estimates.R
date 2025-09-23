lapply(c("survival", "dplyr", "stringr", "survey", "mice"), require, character.only = T)
source("00_utils_functions.R")
options(survey.lonely.psu = "certainty")

sampling_designs <- c("SRS", "Balance", "Neyman")
methods <- c("mimegans", "mice", "mixgb", "raking")

# 5 * 3 + 3 + 1 + 1
resultCoeff <- matrix(0, nrow = 17 * 500, ncol = 17)
resultStdError <- matrix(0, nrow = 17 * 500, ncol = 17)
resultCI <- matrix(0, nrow = 17 * 500, ncol = 31)
## available <- as.integer(sub("\\.[Rr]Data$", "", list.files("./simulations/Neyman/mimegans")))

ind <- 1
for (i in 1:500){
  digit <- stringr::str_pad(i, 4, pad = 0)
  cat("Current:", digit, "\n")
  load(paste0("./data/Complete/", digit, ".RData"))
  cox.true <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) +
                      rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                      SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE, data = data)
  cox.me <- coxph(Surv(T_I_STAR, EVENT_STAR) ~ I((HbA1c_STAR - 50) / 5) +
                    rs4506565_STAR + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                    SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE_STAR, data = data)
  resultCoeff[ind, ] <- c(exp(coef(cox.me)), "ME", "ME", digit)
  resultStdError[ind, ]<- c(sqrt(diag(vcov(cox.me))), "ME", "ME", digit)
  resultCI[ind, ] <- c(exp(confint(cox.me)[, 1]), exp(confint(cox.me)[, 2]), "ME", "ME", digit)
  ind <- ind + 1
  resultCoeff[ind, ] <- c(exp(coef(cox.true)), "TRUE", "TRUE", digit)
  resultStdError[ind, ] <- c(sqrt(diag(vcov(cox.true))), "TRUE", "TRUE", digit)
  resultCI[ind, ] <- c(exp(confint(cox.true)[, 1]), exp(confint(cox.true)[, 2]), "TRUE", "TRUE", digit)
  ind <- ind + 1
  for (j in sampling_designs){
    samp <- read.csv(paste0("./data/Sample/", j, "/", digit, ".csv"))
    samp <- match_types(samp, data)
    if (j %in% c("Balance", "Neyman")){
      design <- svydesign(ids = ~1, strata = ~STRATA, weights = ~W, 
                          data = samp)
      cox.comp <- svycoxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) +
                             rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                             SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE, design)
    }else{
      cox.comp <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) +
                          rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                          SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE, data = samp)
    }
    resultCoeff[ind, ] <- c(exp(coef(cox.comp)), toupper(j), "COMPLETE", digit)
    resultStdError[ind, ] <- c(sqrt(diag(vcov(cox.comp))), toupper(j), "COMPLETE", digit)
    resultCI[ind, ] <- c(exp(confint(cox.comp)[, 1]), exp(confint(cox.comp)[, 2]), toupper(j), "COMPLETE", digit)
    ind <- ind + 1
    for (k in methods){
      if (!file.exists(paste0("./simulations/", j, "/", k, "/", digit, ".RData"))){
        next
      }
      load(paste0("./simulations/", j, "/", k, "/", digit, ".RData"))
      if (k != "raking"){
        if (k == "mimegans"){
          mimegans_imp$imputation <- lapply(mimegans_imp$imputation, function(dat){
            match_types(dat, data)
          })
          imp.mids <- as.mids(mimegans_imp$imputation)
        }else if (k == "gain"){
          gain_imp$imputation <- lapply(gain_imp$imputation, function(dat){
            match_types(dat, data)
          })
          imp.mids <- as.mids(gain_imp$imputation)
        }else if (k == "mice"){
          mice_imp <- mice::complete(mice_imp, "all")
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
                                  rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                                  SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE))
        pooled <- mice::pool(cox.fit)
        sumry <- summary(pooled, conf.int = TRUE)
        resultCoeff[ind, ] <- c(exp(sumry$estimate), toupper(j), toupper(k), digit)
        resultStdError[ind, ] <- c(sumry$std.error, toupper(j), toupper(k), digit)
        resultCI[ind, ] <- c(exp(sumry$`2.5 %`), exp(sumry$`97.5 %`), toupper(j), toupper(k), digit)
      }else{
        resultCoeff[ind, ] <- c(exp(coef(rakingest)), toupper(j), toupper(k), digit)
        resultStdError[ind, ] <- c(sqrt(diag(vcov(rakingest))), toupper(j), toupper(k), digit)
        resultCI[ind, ] <- c(exp(confint(rakingest))[, 1], exp(confint(rakingest))[, 2], toupper(j), toupper(k), digit)
      }
      ind <- ind + 1
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


