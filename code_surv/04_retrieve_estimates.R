lapply(c("survival", "dplyr", "stringr", "survey", "mice", "readxl"), require, character.only = T)
source("00_utils_functions.R")

options(survey.lonely.psu = "certainty")
retrieveEst <- function(method){
  resultCoeff <- resultStdError <- resultCI <- NULL
  sampling_designs <- c("SRS", "Balance", "Neyman")
  for (i in 1:500){
    digit <- stringr::str_pad(i, 4, pad = 0)
    cat("Current:", digit, "\n")
    load(paste0("./data/Complete/", digit, ".RData"))
    if (method == "true_me"){
      cox.mod <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) +
                          rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                          SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE, data = data)
      cox.me <- coxph(Surv(T_I_STAR, EVENT_STAR) ~ I((HbA1c_STAR - 50) / 5) +
                        rs4506565_STAR + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                        SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE_STAR, data = data)
      resultCoeff <- rbind(resultCoeff, c(exp(coef(cox.me)), "ME", "ME", digit))
      resultStdError<- rbind(resultStdError, c(sqrt(diag(vcov(cox.me))), "ME", "ME", digit))
      resultCI <- rbind(resultCI, c(exp(confint(cox.me)[, 1]), exp(confint(cox.me)[, 2]), "ME", "ME", digit))
      
      resultCoeff <- rbind(resultCoeff, c(exp(coef(cox.mod)), "TRUE", "TRUE", digit))
      resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(cox.mod))), "TRUE", "TRUE", digit))
      resultCI <- rbind(resultCI, c(exp(confint(cox.mod)[, 1]), exp(confint(cox.mod)[, 2]), "TRUE", "TRUE", digit))
    }else{
      for (j in sampling_designs){
        if (method == "Sample"){
          samp <- read.csv(paste0("./data/Sample/", j, "/", digit, ".csv"))
          samp <- match_types(samp, data)
          if (j %in% c("Balance", "Neyman")){
            design <- svydesign(ids = ~1, strata = ~STRATA, weights = ~W, 
                                data = samp)
            cox.mod <- svycoxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) +
                                   rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                                   SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE, design)
          }else{
            cox.mod <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) +
                                rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                                SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE, data = samp)
          }
          resultCoeff <- rbind(resultCoeff, c(exp(coef(cox.mod)), toupper(j), "COMPLETE", digit))
          resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(cox.mod))), toupper(j), "COMPLETE", digit))
          resultCI <- rbind(resultCI, c(exp(confint(cox.mod)[, 1]), exp(confint(cox.mod)[, 2]), toupper(j), "COMPLETE", digit))
        }else if (method == "raking"){
          load(paste0("./simulations/", j, "/", method, "/", digit, ".RData"))
          cox.mod <- rakingest
          resultCoeff <- rbind(resultCoeff, c(exp(coef(cox.mod)), toupper(j), toupper(method), digit))
          resultStdError <- rbind(resultStdError, c(sqrt(diag(vcov(cox.mod))), toupper(j), toupper(method), digit))
          resultCI <- rbind(resultCI, c(exp(confint(cox.mod))[, 1], exp(confint(cox.mod))[, 2], toupper(j), toupper(method), digit))
        }else{
          if (method != "tpvmi_rddm"){
            temp_env <- new.env()
            load(paste0("./simulations/", j, "/", method, "/", digit, ".RData"), envir = temp_env)
            multi_impset <- temp_env[[ls(temp_env)[1]]]
          }else{
            multi_impset <- lapply(excel_sheets(paste0("./simulations/", j, "/", method, "/", digit, ".xlsx")), function(x) {
              read_excel(path = paste0("./simulations/", j, "/", method, "/", digit, ".xlsx"), sheet = x)
            })
          }
          
          if (method == "tpvmi_gans"){
            multi_impset$imputation <- lapply(multi_impset$imputation, function(dat){
              match_types(dat, data)
            })
            imp.mids <- as.mids(multi_impset$imputation)
          }else if (method == "tpvmi_rddm"){
            multi_impset <- lapply(multi_impset, function(dat){
              match_types(dat, data)
            })
            imp.mids <- as.mids(multi_impset)
          }else if (method == "mice"){
            multi_impset <- mice::complete(multi_impset, "all")
            multi_impset <- lapply(multi_impset, function(dat){
              match_types(dat, data)
            })
            imp.mids <- as.mids(multi_impset)
          }else if (method == "mixgb"){
            multi_impset <- lapply(multi_impset, function(dat){
              match_types(dat, data)
            })
            imp.mids <- as.mids(multi_impset)
          }
          cox.mod <- with(data = imp.mids, 
                          exp = coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) +
                                        rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                                        SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE))
          pooled <- mice::pool(cox.mod)
          sumry <- summary(pooled, conf.int = TRUE)
          resultCoeff <- rbind(resultCoeff, c(exp(sumry$estimate), toupper(j), toupper(method), digit))
          resultStdError <- rbind(resultStdError, c(sumry$std.error, toupper(j), toupper(method), digit))
          resultCI <- rbind(resultCI, c(exp(sumry$`2.5 %`), exp(sumry$`97.5 %`), toupper(j), toupper(method), digit))
        }
      }
    }
  }
  resultCoeff <- as.data.frame(resultCoeff)
  names(resultCoeff) <- c(names(coef(cox.mod)), "Design", "Method", "ID")
  resultStdError <- as.data.frame(resultStdError)
  names(resultStdError) <- c(names(coef(cox.mod)), "Design", "Method", "ID")
  resultCI <- as.data.frame(resultCI)
  names(resultCI) <- c(paste0(names(coef(cox.mod)), ".lower"), 
                       paste0(names(coef(cox.mod)), ".upper"),
                       "Design", "Method", "ID")
  save(resultCoeff, resultStdError, resultCI, 
       file = paste0("./simulations/results_", method,".RData"))
}

for (method in c("true_me", "complete", "mice", "mixgb", "tpvmi_gans", "tpvmi_rddm")){
  retrieveEst(method)
}

i <- 1
digit <- stringr::str_pad(i, 4, pad = 0)
cat("Current:", digit, "\n")
load(paste0("./data/Complete/", digit, ".RData"))
cox.fit <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) +
                   rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                   SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE, data = data)
j = "SRS"
multi_impset <- lapply(excel_sheets(paste0("./simulations/", j, "/", method, "/", digit, ".xlsx")), function(x) {
  read_excel(path = paste0("./simulations/", j, "/", method, "/", digit, ".xlsx"), sheet = x)
})
multi_impset <- lapply(multi_impset, function(dat){
  match_types(dat, data)
})
imp.mids <- as.mids(multi_impset)
cox.mod <- with(data = imp.mids, 
                exp = coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) +
                              rs4506565 + I((AGE - 50) / 5) + I((eGFR - 90) / 10) +
                              SEX + INSURANCE + RACE + I(BMI / 5) + SMOKE))
pooled <- mice::pool(cox.mod)
sumry <- summary(pooled, conf.int = TRUE)
exp(coef(cox.fit)) - exp(sumry$estimate)

table(multi_impset[[1]]$EVENT, data$EVENT)
table(multi_impset[[1]]$SMOKE, data$SMOKE)
plot(x = data$HbA1c, y = multi_impset[[1]]$HbA1c)
lines(x = 1:200, y = 1:200, col = "red")
plot(x = data$T_I, y = multi_impset[[1]]$T_I)


ggplot(data) + 
  geom_density(aes(x = T_I, colour = SMOKE))

ggplot(multi_impset[[1]]) + 
  geom_density(aes(x = T_I, colour = SMOKE))
