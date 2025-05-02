pacman::p_load("survey", "readxl", "stringr", "dplyr", "purrr", "ggplot2")

find_coef_var <- function(imp){
  m_coefs.1 <- NULL
  m_coefs.2 <- NULL
  m_vars.1 <- NULL
  m_vars.2 <- NULL
  for (m in 1:length(imp)){
    ith_imp <- imp[[m]]
    
    imp_mod.1 <- glm(hypertension ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr, ith_imp, family = binomial())
    imp_mod.2 <- glm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn + female + bkg_o + bkg_pr, ith_imp, family = gaussian())
    
    m_coefs.1 <- rbind(m_coefs.1, coef(imp_mod.1))
    m_coefs.2 <- rbind(m_coefs.2, coef(imp_mod.2))
    m_vars.1 <- rbind(m_vars.1, diag(vcov(imp_mod.1)))
    m_vars.2 <- rbind(m_vars.2, diag(vcov(imp_mod.2)))
  }
  var_between.1 <- apply(m_coefs.1, 2, var)
  var_between.2 <- apply(m_coefs.2, 2, var)
  var.1 <- 1/length(imp) * colSums(m_vars.1) + (length(imp) + 1) / length(imp) * var_between.1
  var.2 <- 1/length(imp) * colSums(m_vars.2) + (length(imp) + 1) / length(imp) * var_between.2
  
  lambda.1 <- (var_between.1 + var_between.1 / length(imp)) / var.1
  lambda.2 <- (var_between.2 + var_between.2 / length(imp)) / var.2
  df_old.1 <- (length(imp) - 1) / (lambda.1 ^ 2)
  df_old.2 <- (length(imp) - 1) / (lambda.2 ^ 2)
  df_obs.1 <- (imp_mod.1$df.residual + 1) / (imp_mod.1$df.residual + 3) * imp_mod.1$df.residual * (1 - lambda.1)
  df_obs.2 <- (imp_mod.2$df.residual + 1) / (imp_mod.2$df.residual + 3) * imp_mod.2$df.residual * (1 - lambda.2)
  
  df_adj.1 <- (df_old.1 * df_obs.1) / (df_old.1 + df_obs.1)
  df_adj.2 <- (df_old.2 * df_obs.2) / (df_old.2 + df_obs.2)
  
  return (list(coef = list(colMeans(m_coefs.1), colMeans(m_coefs.2)), var = list(var.1, var.2),
               df_adj = list(df_adj.1, df_adj.2)))
}


CI_coverage <- function(output, true_coef.1, true_coef.2){
  lower.1 <- output$coef[[1]] - qt(0.975, output$df_adj[[1]]) * sqrt(output$var[[1]])
  upper.1 <- output$coef[[1]] + qt(0.975, output$df_adj[[1]]) * sqrt(output$var[[1]])
  
  lower.2 <- output$coef[[2]] - qt(0.975, output$df_adj[[2]]) * sqrt(output$var[[2]])
  upper.2 <- output$coef[[2]] + qt(0.975, output$df_adj[[2]]) * sqrt(output$var[[2]])
  
  logistic <- (true_coef.1 >= lower.1) & (true_coef.1 <= upper.1)
  linear <- (true_coef.2 >= lower.2) & (true_coef.2 <= upper.2)
  return (list(logistic, linear))
}

# loading GAIN functions
source("./comparing-model-based/gain/gain.R")
source("./comparing-model-based/gain/utils.R")
imputation <- gain(data, device = "cpu", batch_size = 128, hint_rate = 0.9, 
                   alpha = 10, beta = 1, n = 10000)
lm(Y ~ X + Z, data = imputation[[1]])
ggplot(imputation[[1]]) + geom_density(aes(x = X))

# loading megans functions
source("./megans/mmer.impute.cwgangp.R")
source("./megans/cwgangp.nets.R")
source("./megans/normalizing.R")
source("./megans/encoding.R")
source("./megans/sample.batches.R")
source("./megans/loss.funs.R")
source("./megans/generate.impute.R")

generateLinearData <- function(digit){
  beta <- c(1, 1, 1)
  e_U <- c(sqrt(3), sqrt(3))
  mx <- 0; sx <- 1; zrange <- 1; zprob <- .5
  simZ   <- rbinom(10000, zrange, zprob)
  simX   <- (1-simZ)*rnorm(10000, 0, 1) + simZ*rnorm(10000, 0.5, 1)
  epsilon <- rnorm(10000, 0, 1)
  simY    <- beta[1] + beta[2]*simX + beta[3]*simZ + epsilon
  simX_tilde <- simX + rnorm(10000, 0, e_U[1]*(simZ==1) + e_U[2]*(simZ==2))
  data <- data.frame(X_tilde=simX_tilde, Y=simY, X=simX, Z=simZ)
  data$W <- 1
  data_mis <- data
  data_mis$X[sample(10000, 8000)] <- NA
  return (list(data = data, data_mis = data_mis))
}

d <- generateLinearData(0)
data <- d$data
data_mis <- d$data_mis
data_info <- list(phase1_vars = c("X_tilde"), phase2_vars = "X", weight_var = "W",
                  cat_vars = "Z", num_vars = names(data)[!names(data) %in% c("W", "Z")])

source("./megans/mmer.impute.cwgangp.R")
source("./megans/generate.impute.R")
megans_imp <- mmer.impute.cwgangp(data_mis, m = 5, num.normalizing = "zscore", cat.encoding = "onehot", 
                                   device = "cpu", epochs = 3000, 
                                   params = list(gamma = 1, alpha = 0.25, beta = 0), 
                                   data_info = data_info, save.step = 1000)
find_coef_var(megans_imp$imputation)

summary(megans_imp$gsample[[1]]$X[!is.na(data_mis$X)])
summary(data$X)
lm(Y ~ X_tilde + Z, data = megans_imp$gsample[[3]])

head(minmax$data)
head(zscore$data)

ggplot() + geom_density(aes(x = Y), data = megans_imp$gsample[[1]], colour = "red", alpha = 0.5) + 
  geom_density(aes(x = Y), data = data, alpha = 0.5)

data_surv <- read.csv("/nesi/project/uoa03789/PhD/SamplingDesigns/SurvivalData/SurvivalSample/-1_0_-2_0_-0.25/SRS/SRS_0001.csv")
for (var in c("A.star", "D.star", "C.star", "A", "D", "C", "CFAR_PID", "X.1", "N_h", "X_cut", "fu.star_cut", "Strata", "R")){
  data_surv[[var]] <- NULL
}

vars <- c("lastC", 
          "FirstOImonth", "FirstARTmonth",
          "AGE_AT_LAST_VISIT",
          "ARTage", "OIage", "last.age", "fu")

categorical_cols <- c("ade.star", "ade")
target_variables_1 <- c("lastC.star", 
                        "FirstOImonth.star", "FirstARTmonth.star",
                        "AGE_AT_LAST_VISIT.star", 
                        "ARTage.star", "OIage.star", "last.age.star", 
                        "ade.star", "fu.star")
target_variables_2 <- c("lastC", 
                        "FirstOImonth", "FirstARTmonth",
                        "AGE_AT_LAST_VISIT",
                        "ARTage", "OIage", "last.age", 
                        "ade", "fu")
data_info = list(phase1_vars = target_variables_1, 
                 phase2_vars = target_variables_2, 
                 weight_var = "W",
                 cat_vars = categorical_cols,
                 num_vars = names(data_surv)[!names(data_surv) %in% c("W", categorical_cols)])


data_nut <- read.csv("/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalData/NutritionalSample/SRS/SRS_0001.csv")
data_nut$X <- NULL
data_nut$id <- NULL
data_nut$R <- NULL
data_info = list(phase1_vars = c("c_ln_na_bio1", "c_ln_k_bio1", "c_ln_kcal_bio1", "c_ln_protein_bio1"), 
                 phase2_vars = c("c_ln_na_true", "c_ln_k_true", "c_ln_kcal_true", "c_ln_protein_true"), 
                 weight_var = "W",
                 cat_vars = c("hypertension", "bkg_pr", "bkg_o", "female", "high_chol", "usborn", "idx"),
                 num_vars = names(data_nut)[!names(data_nut) %in% c("W", "hypertension", 
                                                                    "bkg_pr", "bkg_o", "female", 
                                                                    "high_chol", "usborn", "idx")])
source("./megans/mmer.impute.ccwgangp.R")
source("./megans/generate.impute.R")
megans_imp <- mmer.impute.cwgangp(data_surv, m = 5, num.normalizing = "zscore", cat.encoding = "onehot", 
                                  device = "cpu", epochs = 1, 
                                  params = list(gamma = 1, alpha = 0, beta = 0), 
                                  data_info = data_info, save.step = 1000)
find_coef_var(megans_imp$imputation)


replicate <- reCalc(megans_imp$step_result[[4]][[1]])

st1 <- exclude(replicate, FirstARTmonth = "FirstARTmonth", 
               OIage = "OIage", ARTage = "ARTage", 
               fu = "fu", AGE_AT_LAST_VISIT = "AGE_AT_LAST_VISIT")
imp_mod.1 <- coxph(Surv(fu, ade) ~ X, data = st1, y = FALSE)
imp_mod.1



find_coef_var(megans_imp$imputation)
find_coef_var(megans_imp$step_result[[10]])

mode <- normalize.zscore(data, "X") 

coef(lm(X ~ X_tilde + Y + Z, data = megans_imp$gsample[[1]][is.na(data_mis$X), ]))
coef(lm(X ~ X_tilde + Y + Z, data = megans_imp$gsample[[1]][!is.na(data_mis$X), ]))
coef(lm(X ~ X_tilde + Y + Z, data = data[is.na(data_mis$X), ]))

ggplot() + 
  geom_point(aes(x = X, y = Y), data = megans_imp$imputation[[4]], colour = "red", alpha = 0.01) + 
  geom_point(aes(x = X, y = Y), data = data, colour = "blue", alpha = 0.01)

ggplot() + 
  geom_density(aes(x = X), data = mode$data)

sd(mode$data$X)

ggplot(megans_imp$loss) + geom_line(aes(x = 1:dim(megans_imp$loss)[1], y = `G Loss`), colour = "red") +
  geom_line(aes(x = 1:dim(megans_imp$loss)[1], y = `D Loss`), colour = "blue")


c(mean(data$X[data$Z == 1], na.rm = T), mean(data$X[data$Z == 0], na.rm = T))
c(mean(megans_imp$imputation[[1]]$X[data$Z == 1], na.rm = T), mean(megans_imp$imputation[[1]]$X[data$Z == 0], na.rm = T))

c(sd(data$X[data$Z == 1], na.rm = T), sd(data$X[data$Z == 0], na.rm = T))
c(sd(megans_imp$imputation[[1]]$X[data$Z == 1], na.rm = T), sd(megans_imp$imputation[[1]]$X[data$Z == 0], na.rm = T))


