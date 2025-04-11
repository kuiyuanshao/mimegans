pacman::p_load("survey", "readxl", "stringr", "dplyr", "purrr", "ggplot2", "survival")

reCalc <- function(data){
  # Bound FirstOImonth and FirstARTmonth between 0 and 101
  data <- data %>% mutate(FirstOImonth = round(ifelse(FirstOImonth > 101, 101, FirstOImonth)),
                          FirstARTmonth = round(ifelse(FirstARTmonth > 101, 101, FirstARTmonth)),
                          # Bound AGE_AT_LAST_VISIT between 0 and 100 (adjusted for month-to-year conversion)
                          AGE_AT_LAST_VISIT = ifelse(
                            (AGE_AT_LAST_VISIT / (30.437 / 365.25)) >= 100, 
                            100 * (30.437 / 365.25), AGE_AT_LAST_VISIT
                          ),
                          # Bound last.age between 0 and 101
                          last.age = ifelse(
                            (last.age / (30.437 / 365.25)) >= 101, 
                            101 * (30.437 / 365.25), last.age
                          ),
                          # Bound OIage between 0 and 101
                          OIage = ifelse(
                            (OIage / (30.437 / 365.25)) > 101, 
                            101 * (30.437 / 365.25), OIage
                          ),
                          # Bound ARTage between 0 and 101
                          ARTage = ifelse(
                            (ARTage / (30.437 / 365.25)) > 101, 
                            101 * (30.437 / 365.25), ARTage
                          ),
                          ade = pmin(pmax(round(ade), 0), 1),
                          fu = ifelse(fu < 0, 0, fu))
  return (data)
}

exclude <- function(data, 
                    FirstARTmonth = "FirstARTmonth", 
                    OIage = "OIage", ARTage = "ARTage", 
                    fu = "fu", AGE_AT_LAST_VISIT = "AGE_AT_LAST_VISIT"){
  data <- as.data.frame(data)
  data$exclude.no.art <- ifelse(data[[FirstARTmonth]] >= 101, 1, ifelse(data[[ARTage]] > data[[AGE_AT_LAST_VISIT]], 1, 0))
  data$exclude.prior.ade <- ifelse(data[[OIage]] != (101 * 30.437/365.25) & data[[FirstARTmonth]] != 101 & data[[OIage]] < data[[ARTage]], 1, 0)
  data$exclude.not.naive <- ifelse(data[[FirstARTmonth]] != 101 & data[[ARTage]] < 0, 1, 0)
  data$exclude <- with(data, ifelse(exclude.no.art==1 | exclude.prior.ade==1 | exclude.not.naive, 1, 0))
  
  data <- data[data$exclude == 0, ]
  data <- data[data[[fu]] > 0, ]
  
  return (data)
}


find_coef_var <- function(imp){
  m_coefs <- NULL
  m_vars <- NULL
  inclusion <- NULL
  for (m in 1:length(imp)){
    ith_imp <- imp[[m]]
    replicate <- reCalc(ith_imp)
    
    st1 <- exclude(replicate, FirstARTmonth = "FirstARTmonth", 
                   OIage = "OIage", ARTage = "ARTage", 
                   fu = "fu", AGE_AT_LAST_VISIT = "AGE_AT_LAST_VISIT")
    
    imp_mod.1 <- coxph(Surv(fu, ade) ~ X, data = st1, y = FALSE)
    m_coefs <- rbind(m_coefs, imp_mod.1$coef['X'])
    m_vars <- rbind(m_vars, diag(vcov(imp_mod.1))['X'])
    
    inclusion <- rbind(inclusion, nrow(st1))
  }

  var <- 1/length(imp) * colSums(m_vars) + (length(imp) + 1) * var(m_coefs[, 1]) / length(imp)
  return (list(coef = colMeans(m_coefs), var = var,
               inclusion = colMeans(inclusion)))
}

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

source("./megans/mmer.impute.cwgangp.R")
source("./megans/cwgangp.nets.R")
source("./megans/normalizing.R")
source("./megans/encoding.R")
source("./megans/sample.batches.R")
source("./megans/loss.funs.R")
source("./megans/generate.impute.R")
megans_imp <- mmer.impute.cwgangp(data_surv, m = 5, num.normalizing = "", cat.encoding = "onehot", 
                                  device = "cpu", epochs = 6000, 
                                  params = list(alpha = 0, beta = 0, n_g_layers = 3, n_d_layers = 2), 
                                  data_info = data_info, save.step = 1000)
save(megans_imp, file = "Surv_NOMSE.RData")
find_coef_var(megans_imp$imputation)

imp_mod.1 <- coxph(Surv(fu, ade) ~ X, data = exclude(reCalc(megans_imp$imputation[[5]])), y = FALSE)
imp_mod.1



ggplot() + 
  geom_density(aes(x = fu), data = exclude(reCalc(megans_imp$imputation[[1]])), colour = "red") + 
  geom_density(aes(x = fu), data = exclude(reCalc(data_surv)), colour = "blue")
