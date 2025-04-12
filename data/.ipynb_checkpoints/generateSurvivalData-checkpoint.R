####### generate Survival Population #######
pacman::p_load('data.table', 'mvtnorm', 'magrittr', 'dplyr', 'MASS')

#Demographics: Variables such as sex, race, ethnicity, and insurance type.
#Diagnostic Codes: ICD codes, which are often stored categorically.
#Medications and Procedures: Often represented as binary indicators (yes/no) or categorical groupings.
#Other Indicators: Variables like smoking status or comorbidity flags, which are usually binary or categorical.

#Categorical Variables: Sex, Hypertension, Diabetes, Race, BMI category, 
####################### Insurance Status, Smoking Status, ICU Admission Before, Emergency Department Present Before
####################### 
#Numeric Variables: BMI, Weight, Height, SBP, Pulse, ICU Admission Counts, EDP Counts, 

generateErrorFreeCatVars <- function(ncat = 40, range = 6, seed = 1234){
  set.seed(seed)
  cats <- lapply(1:ncat, function(i){1:sample(2:range, 1)})
  probs <- lapply(1:ncat, function(i){0.1 + (x <- runif(length(cats[[i]])))/sum(x) * (1 - 0.1*length(cats[[i]]))})
  
  n3 <- sample(1:5, 1)
  n2 <- sample(2:10, 1)
  n1 <- ncat - n3 * 3 - n2 * 2
  pairsVars <- list(1:n1, (n1 + 1):(n1 + n2 * 2), (n1 + n2 * 2 + 1):(n1 + n2 * 2 + n3 * 3))
  
  return (list(cats = cats, probs = probs, pairsVars = pairsVars))
  #save(cats, probs, file = "./ErrorFreeCatVars.RData")
}
simErrorFreeCatVars = function(data, p, cat1 = 1:2, cat2 = 1:3, names){
  tb <- expand.grid(cat1, cat2)
  colnames(tb) <- c(names[1], names[2])
  tb$idx <- 1:nrow(tb)
  tb$prob <- p
  
  data$idx <- apply(rmultinom(nrow(data), size = 1, prob = tb$prob), 2, which.max)
  data <- merge(data, subset(tb, select = -prob), by = 'idx', all.x = T)
  data$idx <- NULL
  data <- data[order(data$id),]
  return (data)
}


generateSurvivalData <- function(digit, n = 100, 
                                 p_total = 120, prop_epv = 0.4, prop_cat = 0.6, 
                                 seed = 1234) {
  set.seed(seed)
  
  pop <- data.table(id = 1:n)
  expit <- function(x) {exp(x)/(1+exp(x))}
  
  ### Simulate Multivariate Normal Covariates ###
  
  nepv <- round(p_total * prop_epv)
  nefv <- p_total - nepv
  
  nefv_cat <- round(nefv * prop_cat)
  nefv_cont <- nefv - nefv_cat
  
  for (i in 1:nefv_cat){
    cat1 <- 1:sample(2:6, 1)
    cat2 <- 1:sample(2:6, 1)
    pop <- sim_catvars(pop, p = diff(sort(c(0, runif(length(cat1) * length(cat2) - 1), 1))), 
                       cat1 = cat1, cat2 = cat2,
                       names = )
  }
  
  X <- rmvnorm(n, mean = rep(0, p_total), sigma = diag(p_total))
  X <- as.data.table(X)
  setnames(X, paste0("X", 1:p_total))
  
  ### Split Covariates into Groups ###
  p_epv  <- round(p_total * prop_epv)
  p_vepv <- round(p_total * prop_vepv)
  p_efv  <- p_total - p_epv - p_vepv
  
  # EPV: Error-Prone Variables (simulate measurement error)
  EPV_true <- X[, 1:p_epv, with = FALSE]
  EPV_obs  <- copy(EPV_true)
  EPV_obs  <- EPV_obs + data.table(matrix(rnorm(n * p_epv, mean = 0, sd = 0.5), nrow = n))
  setnames(EPV_obs, paste0("EPV", 1:p_epv))
  
  # VEPV: Validated Error-Prone Variables (simulate with error, but true values are saved separately)
  VEPV_true <- X[, (p_epv + 1):(p_epv + p_vepv), with = FALSE]
  VEPV_obs  <- copy(VEPV_true)
  VEPV_obs  <- VEPV_obs + data.table(matrix(rnorm(n * p_vepv, mean = 0, sd = 0.5), nrow = n))
  setnames(VEPV_obs, paste0("VEPV", 1:p_vepv))
  
  # EFV: Error-Free Variables (observed exactly)
  EFV <- X[, (p_epv + p_vepv + 1):p_total, with = FALSE]
  setnames(EFV, paste0("EFV", 1:p_efv))
  
  ### Convert a Percentage of EPV and EFV to Categorical ###
  # For EPV: randomly select cat_percent columns and discretize them
  num_cat_epv <- round(ncol(EPV_obs) * cat_percent)
  if (num_cat_epv > 0) {
    cat_epv_cols <- sample(colnames(EPV_obs), size = num_cat_epv, replace = FALSE)
    for (col in cat_epv_cols) {
      EPV_obs[[col]] <- cut(EPV_obs[[col]],
                            breaks = quantile(EPV_obs[[col]], probs = seq(0, 1, length.out = 4), na.rm = TRUE),
                            include.lowest = TRUE,
                            labels = c("Low", "Medium", "High"))
    }
  }
  
  # For EFV: randomly select cat_percent columns and discretize them
  num_cat_efv <- round(ncol(EFV) * cat_percent)
  if (num_cat_efv > 0) {
    cat_efv_cols <- sample(colnames(EFV), size = num_cat_efv, replace = FALSE)
    for (col in cat_efv_cols) {
      EFV[[col]] <- cut(EFV[[col]],
                        breaks = quantile(EFV[[col]], probs = seq(0, 1, length.out = 4), na.rm = TRUE),
                        include.lowest = TRUE,
                        labels = c("Low", "Medium", "High"))
    }
  }
  
  ### Combine All Observed Covariates ###
  covariates_obs <- cbind(EPV_obs, VEPV_obs, EFV)
  pop <- cbind(pop, covariates_obs)
  
  ### Simulate Survival Outcome ###
  # For the survival model, we use the true (error-free) covariate matrix X.
  # Here, we assume a Cox proportional hazards model:
  #   hazard(t) = h0 * exp(lp), where lp = X * beta
  beta <- rnorm(p_total, mean = 0, sd = 0.2)  # Random coefficients for illustration
  lp <- as.matrix(X) %*% beta
  baseline_hazard <- 0.01
  
  # Generate survival times
  survival_time <- -log(runif(n)) / (baseline_hazard * exp(lp))
  # Simulate independent censoring times
  censoring_time <- rexp(n, rate = 0.001)
  time <- pmin(survival_time, censoring_time)
  event <- as.numeric(survival_time <= censoring_time)
  
  # Append survival outcome to the population data.table
  pop[, time := time]
  pop[, event := event]
  
  ### Save the Data ###
  output_dir <- '../SurvivalData/Output'
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  save(pop, file = paste0(output_dir, "/SurvivalData_", digit, ".RData"), compress = "xz")
  
  return(pop)
}

# Example usage:
# This will simulate a survival population with 4000 individuals and 120 covariates,
# then save the data to "../SurvivalData/Output/SurvivalData_digit.RData".
simulated_pop <- generateSurvivalData(digit = "001", n = 4000, p_total = 120)
head(simulated_pop)
