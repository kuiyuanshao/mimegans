#################################### Generate Population ####################################
# T2D across the whole population
lapply(c("LDlinkR", "dplyr", "purrr", "mvtnorm", "stringr", "tidyr", "haven"), require, character.only = T)
source("00_utils_functions.R")
generateData <- function(n, seed){
  set.seed(seed)
  n <- n
  data <- data.frame(ID = as.character(1:n))
  # AGE: 18 ~ 90
  data$AGE <- pmin(90, pmax(18, rnorm(n, 50, 15)))
  # SEX: 0 For 45% Female, 1 For 55% Male
  data$SEX <- as.logical(rbinom(n, 1, 0.55))
  # RACE: 50% EUR, 20% AFR, 10% EAS, 10% SAS, 10% AMR
  data$RACE <- sample(c("EUR", "AFR", "AMR", "SAS", "EAS"), 
                      size = n, replace = T, c(0.5, 0.2, 0.1, 0.1, 0.1)) 
  data$RACE <- factor(data$RACE, levels = c("EUR", "AFR", "AMR", "SAS", "EAS"))
  # INSURANCE: 70% has insurance
  data$INSURANCE <- as.logical(rbinom(n, 1, prob = 0.7))
  # REGION & URBAN:
  getRandProp <- function(i){
    set.seed(1)
    v <- abs(rnorm(i))
    v <- v / sum(v)
    return (v)
  }
  data$REGION <- sample(as.character(1:20), size = n,
                        replace = T, prob = getRandProp(20))
  data$URBAN <- data$REGION %in% as.character(1:15)
  # HOUSEHOLD INCOME:
  data$INCOME <- sample(as.character(1:5), size = n,
                        replace = T, prob = getRandProp(5))
  # EDUCATION YEARS:
  data$EDU <- rpois(n, 14)
  # 15 Genotypes
  genoInfo <- loadGenotypeInfo()
  data <- simGenotypes(data, genoInfo)
  # Load Covariates Info
  covarInfo <- loadCovarInfo(genoInfo)
  # HEIGHT: 149 ~ 
  data$HEIGHT <- pmax(149, simCovs(data, covarInfo$formu$HEIGHT, covarInfo$betas$HEIGHT, covarInfo$type$HEIGHT, covarInfo$sigma$HEIGHT))
  # SMOKE: 
  data$SMOKE <- simCovs(data, covarInfo$formu$SMOKE, covarInfo$betas$SMOKE, covarInfo$type$SMOKE)
  # EXER:
  data$EXER <- simCovs(data, covarInfo$formu$EXER, covarInfo$betas$EXER, covarInfo$type$EXER)
  # ALC:
  data$ALC <- simCovs(data, covarInfo$formu$ALC, covarInfo$betas$ALC, covarInfo$type$ALC)
  # BMI: 10 ~ 
  data$BMI <- simCovs(data, covarInfo$formu$BMI, covarInfo$betas$BMI, covarInfo$type$BMI, covarInfo$sigma$BMI)
  # MARRIAGE:
  data$MARRIAGE <- simCovs(data, covarInfo$formu$MARRIAGE, covarInfo$betas$MARRIAGE, covarInfo$type$MARRIAGE)
  # Medication Count:
  data$MED_Count <- simCovs(data, covarInfo$formu$MED_Count, covarInfo$betas$MED_Count, covarInfo$type$MED_Count)
  # RENAL GROUP: 
  data <- cbind(data, simCovs(data, covarInfo$formu$RENAL, covarInfo$betas$RENAL, covarInfo$type$RENAL, covarInfo$sigma$RENAL))
  # LIPIDS: LDL is in inv-norm scale
  data <- cbind(data, simCovs(data, covarInfo$formu$LIPIDS, covarInfo$betas$LIPIDS, covarInfo$type$LIPIDS, covarInfo$sigma$LIPIDS))
  # HEMA: HCT is in inv-norm scale
  data <- cbind(data, simCovs(data, covarInfo$formu$HEMA, covarInfo$betas$HEMA, covarInfo$type$HEMA, covarInfo$sigma$HEMA))
  # NUTRIENTS:
  data <- cbind(data, simCovs(data, covarInfo$formu$NUTRIENTS, covarInfo$betas$NUTRIENTS, covarInfo$type$NUTRIENTS, covarInfo$sigma$NUTRIENTS))
  # LIVER:
  data <- cbind(data, simCovs(data, covarInfo$formu$LIVER, covarInfo$betas$LIVER, covarInfo$type$LIVER, covarInfo$sigma$LIVER))
  # DIABETE MEASUREMENTS:
  data <- cbind(data, simCovs(data, covarInfo$formu$DIABETE, 
                              covarInfo$betas$DIABETE, covarInfo$type$DIABETE, covarInfo$sigma$DIABETE))
  # Ferritin:
  data$Ferritin <- simCovs(data, covarInfo$formu$FERRITIN, 
                           covarInfo$betas$FERRITIN, covarInfo$type$FERRITIN, covarInfo$sigma$FERRITIN)
  # VITAL SIGNS:
  data <- cbind(data, simCovs(data, covarInfo$formu$VITALS, covarInfo$betas$VITALS, covarInfo$type$VITALS, covarInfo$sigma$VITALS))
  
  for (i in c(29, (32:76)[!(32:76) %in% (56:59)])){
    data[, i] <- exp(data[, i]) - 1
  }
  # HYPENTERSION
  data$SBP <- pmin(pmax(data$SBP, 90), 200)
  data$HYPERTENSION <- with(data, SBP >= 140)
  data$Glucose <- data$Glucose + 10
  data$F_Glucose <- data$F_Glucose + 6
  data$HbA1c <- data$HbA1c + 40
  # WEIGHT:
  data$WEIGHT <- round(data$BMI * (data$HEIGHT / 100)^2, 3)
  
  age_penalty <- 0.5 * (data$AGE - 40) * (data$AGE > 40) # Extra creatinine for age > 40
  data$Creatinine <- data$Creatinine + 20 + age_penalty
  
  calculate_eGFR <- function(creat, age, sex) {
    scr <- creat / 88.4 # convert to mg/dL
    k <- ifelse(sex, 0.9, 0.7) 
    alpha <- ifelse(sex, -0.411, -0.329)
    factor_sex <- ifelse(sex, 1, 1.012) 
    egfr <- 142 * (pmin(scr / k, 1) ^ alpha) * (pmax(scr / k, 1) ^ -1.200) * (0.9938 ^ age) * factor_sex
    egfr <- pmin(round(egfr, 0), 140)
    return(egfr)
  }
  data$eGFR <- calculate_eGFR(data$Creatinine, data$AGE, data$SEX)
  
  # Adding Measurement Errors:
  # GENOTYPES:
  GENO_M <- matrix(
    c(0.99, 0.005, 0.005,
      0.005, 0.99, 0.005,
      0.005, 0.005, 0.99),
    nrow = 3, byrow = TRUE,
    dimnames = list(0:2, 
                    0:2)
  )
  for (name in genoInfo$rsids$refsnp_id){
    data[[paste0(name, "_STAR")]] <- sapply(data[[name]], 
                                            function(true_val) {sample(as.character(0:2), 
                                                                       size = 1, 
                                                                       prob = GENO_M[true_val, ])})
  }
  
  # T_I: Self-Reported Time Interval between Treatment Initiation SGLT2 and T2D Diagnosis (Months)
  # 1. SCALING 
  # ----------------------------------------------------------------------------
  A1c_sc  <- (data$HbA1c - 53) / 15 
  Age_sc  <- (data$AGE - 60) / 15
  eGFR_sc <- (data$eGFR - 60) / 20
  BMI_sc  <- (data$BMI - 30) / 5
  geno_eff <- as.numeric(data$rs4506565) 
  
  # 2. PROPENSITY FOR ACUTE SWITCH
  # Boosted coefficients to ensure clean separation and significance
  prob_acute <- plogis(-2.5 +                
                         0.6 * A1c_sc +        
                         0.3 * BMI_sc +        
                         0.3 * geno_eff +      
                         0.6 * as.numeric(data$INSURANCE) +
                         # RACE Effect: Asians/Blacks might have higher acute risk
                         0.4 * as.numeric(data$RACE %in% c("AFR", "SAS")))
  
  is_acute <- (runif(n) < prob_acute)
  
  # 3. LINEAR PREDICTOR (Hazard Strength)
  # - Only ONE Quadratic term (HbA1c)
  # - Boosted SMOKE/RACE coefficients to prevent non-significance
  
  # RACE Effect Vector
  # EUR=0, AFR=0.3, AMR=0.1, SAS=0.4, EAS=0.2
  race_eff <- case_when(
    data$RACE == "AFR" ~ 0.4,
    data$RACE == "SAS" ~ 0.5,
    data$RACE == "EAS" ~ 0.3,
    data$RACE == "AMR" ~ 0.2,
    TRUE ~ 0.0
  )
  
  eta_I <- (0.4 * A1c_sc + 0.1 * A1c_sc^2) +      # HbA1c: Quadratic Only
    (0.3 * eGFR_sc) +                      # eGFR: Linear
    (0.2 * BMI_sc) +                       # BMI: Linear
    (-0.2 * A1c_sc * Age_sc) +             # Interaction
    (0.2 * geno_eff) +                     # Genotype
    (0.3 * as.numeric(data$SEX)) +         # Sex
    (0.5 * as.numeric(data$SMOKE == 1)) +  # SMOKE (Current=1): Strong effect
    race_eff                               # Race Effect
  
  # 4. TIME GENERATION
  # Acute: Mean ~3m
  T_acute <- (3.0 * exp(-0.4 * eta_I)) * (-log(runif(n)))^(1/1.2)
  
  # Routine: Mean ~20m
  T_routine <- rlnorm(n, meanlog = log(20) - 0.25 * eta_I, sdlog = 0.5)
  
  T_I <- ifelse(is_acute, T_acute, T_routine)
  
  # 5. CENSORING & TRUE OBSERVED
  # Weaker censoring to ensure we have enough events for significance
  c_rate <- 0.02 * exp(-0.01 * (data$AGE - 60) + 0.1 * as.numeric(data$URBAN))
  C <- pmin(rexp(n, c_rate), 36.0) 
  
  # TRUE OUTCOMES (For "True Model" Benchmark)
  data$T_I <- T_I
  data$C <- C
  data$EVENT <- as.integer(T_I <= C)
  data <- add_measurement_errors(data)
  data$T_I <- pmin(T_I, C)
  
  cols_to_transform <- setdiff(data_info_srs$cat_vars, c("R", "EVENT", "EVENT_STAR"))
  data[cols_to_transform] <- lapply(data[cols_to_transform], as.character)
  
  return (data)
}

add_measurement_errors <- function(data) {
  n <- nrow(data)
  
  # --- Helper for Categorical Sampling ---
  vectorized_choice <- function(prob_matrix) {
    cum_probs <- t(apply(prob_matrix, 1, cumsum))
    rowSums(runif(nrow(prob_matrix)) > cum_probs) + 1
  }
  
  # ============================================================================
  # 1. SURVIVAL & EVENT (Aggressive Misclassification & Delay)
  # ============================================================================
  
  # A. Censoring (Ghost Follow-up)
  # Base log-odds increased to -0.5 to create more ghosts (patients lost but not recorded)
  prob_ghost <- plogis(-0.5 - 0.02 * (data$AGE - 50) - 0.8 * as.numeric(data$INSURANCE))
  is_ghost <- (data$C < 36.0) & (runif(n) < prob_ghost)
  
  data$C_STAR <- data$C + rnorm(n, 0, 0.2)
  data$C_STAR[is_ghost] <- 36.0 + rnorm(sum(is_ghost), 0, 0.2) 
  data$C_STAR <- pmax(0.1, data$C_STAR)
  
  # B. Latent Event Time (Administrative Delay)
  # Rate modifier adjusted so delay is longer and more variable
  rate_mod <- 2.0 * (1 + 0.5 * as.numeric(data$INSURANCE)) 
  t_latent <- data$T_I + rgamma(n, shape = 3.0, rate = rate_mod) 
  
  # C. Observed Outcomes (Lower Sensitivity / Specificity)
  data$EVENT_STAR <- 0 
  data$T_I_STAR <- data$C_STAR
  
  # Sensitivity: Urban effect stronger, base lower (harder to capture events)
  sens_prob <- plogis(1.5 + 0.5 * as.numeric(data$URBAN)) 
  captured <- (runif(n) < sens_prob)
  
  valid <- (data$EVENT == 1) & captured & (t_latent <= data$C_STAR)
  data$EVENT_STAR[valid] <- 1
  data$T_I_STAR[valid] <- t_latent[valid]
  
  # False Positives: Specificity reduced to 0.98 (2% random spurious events)
  false_alarm <- (data$EVENT == 0) & (runif(n) > 0.98)
  fp_idx <- which(false_alarm)
  if(length(fp_idx) > 0){
    data$EVENT_STAR[fp_idx] <- 1
    data$T_I_STAR[fp_idx] <- runif(length(fp_idx), 0, data$C_STAR[fp_idx])
  }
  
  # ============================================================================
  # 2. RENAL (Noisier Creatinine & eGFR)
  # ============================================================================
  # Increased protein effect (0.15) and analytical noise (0.15)
  prot_eff <- 0.15 * pmax(0, data$PROTEIN_INTAKE)
  gluc_int <- ifelse(data$Glucose > 15, 0.10, 0.0)
  noise <- rnorm(n, 0, 0.15) 
  
  data$Creatinine_STAR <- round(data$Creatinine * exp(prot_eff + gluc_int + noise), 0)
  
  # Recalculate eGFR
  scr <- data$Creatinine_STAR / 88.4
  k <- ifelse(data$SEX, 0.9, 0.7); alpha <- ifelse(data$SEX, -0.411, -0.329)
  sex_f <- ifelse(data$SEX, 1.0, 1.018)
  data$eGFR_STAR <- 141 * (pmin(scr/k, 1)^alpha) * (pmax(scr/k, 1)^-1.209) * (0.993^data$AGE) * sex_f
  data$eGFR_STAR <- pmin(round(data$eGFR_STAR, 0), 140)
  
  # ============================================================================
  # 3. DIABETES (Stronger HbA1c Bias)
  # ============================================================================
  # HbA1c: "Optimism Bias" slope increased to -0.40
  bias_a1c <- -0.40 * pmax(0, data$HbA1c - 55) 
  sigma_a1c <- exp(log(5.0) + 0.01*(data$AGE-50) - 0.1*(data$EDU-14)) 
  data$HbA1c_STAR <- round(data$HbA1c + bias_a1c + rnorm(n, 0, sigma_a1c), 0)
  
  # Glucose: Higher volatility
  sigma_fg <- 1.0 * (1 + 0.05*(data$BMI-25)) 
  data$F_Glucose_STAR <- pmax(3.0, round(data$F_Glucose + 0.5 + rt(n, 4)*sigma_fg, 1))
  data$Glucose_STAR <- pmax(3.0, round(data$Glucose + rnorm(n, 0, 3.0), 1))
  data$Insulin_STAR <- round(data$Insulin * exp(rnorm(n, 0, 0.25)), 1)
  
  # ============================================================================
  # 4. BIOMETRICS (Stronger Vanity Bias)
  # ============================================================================
  # Weight: Under-reporting slope -0.5 kg per kg over 75
  TRUE_WEIGHT <- data$BMI * (data$HEIGHT / 100)^2
  bias_w <- -0.5 * pmax(0, TRUE_WEIGHT - 75)
  WEIGHT_STAR <- TRUE_WEIGHT + bias_w + rnorm(n, 0, 3.0)
  
  # Height: Over-reporting (Men +2.5cm)
  bias_h <- ifelse(data$SEX, 2.5, 1.0) + 0.1 * pmax(0, data$AGE - 50)
  data$HEIGHT_STAR <- round(data$HEIGHT + bias_h + rnorm(n, 0, 2.0), 0)
  
  data$BMI_STAR <- round(WEIGHT_STAR / (data$HEIGHT_STAR / 100)^2, 1)
  
  # SBP: Stronger White Coat + Cuff Error
  white_coat <- pmax(0, 5.0 + 0.2 * (data$AGE - 50))
  cuff_error <- rep(0, n)
  obese_idx <- which(data$BMI > 30)
  if(length(obese_idx) > 0) {
    bad_cuff <- sample(obese_idx, size = 0.4 * length(obese_idx)) # 40% wrong cuff
    cuff_error[bad_cuff] <- 8.0 + 0.6 * (data$BMI[bad_cuff] - 30)
  }
  data$SBP_STAR <- round(data$SBP + white_coat + cuff_error + rnorm(n, 0, 10), 0)
  
  # Lipids: Non-fasting artifact
  gluc_diff <- (data$F_Glucose_STAR - data$F_Glucose) 
  non_fasting <- (gluc_diff > 2.0)
  tg_bias <- rep(0, n); tg_bias[non_fasting] <- 0.6 * data$Triglyceride[non_fasting]
  data$Triglyceride_STAR <- round(data$Triglyceride + tg_bias + rnorm(n, 0, 0.3), 1)
  
  ldl_err <- -(data$Triglyceride_STAR - data$Triglyceride) / 2.2
  data$LDL_STAR <- round(data$LDL + ldl_err + rnorm(n, 0, 0.2), 1)
  
  # ============================================================================
  # 5. NUTRIENTS (High Reporting Error)
  # ============================================================================
  log_RR <- -0.20 - 0.02*(data$BMI - 25) + rnorm(n, 0, 0.35)
  RR <- exp(log_RR)
  
  data$KCAL_INTAKE_STAR <- round(data$KCAL_INTAKE * RR, 0)
  data$PROTEIN_INTAKE_STAR <- round(data$PROTEIN_INTAKE * RR * exp(rnorm(n, 0, 0.15)), 1)
  data$Na_INTAKE_STAR <- round(data$Na_INTAKE * RR * exp(rnorm(n, 0, 0.5)), 0)
  data$K_INTAKE_STAR <- round(data$K_INTAKE * RR * exp(rnorm(n, 0, 0.4)), 0)
  
  # ============================================================================
  # 6. CATEGORICAL (High Misclassification)
  # ============================================================================
  data$SMOKE <- as.integer(data$SMOKE); data$ALC <- as.integer(data$ALC)
  data$EXER <- as.integer(data$EXER); data$INCOME <- as.integer(data$INCOME)
  
  # SMOKE: High denial rate
  P_S <- matrix(0, n, 3)
  idx <- which(data$SMOKE==1); if(length(idx)>0) { 
    p <- pmin(0.40 + 0.05*(data$EDU[idx]-14), 0.9) 
    P_S[idx,] <- cbind(1-p, p*0.6, p*0.4) 
  }
  idx <- which(data$SMOKE==2); if(length(idx)>0) {
    p <- 0.2 
    P_S[idx,] <- cbind(0.05, 1-p, p)
  }
  idx <- which(data$SMOKE==3); if(length(idx)>0) P_S[idx,] <- rep(c(0.02,0.05,0.93), each=length(idx))
  data$SMOKE_STAR <- vectorized_choice(P_S)
  
  # ALC: High denial rate
  P_A <- matrix(0, n, 3)
  idx <- which(data$ALC==1); if(length(idx)>0) P_A[idx,] <- rep(c(0.85,0.1,0.05), each=length(idx))
  idx <- which(data$ALC==2); if(length(idx)>0) P_A[idx,] <- rep(c(0.15,0.85,0.0), each=length(idx))
  idx <- which(data$ALC==3); if(length(idx)>0) {
    p <- 0.5 
    P_A[idx,] <- cbind(p, 0.05, 1-p-0.05)
  }
  data$ALC_STAR <- vectorized_choice(P_A)
  
  # EXER: Wishful thinking (Low -> Normal)
  P_E <- matrix(0, n, 3)
  idx <- which(data$EXER==1); if(length(idx)>0) P_E[idx,] <- rep(c(0.8,0.1,0.1), each=length(idx))
  idx <- which(data$EXER==2); if(length(idx)>0) {
    p <- 0.4 # 40% of low exercise report normal
    P_E[idx,] <- cbind(p, 1-p, 0.0)
  }
  idx <- which(data$EXER==3); if(length(idx)>0) P_E[idx,] <- rep(c(0.2,0.0,0.8), each=length(idx))
  data$EXER_STAR <- vectorized_choice(P_E)
  
  # INCOME: Bias based on Edu mismatch
  P_I <- matrix(0, n, 5)
  for(k in 1:5){
    idx <- which(data$INCOME == k); if(length(idx)==0) next
    row_p <- rep(0, 5); row_p[k] <- 0.7 # Base accuracy reduced to 70%
    if(k>1) row_p[k-1] <- 0.15; if(k<5) row_p[k+1] <- 0.15
    mat_p <- matrix(rep(row_p, length(idx)), nrow=length(idx), byrow=T)
    if(k==1) { sub <- which(data$EDU[idx]>16); mat_p[sub, 1:2] <- c(0.4, 0.6) }
    if(k==5) { sub <- which(data$EDU[idx]<12); mat_p[sub, 4:5] <- c(0.3, 0.7) }
    P_I[idx,] <- mat_p / rowSums(mat_p)
  }
  data$INCOME_STAR <- vectorized_choice(P_I)
  
  # EDU: Credentialism + Age Noise
  bias_edu <- ifelse(data$INCOME >= 4 & data$EDU < 12, 2.0, 0.0) # +2 years bias
  sigma_edu <- ifelse(data$AGE > 60, 2.0, 1.0)
  data$EDU_STAR <- round(data$EDU + bias_edu + rnorm(n, 0, sigma_edu), 0)
  data$EDU_STAR <- pmax(0, data$EDU_STAR)
  
  return(data)
}

loadGenotypeInfo <- function(){
  rsids <- data.frame(refsnp_id = c("rs10811661", "rs7756992", "rs11708067", "rs17036101", "rs17584499",
                                    "rs1111875", "rs4402960", "rs4607103", "rs7754840", "rs9300039",
                                    "rs5015480", "rs9465871", "rs4506565", "rs5219", "rs358806"),
                      chr = c(9, 6, 3, 3, 9, 
                              10, 3, 3, 6, 11,
                              10, 6, 10, 11, 3))
  unique_chrs <- unique(rsids$chr)
  race <- c(unique(list_pop()$super_pop_code)[-1])
  retrieveFreq <- function(rsid, race){
    LDhap(snps = rsid,
          pop = race,
          token = "979f246a6b57")
  }
  retrieveCorr <- function(rsids, chr, race){
    LDmatrix(snp = rsids$refsnp_id[rsids$chr == chr],
             pop = race, 
             r2d = "r2",
             token = "979f246a6b57") 
  }
  if (!file.exists("./data/genoFreq.RData")){
    genoFreq <- vector("list", 150)
    m <- 1
    for (i in rsids$refsnp_id){
      for (j in race){
        mat <- retrieveFreq(i, j) %>%
          dplyr::rename(allele = i) %>%
          mutate(Race = j, refsnp_id = i)
        genoFreq[[m]] <- mat
        m <- m + 1
      }
    }
    genoFreq <- bind_rows(genoFreq)
    genoFreq <- merge(genoFreq, rsids)
    save(genoFreq, file = "./data/genoFreq.RData")
  }else{
    load("./data/genoFreq.RData")
  }
  if (!file.exists("./data/genoCorr.RData")){
    genoCorr <- list()
    for (i in unique_chrs){
      chr_list <- vector("list", 5)
      if (sum(rsids$chr == i) > 1){
        for (j in race){
          mat <- retrieveCorr(rsids, i, j) %>%
            mutate(Race = j, chr = i)
          chr_list[[j]] <- mat
        }
        genoCorr[[as.character(i)]] <- bind_rows(chr_list) 
      }
    }
    save(genoCorr, file = "./data/genoCorr.RData")
  }else{
    load("./data/genoCorr.RData")
  }
  if (!file.exists("./data/genoTrait.RData")){
    pat <- "^[+-]?[0-9]+(?:\\.[0-9]+)?-[+-]?[0-9]+(?:\\.[0-9]+)?"
    trait <- LDtrait(snps = rsids$refsnp_id,
                     pop = "ALL",
                     win_size = "1",
                     token = "979f246a6b57")
    trait <- trait %>%
      mutate(Beta_or_OR = str_extract(Beta_or_OR, pat)) %>%
      separate(Beta_or_OR, into = c("low", "high"), sep = "-", convert = TRUE) %>%
      mutate(Beta = (low + high) / 2)
    phenos <- list(
      BMI    = "Body mass index",
      T2D    = c("Type 2 diabetes",
                 "Type ii diabetes",
                 "Type 2 diabetes (PheCode 250.2)",
                 "Type 2 diabetes with renal manifestations (PheCode 250.22)",
                 "Type 2 diabetes with ophthalmic manifestations (PheCode 250.23)",
                 "Type 2 diabetes with neurological manifestations (PheCode 250.24)"),
      HCT    = "hematocrit (maximum, inv-norm transformed)",
      SBP    = "Systolic blood pressure",
      Med    = c("Medication use (drugs used in diabetes)",
                 "Takes medication for Diabetes/sugar?"),
      Glucose = "Glucose (finger stick, mean, inv-norm transformed)",
      F_Glucose= c("Fasting blood glucose",
                   "Fasting plasma glucose",
                   "Fasting glucose"),
      LDL    = c("low density lipoprotein cholesterol (LDLC, mean, inv-norm transformed)"),
      HbA1c  = "Hemoglobin A1c (HbA1c, mean, inv-norm transformed)",
      HEIGHT = "Height",
      Triglyceride = "Triglyceride levels",
      ALT    = c("Alanine levels (UKB data field 23460)", "Alanine levels",
                 "Alanine aminotransferase levels"),
      ALP    = "Serum alkaline phosphatase levels",
      Albumin= "Serum albumin levels",
      Insulin= "Fasting insulin"
    )
    all_mats <- imap(phenos, function(trait_names, pheno_code) {
      trait %>%
        filter(GWAS_Trait %in% trait_names) %>%
        group_by(Query) %>%
        summarise(Beta = mean(Beta, na.rm = TRUE), .groups = "drop") %>%
        mutate(PHENO = pheno_code)
    })
    genoTrait <- bind_rows(all_mats)
    save(genoTrait, file = "./data/genoTrait.RData")
  }else{
    load("./data/genoTrait.RData")
  }
  return (list(rsids = rsids, genoFreq = genoFreq, genoCorr = genoCorr, genoTrait = genoTrait))
}

loadCovarInfo <- function(genoInfo){
  if (!file.exists("./data/covarInfo.RData")){
    load("./data/labtest_parameters.RData")
    # HEIGHT ~ SEX + Genos
    formu_HEIGHT <- as.formula(paste0(" ~ SEX + ", paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "HEIGHT", ]$Query, collapse = " + ")))
    betas_HEIGHT <- c(162, 13, transform_betas(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "HEIGHT", ]$Beta))
    sigma_HEIGHT <- 7
    # SMOKE ~ AGE + SEX + RACE (Current Smoker, Ex-Smoker, Never Smoked)
    formu_SMOKE <- as.formula(paste0("~", Mod_list$SMOKE$formula[3]))
    betas_SMOKE <- Mod_list$SMOKE$coeff
    # EXER ~ AGE + SEX + RACE + SMOKE (Normal, Low, High)
    formu_EXER <- as.formula(~ AGE + SEX + RACE + SMOKE)
    betas_EXER <- rbind(
      c(0.5, 0.01, -0.2, 0.1, -0.1, 0.05, 0.08, 0.4, 0.2), # Low vs. Normal
      c(-0.5, -0.02, 0.3, -0.1, 0.1, -0.05, -0.08, -0.3, -0.1) # High vs. Normal
    )
    # ALC ~ AGE + SEX + SMOKE + EXER (Moderate, None, Heavy)
    formu_ALC <- as.formula(~ AGE + SEX + RACE + SMOKE + EXER)
    betas_ALC <- rbind(
      c(-1.0, 0.01, -0.5, 0.2, 0.3, 0.1, 0.15, -0.4, -0.2,  0.3, -0.3), # None vs. Moderate
      c(-1.5, -0.02, 0.6, 0.1, -0.2, 0.05, 0.1, 0.5, 0.2, 0.2, -0.4) # Heavy vs. Moderate
    )
    # MARRIAGE ~ AGE + SEX + RACE + BMI (None, Married, Divorced)
    formu_MARRIAGE <- as.formula(~ AGE + SEX + RACE + BMI)
    betas_MARRIAGE <- rbind(
      c(-2.0, 0.05, -0.3, -0.2, 0.1, -0.1, -0.05, -0.02), # Married vs. None
      c(-3.0, 0.03, 0.2, 0.1, -0.1, 0.05, 0.08, 0.01) # Divorced vs. None
    )
    # BMI ~ SEX + SMOKE + RACE + AGE + GENOs
    formu_BMI <- as.formula(paste0("~",  Mod_list$BMI$BMI$formula[3], " + ", 
                                   paste0(unique(genoInfo$genoTrait$Query[genoInfo$genoTrait$PHENO == "BMI"]), 
                                          collapse = " + ")))
    betas_BMI <- c(Mod_list$BMI$BMI$coeff, 
                   transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "BMI"]))
    sigma_BMI <- Sigma$BMI
    # Medication Count:
    formu_MED_Count <- as.formula(paste0("~", Mod_list$MED_Count$MED_Count$formula[3]))
    betas_MED_Count <- Mod_list$MED_Count$MED_Count$coeff
    # Diabetes
    formu_DIABETE <- list(Glucose = as.formula(paste("~", Mod_list$Diabete$Glucose$formula[3], "+", 
                                                     paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "Glucose", ]$Query, collapse = " + "))),
                          F_Glucose = as.formula(paste("~", Mod_list$Diabete$F_Glucose$formula[3], "+", 
                                                       paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "F_Glucose", ]$Query, collapse = " + "))),
                          HbA1c = as.formula(paste("~", Mod_list$Diabete$HbA1c$formula[3], "+", 
                                                   paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "HbA1c", ]$Query, collapse = " + "))),
                          Insulin = as.formula(paste("~", Mod_list$Diabete$Insulin$formula[3], "+", 
                                                     paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "Insulin", ]$Query, collapse = " + "))))
    betas_DIABETE <- list(Glucose = c(Mod_list$Diabete$Glucose$coeff, 
                                      transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "Glucose"])), 
                          F_Glucose = c(Mod_list$Diabete$F_Glucose$coeff,
                                        transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "F_Glucose"])),
                          HbA1c = c(Mod_list$Diabete$HbA1c$coeff,
                                    transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "HbA1c"])),
                          Insulin = c(Mod_list$Diabete$Insulin$coeff,
                                      transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "Insulin"])))  # Insulin pmol/L
    sigma_DIABETE <- Sigma$Diabete
    # Nutrients
    formu_NUTRIENTS <- list(Na_INTAKE = as.formula(~ AGE + SEX + BMI + RACE),
                            K_INTAKE = as.formula(~ AGE + SEX + BMI + RACE),
                            KCAL_INTAKE = as.formula(~ AGE + SEX + BMI + RACE),
                            PROTEIN_INTAKE = as.formula(~ AGE + SEX + BMI + RACE))
    betas_NUTRIENTS <- list(Na_INTAKE = c(7.66, -0.00084, 0.27, 0.0177, 0.009, -0.10, -0.15, 0.03),
                            K_INTAKE = c(7.2, 0.0075, 0.177, 0.00956, 0.04, -0.08, -0.12, 0.02),
                            KCAL_INTAKE = c(7.4, -0.00256, 0.16, 0.014, 0.20, -0.4, -0.6, 0.12),
                            PROTEIN_INTAKE = c(4, 0.000459, 0.187, 0.0166, 0.02, -0.13, -0.14, 0.1))
    sigma_NUTRIENTS <- matrix(c(0.1820, 0.0350, 0.0250, 0.0360,
                                0.0350, 0.1530, 0.0120, 0.0330,
                                0.0250, 0.0120, 0.1070, 0.0170,
                                0.0360, 0.0330, 0.0170, 0.1220), nrow = 4, byrow = TRUE)
    # LIVER
    formu_LIVER <- list(AST = as.formula(paste("~", Mod_list$Liver$AST$formula[3])), 
                        ALT = as.formula(paste("~", Mod_list$Liver$ALT$formula[3], "+", 
                                               paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "ALT", ]$Query, collapse = " + "))),
                        ALP = as.formula(paste("~", Mod_list$Liver$ALP$formula[3], "+", 
                                               paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "ALP", ]$Query, collapse = " + "))), 
                        GGT = as.formula(paste("~", Mod_list$Liver$GGT$formula[3])), 
                        Bilirubin = as.formula(paste("~", Mod_list$Liver$Bilirubin$formula[3])), 
                        Albumin = as.formula(paste("~", Mod_list$Liver$Albumin$formula[3], "+", 
                                                   paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "Albumin", ]$Query, collapse = " + "))),
                        Globulin = as.formula(paste("~", Mod_list$Liver$Globulin$formula[3])), 
                        Protein = as.formula(paste("~", Mod_list$Liver$Protein$formula[3])))
    betas_LIVER <- list(AST = Mod_list$Liver$AST$coeff,
                        ALT = c(Mod_list$Liver$ALT$coeff,
                                transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "ALT"])),
                        ALP = c(Mod_list$Liver$ALP$coeff,
                                transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "ALP"])), 
                        GGT = Mod_list$Liver$GGT$coeff,
                        Bilirubin = Mod_list$Liver$Bilirubin$coeff,
                        Albumin = c(Mod_list$Liver$Albumin$coeff,
                                    transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "Albumin"])),
                        Globulin = Mod_list$Liver$Globulin$coeff,
                        Protein = Mod_list$Liver$Protein$coeff)
    sigma_LIVER <- Sigma$Liver
    
    # INFLAM
    formu_FERRITIN <- as.formula(paste("~", Mod_list$Ferritin$Ferritin$formula[3]))
    betas_FERRITIN <- Mod_list$Ferritin$Ferritin$coeff
    sigma_FERRITIN <- Sigma$Ferritin
    
    # RENAL
    formu_RENAL <- list(Creatinine = as.formula(paste("~", Mod_list$Renal$Creatinine$formula[3])),
                        Urea = as.formula(paste("~", Mod_list$Renal$Urea$formula[3])),
                        Potassium = as.formula(paste("~", Mod_list$Renal$Potassium$formula[3])),
                        Sodium = as.formula(paste("~", Mod_list$Renal$Sodium$formula[3])),
                        Chloride = as.formula(paste("~", Mod_list$Renal$Chloride$formula[3])),
                        Bicarbonate = as.formula(paste("~", Mod_list$Renal$Bicarbonate$formula[3])),
                        Calcium = as.formula(paste("~", Mod_list$Renal$Calcium$formula[3])),
                        Magnesium = as.formula(paste("~", Mod_list$Renal$Magnesium$formula[3])), 
                        Phosphate = as.formula(paste("~", Mod_list$Renal$Phosphate$formula[3])))
    betas_RENAL <- list(Creatinine = Mod_list$Renal$Creatinine$coeff,
                        Urea = Mod_list$Renal$Urea$coeff,
                        Potassium = Mod_list$Renal$Potassium$coeff,
                        Sodium = Mod_list$Renal$Sodium$coeff,
                        Chloride = Mod_list$Renal$Chloride$coeff,
                        Bicarbonate = Mod_list$Renal$Bicarbonate$coeff,
                        Calcium = Mod_list$Renal$Calcium$coeff,
                        Magnesium = Mod_list$Renal$Magnesium$coeff, 
                        Phosphate = Mod_list$Renal$Phosphate$coeff)
    sigma_RENAL <- Sigma$Renal
    # LIPIDS
    formu_LIPIDS <- list(Triglyceride = as.formula(paste("~", Mod_list$Lipid$Triglyceride$formula[3], " + ",
                                                         paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "Triglyceride", ]$Query, collapse = " + "))), 
                         HDL = as.formula(paste("~", Mod_list$Lipid$HDL$formula[3])), 
                         LDL = as.formula(paste("~", Mod_list$Lipid$LDL$formula[3], " + ",
                                               paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "LDL", ]$Query, collapse = " + "))))
    betas_LIPIDS <- list(Triglyceride = c(Mod_list$Lipid$Triglyceride$coeff, 
                                          transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "Triglyceride"])), 
                         HDL = Mod_list$Lipid$HDL$coeff, 
                         LDL = c(Mod_list$Lipid$LDL$coeff, 
                                 transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "LDL"])))
    sigma_LIPIDS <- Sigma$Lipid
    # HEMATOLOGY
    formu_HEMA <- list(Hb = as.formula(paste("~", Mod_list$Haematology$Hb$formula[3])), 
                       HCT = as.formula(paste("~", Mod_list$Haematology$HCT$formula[3], " + ",
                                              paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "HCT", ]$Query, collapse = " + "))),
                       RBC = as.formula(paste("~", Mod_list$Haematology$RBC$formula[3])), 
                       WBC = as.formula(paste("~", Mod_list$Haematology$WBC$formula[3])), 
                       Platelet = as.formula(paste("~", Mod_list$Haematology$Platelet$formula[3])), 
                       MCV = as.formula(paste("~", Mod_list$Haematology$MCV$formula[3])), 
                       RDW = as.formula(paste("~", Mod_list$Haematology$RDW$formula[3])), 
                       Neutrophils = as.formula(paste("~", Mod_list$Haematology$Neutrophils$formula[3])), 
                       Lymphocytes = as.formula(paste("~", Mod_list$Haematology$Lymphocytes$formula[3])), 
                       Monocytes = as.formula(paste("~", Mod_list$Haematology$Monocytes$formula[3])), 
                       Eosinophils = as.formula(paste("~", Mod_list$Haematology$Eosinophils$formula[3])), 
                       Basophils = as.formula(paste("~", Mod_list$Haematology$Basophils$formula[3])))
    
    betas_HEMA <- list(Hb = Mod_list$Haematology$Hb$coeff, 
                       HCT = c(Mod_list$Haematology$HCT$coeff, 
                               transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "HCT"])),
                       RBC = Mod_list$Haematology$RBC$coeff,
                       WBC = Mod_list$Haematology$WBC$coeff,
                       Platelet = Mod_list$Haematology$Platelet$coeff,
                       MCV = Mod_list$Haematology$MCV$coeff,
                       RDW = Mod_list$Haematology$RDW$coeff,
                       Neutrophils = Mod_list$Haematology$Neutrophils$coeff,
                       Lymphocytes = Mod_list$Haematology$Lymphocytes$coeff,
                       Monocytes = Mod_list$Haematology$Monocytes$coeff,
                       Eosinophils = Mod_list$Haematology$Eosinophils$coeff,
                       Basophils = Mod_list$Haematology$Basophils$coeff)
    sigma_HEMA <- Sigma$Haematology
    
    # Vital Signs
    formu_VITALS <- list(SBP = as.formula(paste("~", Mod_list$Vital$SBP$formula[3], " + ",
                                                paste0(unique(genoInfo$genoTrait$Query[genoInfo$genoTrait$PHENO == "SBP"]), collapse = " + "))), 
                         Temperature = as.formula(paste("~", Mod_list$Vital$Temperature$formula[3])), 
                         HR = as.formula(paste("~", Mod_list$Vital$HR$formula[3])), 
                         SpO2 = as.formula(paste("~", Mod_list$Vital$SpO2$formula[3])))
    betas_VITALS <- list(SBP = c(Mod_list$Vital$SBP$coeff,
                                 transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "SBP"])),
                         Temperature = Mod_list$Vital$Temperature$coeff,
                         HR = Mod_list$Vital$HR$coeff,
                         SpO2 = Mod_list$Vital$SpO2$coeff)
    sigma_VITALS <- Sigma$Vital
    # PACK UP
    formu <- list(HEIGHT = formu_HEIGHT, SMOKE = formu_SMOKE, EXER = formu_EXER, 
                  ALC = formu_ALC, BMI = formu_BMI, MED_Count = formu_MED_Count, MARRIAGE = formu_MARRIAGE,
                  VITALS = formu_VITALS, RENAL = formu_RENAL, LIPIDS = formu_LIPIDS,
                  HEMA = formu_HEMA, NUTRIENTS = formu_NUTRIENTS, LIVER = formu_LIVER,
                  DIABETE = formu_DIABETE, FERRITIN = formu_FERRITIN)
    betas <- list(HEIGHT = betas_HEIGHT, SMOKE = betas_SMOKE, EXER = betas_EXER, 
                  ALC = betas_ALC, BMI = betas_BMI, MED_Count = betas_MED_Count, MARRIAGE = betas_MARRIAGE,
                  VITALS = betas_VITALS, RENAL = betas_RENAL, LIPIDS = betas_LIPIDS,
                  HEMA = betas_HEMA, NUTRIENTS = betas_NUTRIENTS, LIVER = betas_LIVER,
                  DIABETE = betas_DIABETE, FERRITIN = betas_FERRITIN)
    sigma <- list(HEIGHT = sigma_HEIGHT, BMI = sigma_BMI,
                  VITALS = sigma_VITALS, RENAL = sigma_RENAL, LIPIDS = sigma_LIPIDS,
                  HEMA = sigma_HEMA, NUTRIENTS = sigma_NUTRIENTS, LIVER = sigma_LIVER,
                  DIABETE = sigma_DIABETE, FERRITIN = sigma_FERRITIN)
    type <- list(HEIGHT = "Norm", SMOKE = "Cat", EXER = "Cat", 
                 ALC = "Cat", BMI = "Norm", MED_Count = "Pois", MARRIAGE = "Cat",
                 VITALS = "MVNorm", RENAL = "MVNorm", LIPIDS = "MVNorm",
                 HEMA = "MVNorm", NUTRIENTS = "MVNorm", LIVER = "MVNorm",
                 DIABETE = "MVNorm", FERRITIN = "Norm")
    save(formu, betas, sigma, type, file = "./data/covarInfo.RData")
  }else{
    load("./data/covarInfo.RData")
  }
  return (list(formu = formu, betas = betas, sigma = sigma, type = type))
}

simCovs <- function(data, formu, betas, type, sds = NULL){
  if (type == "Cat"){
    mm <- model.matrix(formu, data)
    lp <- exp(cbind(rep(0, nrow(data)), mm %*% t(betas)))
    probmat <- lp / rowSums(lp)
    cummat <- t(apply(probmat, 1, cumsum))
    result <- as.factor(max.col(cummat >= runif(nrow(data)), ties.method = "first"))
  }else if (type == "Norm"){
    mm <- model.matrix(formu, data)
    means <- mm %*% betas
    result <- round(rnorm(nrow(data), means, sds), 3)
  }else if (type == "MVNorm"){
    means <- lapply(1:length(formu), function(i) model.matrix(formu[[i]], data) %*% betas[[i]]) %>% bind_cols
    result <- t(apply(means, 1, function(mu) round(rmvnorm(1, mu, sds), 3)))
    result <- as.data.frame(result)
    names(result) <- gsub("formu_", "", names(formu))
  }else if (type == "Pois"){
    lambda <- exp(model.matrix(formu, data) %*% betas)
    result <- rpois(nrow(data), lambda = lambda)
  }
  return (result)
}

simGenotypes <- function(data, genoInfo){
  df <- data.frame()
  race <- unique(data$RACE)
  for (r in race){
    ind <- which(data$RACE == r)
    curr_subset <- subset(data, RACE == r)
    genoMat <- matrix(0, nrow = length(ind), ncol = 15)
    covnames <- c()
    m <- 1
    for (c in unique(genoInfo$rsids$chr)){
      fsub <- subset(genoInfo$genoFreq, Race == r & chr == c)
      probs <- as.numeric(fsub$Frequency[seq(2, length(fsub$Frequency), by = 2)])
      names(probs) <- fsub$refsnp_id[seq(2, length(fsub$Frequency), by = 2)]
      if (as.character(c) %in% names(genoInfo$genoCorr)){
        rsub <- subset(genoInfo$genoCorr[[as.character(c)]], Race == r)
        Sigma <- as.matrix(rsub[, -which(!names(rsub) %in% unique(fsub$refsnp_id))])
        rownames(Sigma) <- colnames(Sigma)
        Sigma <- Sigma[names(probs), names(probs)]
        
        U <- pnorm(rmvnorm(length(ind), sigma = Sigma))
        G <- lapply(1:ncol(U), function(i){
          as.factor(qbinom(U[, i], 2, probs[i]))
        }) %>% bind_cols
      }else{
        G <- rbinom(length(ind), 2, probs)
      }
      genoMat[, m:(m + length(unique(fsub$refsnp_id)) - 1)] <- as.matrix(G)
      m <- m + length(unique(fsub$refsnp_id))
      covnames <- c(covnames, fsub$refsnp_id[seq(2, length(fsub$Frequency), by = 2)])
    }
    genoMat <- data.frame(genoMat)
    names(genoMat) <- covnames
    curr_subset <- cbind(curr_subset, data.frame(genoMat))
    df <- rbind(df, curr_subset)
  }
  return (df)
}

transform_betas <- function(input_vector) {
  transformed_betas <- numeric(length(input_vector) * 2)
  for (i in 1:length(input_vector)) {
    transformed_betas[2 * i - 1] <- input_vector[i]
    transformed_betas[2 * i] <- input_vector[i] * 2
  }
  return(transformed_betas)
}

fetch_parameters <- function(){
  desired_labtests <- list(
    # ====== Liver-function panel (enzymes & plasma proteins) ======================
    AST        = c("AST"),
    ALT        = c("ALT", "ALT plasma"),
    ALP        = c("Alkaline Phosphatase"),
    GGT        = c("GGT"),
    Bilirubin  = c("Bilirubin", "Total Bilirubin"),
    Albumin    = c("Albumin"),
    Globulin   = c("Globulin plasma", "Globulin"),
    Protein    = c("Protein", "Total Protein"),
    
    # ====== Renal / electrolyte + acid–base panel =================================
    Creatinine = c("Creatinine", "Whole Blood Creatinine"),
    Urea       = c("Urea", "Whole Blood Urea"),
    Potassium  = c("Potassium", "Potassium blood", "Potassium whole blood"),
    Sodium     = c("Sodium", "Sodium blood", "Sodium whole blood"),
    Chloride   = c("Chloride blood", "Chloride"),
    Bicarbonate= c("Bicarbonate", "Actual Bicarbonate", "Standard Bicarbonate",
                   "Gases Standard Bicarbonate", "Gases Actual Bicarbonate"),
    Calcium    = c("Calcium", "Adjusted Calcium"),
    Magnesium  = c("Magnesium"),
    Phosphate  = c("Phosphate"),
    
    # ====== Haematology / full-blood-count panel ==================================
    Hb         = c("Hb  - Haemoglobin", "Hb - Haemoglobin",
                   "Blood Haemoglobin", "Haemoglobin"),
    HCT        = c("Hct - Haematocrit", "HCT", "Haematocrit"),
    RBC        = c("RBC", "RBC - Red Cell Count"),
    WBC        = c("WBC - White Cell Count", "WBC"),
    Platelet   = c("Platelet Count", "Platelets"),
    MCV        = c("MCV", "MCV - Mean Cell Volume"),
    RDW        = c("RDW"),
    Neutrophils= c("Neutrophils"),
    Lymphocytes= c("Lymphocytes"),
    Monocytes  = c("Monocytes"),
    Eosinophils= c("Eosinophils"),
    Basophils  = c("Basophils"),
    
    # ====== Lipid panel ===========================================================
    Triglyceride = c("Triglyceride"),
    HDL          = c("HDL Cholesterol", "Cholesterol (HDL)"),
    LDL          = c("LDL cholesterol",
                     "Cholesterol (LDL)(calculated)", "LDL Cholesterol"),
    
    # ====== Glucose / diabetes panel =============================================
    Glucose     = c("Glucose", "Glucose blood", "Glucose whole blood",
                    "Glucose whole Blood"),
    F_Glucose   = c("Fasting Glucose"),
    HbA1c       = c("HbA1c", "Haemoglobin A1c (%)"),
    Insulin     = c("Insulin"),
    
    # ====== Iron-storage marker ===================================================
    Ferritin    = c("Ferritin")
  )
  cohort <- read_sas("cohort_29nov2022.sas7bdat", "formats.sas7bcat")
  cohort <- cohort %>% select(patient_reference_no, gender, 
                              smoking_status, ethnic_gp, age_g, BMI_g, 
                              medication_count_t, SBP_ews, 
                              temperature_ews,
                              HR_ews, O2sat_ews)
  bmi_bounds <- list(`1` = c(15, 18.49), `2` = c(18.5, 24.99),
                     `3` = c(25, 29.99), `4` = c(30, 34.99),
                     `5` = c(35, 39.99), `6` = c(40, 60))
  sbp_bounds <- list(`0` = c(111, 219), `1` = c(101, 110),
                     `2` = c(91, 100), `3` = c(60, 90), 
                     `4` = c(220, 260))
  temp_bounds <- list(`0` = c(36.1, 38.0), `1` = c(35.1, 36.0),
                      `2` = c(38.1, 40.5), `3` = c(32.0, 35.0))
  hr_bounds <- list(`0` = c(51, 90), `1` = c(91, 110),
                    `2` = c(111, 130), `3` = c(30, 40),
                    `4` = c(131, 220))
  spo2_bounds <- list(`0` = c(96, 100), `1` = c(94, 95),
                      `2` = c(92, 93), `3` = c(85, 91))
  bounds_sample = function(code, bounds) {
    if (is.na(code) || !as.character(code) %in% names(bounds)) 
      return(NA_real_)
    rng = bounds[[as.character(code)]]
    runif(1, rng[1], rng[2])
  }
  cohort <- cohort %>% 
    mutate(BMI = map_dbl(BMI_g, ~ bounds_sample(.x, bmi_bounds)),
           SBP = map_dbl(SBP_ews, ~ bounds_sample(.x, sbp_bounds)),
           Temperature = map_dbl(temperature_ews, ~ bounds_sample(.x, temp_bounds)),
           HR = map_dbl(HR_ews, ~ bounds_sample(.x, hr_bounds)),
           SpO2 = map_dbl(O2sat_ews, ~ bounds_sample(.x, spo2_bounds))) %>%
    select(patient_reference_no, gender, 
           smoking_status, ethnic_gp, age_g, BMI, 
           medication_count_t, SBP, 
           Temperature, HR, SpO2) %>%
    group_by(patient_reference_no) %>%
    summarise(across(c(BMI, SBP, Temperature, HR, SpO2), 
                     ~ mean(.x, na.rm = TRUE)), 
              gender = first(gender),
              smoking_status = first(smoking_status),
              ethnic_gp = first(ethnic_gp),
              age_g = first(age_g), 
              medication_count_t = first(medication_count_t),
              .groups = "drop")
  labtest <- read_sas("labtest2.sas7bdat")
  labtest <- labtest %>% 
    select(patient_reference_no, 
           test_result_date, 
           result_test_name, 
           test_result)
  alias_tbl <- imap_dfr(desired_labtests, ~ tibble(alias = .x, canonical = .y)) %>% 
    mutate(alias_lower = str_to_lower(str_squish(alias))) %>%
    distinct(alias_lower, canonical)   
  labtest <- labtest %>%
    mutate(test_result_date = as.Date(test_result_date),
           alias_lower = str_to_lower(str_squish(result_test_name))) %>%
    left_join(alias_tbl, by = "alias_lower") %>%
    filter(result_test_name %in% unlist(desired_labtests)) %>%
    mutate(result_test_name = coalesce(canonical, result_test_name)) %>% 
    select(-alias_lower, -canonical) %>%
    filter(result_test_name != "" & !is.na(result_test_name))
  labtest_wide <- labtest %>%
    pivot_wider(id_cols = c(patient_reference_no, test_result_date),
                names_from = result_test_name,
                values_from = test_result,
                values_fn = list(test_result = ~ first(na.omit(.))),
                values_fill = NA)
  # Eliminate some extreme values, since it's ED dataset
  for (i in names(desired_labtests)){
    labtest_wide[[i]] <- as.numeric(labtest_wide[[i]])
  }
  extreme_range <- with(labtest_wide, list(AST = AST > 1000,
                                           ALT = ALT > 1000,
                                           ALP = ALP > 2000,
                                           GGT = GGT > 1000,
                                           Bilirubin = Bilirubin > 200,
                                           Creatinine = Creatinine > 2000,
                                           Urea = Urea > 50,
                                           WBC = WBC > 100,
                                           Neutrophils = Neutrophils > 50,
                                           Lymphocytes = Lymphocytes > 20,
                                           Monocytes = Monocytes > 10,
                                           Eosinophils = Eosinophils > 10,
                                           Basophils = Basophils > 10,
                                           Triglyceride = Triglyceride > 20,
                                           Glucose = Glucose > 20,
                                           Ferritin = Ferritin > 1000))
  for (i in names(extreme_range)){
    labtest_wide[[i]][extreme_range[[i]]] <- NA
  }
  bindedframe <- labtest_wide %>%
    left_join(cohort %>% distinct(patient_reference_no, .keep_all = TRUE),
              by = "patient_reference_no")
  # Lots of Skewed densities with outliers, retrieve parameters on log scale:
  for (i in c(names(desired_labtests), "SBP", "Temperature", "HR", "SpO2", "BMI")){
    bindedframe[[i]] <- log(bindedframe[[i]] + 1)
  }
  
  # save(bindedframe, file = "joinedData.RData")
  load("joinedData.RData")
  bindedframe$BMI <- log(bindedframe$BMI + 1)
  bindedframe$smoking_status[bindedframe$smoking_status == "Question not asked"] <- 
    sample(c("Never Smoked", "Ex Smoker", "Current Smoker (within 4wks)"), 1)
  bindedframe$ethnic_gp <- as.factor(bindedframe$ethnic_gp)
  bindedframe <- bindedframe %>%
    rename(SEX = gender, 
           SMOKE = smoking_status,
           RACE = ethnic_gp,
           MED_Count = medication_count_t,
           AGE = age_g)
  # Retrieve COV for each group
  ensure_psd <- function(mat, tol = 1e-8) {
    is_psd <- function(mat, tol = 1e-8) {
      min(eigen(mat, symmetric = TRUE)$values) >= -tol
    }
    if (!is_psd(mat)){
      e <- eigen(mat, symmetric = TRUE)
      vals <- pmax(e$values, tol)
      m2 <- e$vectors %*% diag(vals) %*% t(e$vectors)
      return ((m2 + t(m2)) / 2)
    }else{
      return (mat)
    }
  }
  Vital_Sigma <- ensure_psd(cov(as.matrix(subset(bindedframe, select = c("SBP", "Temperature", "HR", "SpO2"))),
                     use = "pairwise.complete.obs"))
  Liver_Sigma <- ensure_psd(cov(as.matrix(subset(bindedframe, select = names(desired_labtests)[1:8])),
                     use = "pairwise.complete.obs"))
  Renal_Sigma <- ensure_psd(cov(as.matrix(subset(bindedframe, select = names(desired_labtests)[9:18])),
                     use = "pairwise.complete.obs"))
  Haematology_Sigma <- ensure_psd(cov(as.matrix(subset(bindedframe, select = names(desired_labtests)[19:30])),
                           use = "pairwise.complete.obs"))
  Lipid_Sigma <- ensure_psd(cov(as.matrix(subset(bindedframe, select = names(desired_labtests)[31:33])),
                     use = "pairwise.complete.obs"))
  Diabete_Sigma <- ensure_psd(cov(as.matrix(subset(bindedframe, select = names(desired_labtests)[34:37])),
                       use = "pairwise.complete.obs"))
  Ferritin_Sigma <- sd(as.matrix(subset(bindedframe, select = names(desired_labtests)[38])), na.rm = T) ^ 2
  BMI_Sigma <- sd(as.matrix(subset(bindedframe, select = "BMI")), na.rm = T) ^ 2
  # # Retrieve Mean for each group
  # Vital_MU <- colMeans(as.matrix(subset(bindedframe, select = c("SBP", "Temperature", "HR", "SpO2"))), na.rm = T)
  # Liver_MU <- colMeans(as.matrix(subset(bindedframe, select = names(desired_labtests)[1:8])), na.rm = T)
  # Renal_MU <- colMeans(as.matrix(subset(bindedframe, select = names(desired_labtests)[9:18])), na.rm = T)
  # Haematology_MU <- colMeans(as.matrix(subset(bindedframe, select = names(desired_labtests)[19:30])), na.rm = T)
  # Lipid_MU <- colMeans(as.matrix(subset(bindedframe, select = names(desired_labtests)[31:33])), na.rm = T)
  # Diabete_MU <- colMeans(as.matrix(subset(bindedframe, select = names(desired_labtests)[34:37])), na.rm = T)
  # Ferritin_MU <- colMeans(as.matrix(subset(bindedframe, select = names(desired_labtests)[38])), na.rm = T)
  # Retrieve Coefficients:
  retrieveCoeffs <- function(data, formula, family = "gaussian") {
    mod <- glm(formula, data = data, family = family)
    sumry <- summary(mod)
    ct <- sumry$coefficients
    
    pcol <- if ("Pr(>|t|)" %in% colnames(ct)) "Pr(>|t|)" else "Pr(>|z|)"
    pvalues <- ct[, pcol]
    estimates <- ct[, "Estimate"]
    
    terms <- attr(terms(mod), "term.labels")
    keep <- character()
    for (t in terms) {
      if (is.factor(data[[t]]) || is.character(data[[t]])) {
        cn <- grep(paste0("^", t), rownames(ct), value = TRUE)
        pv <- pvalues[cn]
        es <- estimates[cn]
        if (any(pv <= 0.05)) {
          keep <- c(keep, t)
          for (lev in cn) {
            if (!(pvalues[lev] <= 0.05)) {
              mod$coefficients[lev] <- 0
            }
          }
        }
      } else {
        if (pvalues[t] <= 0.05) {
          keep <- c(keep, t)
        }
      }
    }
    resp <- deparse(formula[[2]])
    if (length(keep) > 0) {
      newf <- as.formula(paste0(resp, " ~ ", paste(keep, collapse = " + ")))
    } else {
      newf <- as.formula(paste0(resp, " ~ 1"))
    }
    mod2 <- glm(newf, data = data, family = family)
    coefs <- coef(mod2)
    race_idx <- grep("^RACE", names(coefs))
    if (length(race_idx) > 0) {
      coefs[race_idx] <- coefs[race_idx] +
        rnorm(length(race_idx), mean = 0, sd = 0.05) 
      # Added Small Random Noise to Coefficients belong to RACE
      # Masking the True Coefficients, hence protects data privacy.
    }
    list(coeff = coefs, formula = as.character(newf))
  }
  Liver_mod_list <- lapply(names(desired_labtests)[1:8], function(name){
    retrieveCoeffs(bindedframe, as.formula(paste0(name, "~ SEX + SMOKE + RACE + AGE + BMI + MED_Count")))
  })
  names(Liver_mod_list) <- names(desired_labtests)[1:8]
  Renal_mod_list <- lapply(names(desired_labtests)[9:18], function(name){
    retrieveCoeffs(bindedframe, as.formula(paste0(name, "~ SEX + SMOKE + RACE + AGE + BMI + MED_Count")))
  })
  names(Renal_mod_list) <- names(desired_labtests)[9:18]
  Haematology_mod_list <- lapply(names(desired_labtests)[19:30], function(name){
    retrieveCoeffs(bindedframe, as.formula(paste0(name, "~ SEX + SMOKE + RACE + AGE + BMI + MED_Count")))
  })
  names(Haematology_mod_list) <- names(desired_labtests)[19:30]
  Lipid_mod_list <- lapply(names(desired_labtests)[31:33], function(name){
    retrieveCoeffs(bindedframe, as.formula(paste0(name, "~ SEX + SMOKE + RACE + AGE + BMI + MED_Count")))
  })
  names(Lipid_mod_list) <- names(desired_labtests)[31:33]
  Diabete_mod_list <- lapply(names(desired_labtests)[34:37], function(name){
    retrieveCoeffs(bindedframe, as.formula(paste0(name, "~ SEX + SMOKE + RACE + AGE + BMI + MED_Count")))
  })
  names(Diabete_mod_list) <- names(desired_labtests)[34:37]
  Ferritin_mod_list <- lapply(names(desired_labtests)[38], function(name){
    retrieveCoeffs(bindedframe, as.formula(paste0(name, "~ SEX + SMOKE + RACE + AGE + BMI + MED_Count")))
  })
  names(Ferritin_mod_list) <- names(desired_labtests)[38]
  Vital_mod_list <- lapply(c("SBP", "Temperature", "HR", "SpO2"), function(name){
    retrieveCoeffs(bindedframe, as.formula(paste0(name, "~ SEX + SMOKE + RACE + AGE + BMI + MED_Count")))
  })
  names(Vital_mod_list) <- c("SBP", "Temperature", "HR", "SpO2")
  MED_Count_mod_list <- lapply("MED_Count", function(name){
    retrieveCoeffs(bindedframe, as.formula(paste0(name, "~ SEX + SMOKE + RACE + AGE + BMI")), family = "poisson")
  })
  names(MED_Count_mod_list) <- c("MED_Count")
  BMI_mod_list <- lapply("BMI", function(name){
    retrieveCoeffs(bindedframe, as.formula(paste0(name, "~ SEX + SMOKE + RACE + AGE")))
  })
  names(BMI_mod_list) <- c("BMI")
  
  SMOKE_mod <- multinom(SMOKE ~ AGE + SEX + RACE, data = bindedframe)
  SMOKE_mod_list <- list(coeff = as.matrix(coef(SMOKE_mod)), 
                         formula = as.character(as.formula("SMOKE ~ AGE + SEX + RACE")))
  
  Mod_list <- list(Vital = Vital_mod_list,
                   Liver = Liver_mod_list,
                   Renal = Renal_mod_list,
                   Haematology = Haematology_mod_list,
                   Lipid = Lipid_mod_list,
                   Diabete = Diabete_mod_list,
                   Ferritin = Ferritin_mod_list,
                   BMI = BMI_mod_list,
                   SMOKE = SMOKE_mod_list,
                   MED_Count = MED_Count_mod_list)
  Sigma <- list(Vital = Vital_Sigma, 
                Liver = Liver_Sigma,
                Renal = Renal_Sigma, 
                Haematology = Haematology_Sigma,
                Lipid = Lipid_Sigma,
                Diabete = Diabete_Sigma,
                Ferritin = Ferritin_Sigma,
                BMI = BMI_Sigma)
  save(Mod_list, Sigma, file = "./data/labtest_parameters.RData",
       compress = TRUE)
}

####### STARTING SIMULATION.  SAVING FILES ########
if(!dir.exists('./data')){dir.create("./data")}
if(!dir.exists('./data/True')){dir.create("./data/True")}
replicate <- 500
n <- 2e4
if (file.exists("./data/data_generation_seed.RData")){
  load("./data/data_generation_seed.RData")
}else{
  seed <- sample(1:100000, 500)
  save(seed, file = "./data/data_generation_seed.RData")
}

for (i in 1:replicate){
  digit <- stringr::str_pad(i, 4, pad = 0)
  cat("Current:", digit, "\n")
  data <- suppressMessages({generateData(n, seed[i])})
  
  # data$HbA1c_c  <- (data$HbA1c - 53) / 15
  # data$eGFR_c <- (data$eGFR - 60) / 20
  # data$BMI_c <- (data$BMI - 30) / 5
  # data$AGE_c <- (data$AGE - 60) / 15
  # data$SMOKE <- as.character(data$SMOKE)
  # 
  # data$HbA1c_STAR_c  <- (data$HbA1c_STAR - 53) / 15
  # data$eGFR_STAR_c <- (data$eGFR_STAR - 60) / 20
  # data$BMI_STAR_c <- (data$BMI_STAR - 30) / 5
  # data$SMOKE_STAR <- as.character(data$SMOKE_STAR)
  
  # fit.STAR <- coxph(Surv(T_I_STAR, EVENT_STAR) ~ 
  #                     poly(HbA1c_STAR_c, 2, raw = TRUE) + eGFR_STAR_c + BMI_STAR_c +
  #                     rs4506565_STAR + AGE_c + SEX + 
  #                     INSURANCE + RACE + SMOKE_STAR +
  #                     HbA1c_STAR_c:AGE_c,
  #                   data = data)
  
  # fit.TRUE <- coxph(Surv(T_I, EVENT) ~ 
  #                     poly(HbA1c_c, 2, raw = TRUE) + eGFR_c + BMI_c +
  #                     rs4506565 + AGE_c + SEX + 
  #                     INSURANCE + RACE + SMOKE +
  #                     HbA1c_c:AGE_c,
  #                   data = data)
  save(data, file = paste0("./data/True/", digit, ".RData"))
}





