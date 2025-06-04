#################################### Generate Population ####################################
# T2D across the whole population
lapply(c("LDlinkR", "dplyr", "purrr", "mvtnorm", "stringr", "tidyr"), require, character.only = T)
generateData <- function(n, seed){
  set.seed(seed)
  n <- n
  data <- data.frame(ID = as.character(1:n))
  # AGE: 18 ~ 90
  data$AGE <- pmin(90, pmax(18, rnorm(n, 50, 15)))
  # SEX: 0 For 45% Female, 1 For 55% Male
  data$SEX <- as.logical(rbinom(n, 1, 0.55))
  # RACE: 50% EUR, 20% AFR, 10% EAS, 10% SAS, 10% AMR
  data$RACE <- sample(c("EUR", "AFR", "EAS", "SAS", "AMR"), 
                      size = n, replace = T, c(0.5, 0.2, 0.1, 0.1, 0.1)) 
  data$RACE <- factor(data$RACE, levels = c("EUR", "AFR", "EAS", "SAS", "AMR"))
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
  data$BMI <- pmax(10, simCovs(data, covarInfo$formu$BMI, covarInfo$betas$BMI, covarInfo$type$BMI, covarInfo$sigma$BMI))
  # WEIGHT:
  data$WEIGHT <- round(data$BMI * (data$HEIGHT / 100)^2, 3)
  # MARRIAGE:
  data$MARRIAGE <- simCovs(data, covarInfo$formu$MARRIAGE, covarInfo$betas$MARRIAGE, covarInfo$type$MARRIAGE)
  # RENAL GROUP: Creatinine, BUN, Uric Acid. 
  data <- cbind(data, apply(simCovs(data, covarInfo$formu$RENAL, covarInfo$betas$RENAL, covarInfo$type$RENAL, covarInfo$sigma$RENAL), 2, exp))
  # LIPIDS: LDL is in inv-norm scale
  data <- cbind(data, simCovs(data, covarInfo$formu$LIPIDS, covarInfo$betas$LIPIDS, covarInfo$type$LIPIDS, covarInfo$sigma$LIPIDS))
  data$HDL <- pmax(0.15, data$HDL)
  data$TG <- pmax(0.3, data$TG)
  # HEMA: HCT is in inv-norm scale
  data <- cbind(data, simCovs(data, covarInfo$formu$HEMA, covarInfo$betas$HEMA, covarInfo$type$HEMA, covarInfo$sigma$HEMA))
  data$WBC <- pmax(0.2, data$WBC)
  data$RBC <- pmax(0.02, data$RBC)
  data$PLATELET <- pmax(1, data$PLATELET)
  data$PT <- pmax(0.75, data$PT)
  # NUTRIENTS:
  data <- cbind(data, simCovs(data, covarInfo$formu$NUTRIENTS, covarInfo$betas$NUTRIENTS, covarInfo$type$NUTRIENTS, covarInfo$sigma$NUTRIENTS))
  # LIVER:
  data <- cbind(data, apply(simCovs(data, covarInfo$formu$LIVER, covarInfo$betas$LIVER, covarInfo$type$LIVER, covarInfo$sigma$LIVER), 2, function(i) exp(i) - 1))
  data$ALT <- pmax(5, data$ALT)
  data$AST <- pmax(3, data$AST)
  data$ALP <- pmax(14, data$ALP)
  data$GGT <- pmax(3, data$GGT)
  data$BILIRUBIN <- pmax(0, data$BILIRUBIN)
  # DIABETE MEASUREMENTS: GLUCOSE & HbA1c are in inv-norm scale
  data <- cbind(data, simCovs(data, covarInfo$formu$DIABETE, 
                              covarInfo$betas$DIABETE, covarInfo$type$DIABETE, covarInfo$sigma$DIABETE))
  data$GLUCOSE <- pmax(4, data$GLUCOSE)
  data$F_GLUCOSE <- pmax(3.5, data$F_GLUCOSE)
  data$INSULIN <- pmax(2, data$INSULIN)
  # PROTEIN:
  data <- cbind(data, simCovs(data, covarInfo$formu$PROTEIN, 
                              covarInfo$betas$PROTEIN, covarInfo$type$PROTEIN, covarInfo$sigma$PROTEIN))
  # INFLAM:
  data <- cbind(data, apply(simCovs(data, covarInfo$formu$INFLAM, 
                                    covarInfo$betas$INFLAM, covarInfo$type$INFLAM, covarInfo$sigma$INFLAM), 2, exp))
  # VITAL SIGNS: SBP, DBP, PULSE
  data <- cbind(data, simCovs(data, covarInfo$formu$VITALS, covarInfo$betas$VITALS, covarInfo$type$VITALS, covarInfo$sigma$VITALS))
  data$PULSE <- round(data$PULSE)
  # HYPENTERSION
  data$HYPERTENSION <- with(data, SBP >= 140 | DBP >= 90)
  # PULSE PRESSURE
  data$PP <- data$SBP - data$DBP
  
  # Adding Measurement Errors:
  # SMOKE:
  SMOKE_M <- matrix(
    c(0.90, 0.04, 0.06, # 90% Current Smoker (right), 4% Never Smoked, 6% Ex-Smoker
      0.01, 0.99, 0.00,
      0.06, 0.02, 0.92),
    nrow = 3, byrow = TRUE,
    dimnames = list(1:3, 
                    1:3)
  )
  data$SMOKE_STAR <- sapply(data$SMOKE, 
                            function(true_val) {sample(as.character(1:3), 
                                                       size = 1, 
                                                       prob = SMOKE_M[true_val, ])})
  # ALC:
  ALC_M <- matrix(
    c(0.85, 0.05, 0.10, # 85% Moderate (right), 5% None, 10% Heavy
      0.15, 0.80, 0.05, 
      0.05, 0.05, 0.90),
    nrow = 3, byrow = TRUE,
    dimnames = list(1:3, 
                    1:3)
  )
  data$ALC_STAR <- sapply(as.character(data$ALC), 
                            function(true_val) {sample(as.character(1:3), 
                                                       size = 1, 
                                                       prob = ALC_M[true_val, ])})
  # EXER:
  EXER_M <- matrix(
    c(0.93, 0.02, 0.05, # 93 % Normal (right), 2% Low, 5% High
      0.10, 0.84, 0.06,
      0.02, 0.00, 0.98),
    nrow = 3, byrow = TRUE,
    dimnames = list(1:3, 
                    1:3)
  )
  data$EXER_STAR <- sapply(as.character(data$EXER), 
                          function(true_val) {sample(as.character(1:3), 
                                                     size = 1, 
                                                     prob = EXER_M[true_val, ])})
  # INCOME:
  INCOME_M <- matrix(
    c(0.93, 0.03, 0.02, 0.01, 0.01,
      0.01, 0.94, 0.04, 0.01, 0,
      0.00, 0.01, 0.96, 0.03, 0,
      0.00, 0.00, 0.00, 0.99, 0.01,
      0.00, 0.00, 0.00, 0.00, 1),
    nrow = 5, byrow = TRUE,
    dimnames = list(1:5, 
                    1:5)
  )
  data$INCOME_STAR <- sapply(as.character(data$INCOME), 
                             function(true_val) {sample(as.character(1:5), 
                                                        size = 1, 
                                                        prob = INCOME_M[true_val, ])})
  # EDU:
  data$EDU_STAR <- data$EDU + sample(-3:3, n, replace = T, 
                                     prob = c(0.02, 0.03, 0.02, 0.8, 0.03, 0.04, 0.06))
  # NUTRIENTS:
  data$Na_INTAKE_STAR <- round(data$Na_INTAKE + rnorm(n, 0, sqrt(0.125)), 3)
  data$K_INTAKE_STAR <- round(data$K_INTAKE + rnorm(n, 0, sqrt(0.09)), 3)
  data$KCAL_INTAKE_STAR <- round(data$KCAL_INTAKE + rnorm(n, 0, sqrt(0.16)), 3)
  data$PROTEIN_INTAKE_STAR <- round(data$PROTEIN_INTAKE + rnorm(n, 0, sqrt(0.055)), 3)
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
  # DIABETES:
  selfReport <- function(true_today, sd_true_past, sd_mis_report,
                         p_mis_report = 0.25){
    past_value <- true_today + rnorm(length(true_today), 0, sd_true_past)
    ind <- sample(length(true_today), p_mis_report * length(true_today))
    past_value[ind] <- past_value[ind] + rnorm(length(ind), 0, sd_mis_report)
    return (round(past_value, 3))
  }
  data$GLUCOSE_STAR <- round(data$GLUCOSE + rnorm(n, 0, 0.5), 3)
  data$F_GLUCOSE_STAR <- selfReport(data$F_GLUCOSE, 0.5, 0.5)
  data$HbA1c_STAR <- selfReport(data$HbA1c, 2, 1)
  data$INSULIN_STAR <- selfReport(data$INSULIN, 2, 1)
  
  # T_I: Self-Reported Time Interval between Treatment Initiation SGLT2 and T2D Diagnosis (Months)
  mm_T_I <- model.matrix(~ I(HbA1c / 10) + rs4506565 + I((AGE - 60) / 5) + SEX + INSURANCE + 
                           RACE + I(BMI / 5) + ALC + SMOKE + EXER, data = data)
  betas_T_I <- log(c(1, 1.15, 1.12, 1.24, 1.08, 1.1, 0.75, 
                     0.90, 0.92, 1, 0.95, 1.1, 0.95, 1.2, 0.85, 0.95, 1.15, 0.9))
  eta_I <- as.vector(mm_T_I %*% betas_T_I)
  k <- 1.2
  lambda <- log(2) / (100 ^ k)
  data$T_I <- (-log(runif(n)) / (lambda*exp(eta_I)))^(1/k) + 6
  
  data$T_I_STAR <- data$T_I
  ind_event <- which(data$T_I < 24)
  FN <- sample(ind_event, round(0.1 * length(ind_event))) # False Negative
  data$T_I_STAR[FN] <- 24
  ind_noevent <- which(data$T_I > 24)
  FP <- sample(ind_noevent, round(0.1 * length(ind_noevent))) # False Positive
  data$T_I_STAR[FP] <- runif(length(FP), 6, 23.99)
  ind_rest <- which(!(c(ind_event, ind_noevent) %in% 1:n))
  ME <- sample(ind_rest, round(0.35 * length(ind_rest))) # Mis-Reported
  data$T_I_STAR[ME] <- data$T_I[ME] + rnorm(length(ME), 0, 1)
  
  data$T_I <- round(pmin(data$T_I, 24), 3)
  data$T_I_STAR <- round(pmin(data$T_I_STAR, 24), 3)
  # EVENT:
  data$EVENT <- data$T_I < 24
  data$EVENT_STAR <- data$T_I_STAR < 24
  
  return (data)
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
      DBP    = "Diastolic blood pressure",
      SBP    = "Systolic blood pressure",
      Med    = c("Medication use (drugs used in diabetes)",
                 "Takes medication for Diabetes/sugar?"),
      GLUCOSE = "Glucose (finger stick, mean, inv-norm transformed)",
      F_GLUCOSE= c("Fasting blood glucose",
                   "Fasting plasma glucose",
                   "Fasting glucose"),
      LDL    = c("low density lipoprotein cholesterol (LDLC, mean, inv-norm transformed)"),
      CRP    = c("C-reactive protein levels",
                 "C-reactive protein levels (MTAG)"),
      HbA1c  = "Hemoglobin A1c (HbA1c, mean, inv-norm transformed)",
      HEIGHT = "Height",
      TG     = "Triglyceride levels",
      ALT    = c("Alanine levels (UKB data field 23460)", "Alanine levels",
                 "Alanine aminotransferase levels"),
      ALP    = "Serum alkaline phosphatase levels",
      ALBUMIN= "Serum albumin levels",
      INSULIN= "Fasting insulin"
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
    # HEIGHT ~ SEX + Genos
    formu_HEIGHT <- as.formula(paste0(" ~ SEX + ", paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "HEIGHT", ]$Query, collapse = " + ")))
    betas_HEIGHT <- c(162, 13, transform_betas(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "HEIGHT", ]$Beta))
    sigma_HEIGHT <- 7
    # SMOKE ~ AGE + SEX + RACE (Current Smoker, Never Smoked, Ex-Smoker)
    formu_SMOKE <- as.formula(~ AGE + SEX + RACE)
    betas_SMOKE <- rbind(
      c(0.27, 0.030, -0.80, -1.36, 1.12, 1.17, -1.38), # Never vs. Current
      c(-1.27, 0.043, -0.23, -0.53, 0.13, 0.20, -0.54) # Ex vs. Current
    )
    # EXER ~ AGE + SEX + RACE + SMOKE (Normal, Low, High)
    formu_EXER <- as.formula(~ AGE + SEX + RACE + SMOKE)
    betas_EXER <- rbind(
      c(0.5, 0.01, -0.2, 0.1, -0.1, 0.05, 0.08, 0.4, 0.2), # Low vs. Normal
      c(-0.5, -0.02, 0.3, -0.1, 0.1, -0.05, -0.08, -0.3, -0.1) # High vs. Normal
    )
    # ALC ~ AGE + SEX + SMOKE + EXER (Moderate, None, Heavy)
    formu_ALC <- as.formula(~ AGE + SEX + RACE + SMOKE + EXER)
    betas_ALC <- rbind(
      c(-1.0, 0.01, -0.5, 0.2, 0.3, 0.1, 0.15, -0.4, -0.2, 0.3, -0.3), # None vs. Moderate
      c(-1.5, -0.02, 0.6, 0.1, -0.2, 0.05, 0.1, 0.5, 0.2, 0.2, -0.4) # Heavy vs. Moderate
    )
    # MARRIAGE ~ AGE + SEX + RACE + BMI (None, Married, Divorced)
    formu_MARRIAGE <- as.formula(~ AGE + SEX + RACE + BMI)
    betas_MARRIAGE <- rbind(
      c(-2.0, 0.05, -0.3, -0.2, 0.1, -0.1, -0.05, -0.02), # Married vs. None
      c(-3.0, 0.03, 0.2, 0.1, -0.1, 0.05, 0.08, 0.01) # Divorced vs. None
    )
    # BMI ~ AGE + SEX + RACE + SMOKE + EXER + ALC + GENOs
    formu_BMI <- as.formula(paste(" ~ AGE + SEX + RACE + SMOKE + EXER + ALC + ",  
                                  paste0(unique(genoInfo$genoTrait$Query[genoInfo$genoTrait$PHENO == "BMI"]), collapse = " + ")))
    betas_BMI <- c(30, 0.05, 0.5, 1.0, -1.0, 0.5, 0.8, -1.0, 0.5, 2.0, -1.5, -0.5, 1.0, 
                   transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "BMI"]))
    sigma_BMI <- 5
    # Diabetes
    formu_DIABETE <- list(GLUCOSE = as.formula(paste("~ AGE + SEX + RACE + BMI + SMOKE + ", 
                                                     paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "GLUCOSE", ]$Query, collapse = " + "))),
                          F_GLUCOSE = as.formula(paste(" ~ AGE + SEX + RACE + BMI + SMOKE + ",
                                                       paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "F_GLUCOSE", ]$Query, collapse = " + "))),
                          HbA1c = as.formula(paste("~ AGE + SEX + RACE + BMI + SMOKE + ", 
                                                   paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "HbA1c", ]$Query, collapse = " + "))),
                          INSULIN = as.formula(paste("~ AGE + SEX + RACE + ",
                                                     paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "INSULIN", ]$Query, collapse = " + "))))
    betas_DIABETE <- list(GLUCOSE = c(15, 0.0275, 0.41152, 0.748, 0.785, 0.429, 1.321, 0.0065, 0, 0.08, 
                                      transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "GLUCOSE"])),  # Glucose mmol/L
                          F_GLUCOSE = c(7, 0.0042, 0.85, 0.92, 2.78, -0.35, 1, 0.054, 0.478, 0.686, 
                                        transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "F_GLUCOSE"])), # Fasting Glucose mmol/L
                          HbA1c = c(62, 0.1441, 2.5621, 11.2332, 5.7360, 0.7964, 6.6452, 0.12236, -1.8026, -0.8387, 
                                    transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "HbA1c"])),  # HbA1c mmol/mol
                          INSULIN = c(120, 0.545, -35, -30, -20, -30, -25, 
                                      transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "INSULIN"])))  # Insulin pmol/L
    sigma_DIABETE <- matrix(c(11.015450747, 2.182595816, 10.171099508,  2.384898863,
                              2.182595816, 0.4324928079,  2.046028358,  0.4598906966,
                              10.171099508, 2.046028358, 37.219300807, -9.254347326,
                              2.384898863, 0.4598906966, -9.254347326,  5.2328446832), byrow = TRUE, nrow = 4)
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
    # Vital Signs
    formu_VITALS <- list(SBP = as.formula(paste(" ~ ", "AGE + SEX + RACE + BMI + SMOKE + EXER + ALC + Na_INTAKE + HbA1c + ",  
                                                paste0(unique(genoInfo$genoTrait$Query[genoInfo$genoTrait$PHENO == "SBP"]), collapse = " + "))), 
                         DBP = as.formula(paste(" ~ ", "AGE + SEX + RACE + BMI + SMOKE + EXER + ALC + Na_INTAKE + HbA1c + ",  
                                                paste0(unique(genoInfo$genoTrait$Query[genoInfo$genoTrait$PHENO == "DBP"]), collapse = " + "))), 
                         PULSE = as.formula(paste(" ~ ", "AGE + SEX + RACE + BMI + SMOKE + EXER + ALC")))
    betas_VITALS <- list(SBP = c(-195, 0.5, 13, -2, -4, -1, 0.5, 0.7, 3, 1, 2, -2, -1, 3, 30, 0.18, 
                                 transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "SBP"])), # SBP (mmHg)
                         DBP = c(-135, 0.3, 10, -1, -3, -0.5, 0.3, 0.3, 2, 0.5, 1, -1, -0.5, 2, 20, 0.11, 
                                 transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "DBP"])), # DBP (mmHg)
                         PULSE = c(70, 0.1, -2, 0, 0, 0, 0, 0.2, 5, 1, 3, -5, 0, 2))   # PULSE (bpm)
    sigma_VITALS <- matrix(c(25, 10, 5,
                             10, 16, 3, 
                             5, 3, 9), nrow = 3)
    # RENAL
    formu_RENAL <- list(CREATININE = as.formula(~ AGE + SEX + RACE + BMI),
                        BUN = as.formula(~ AGE + SEX + RACE + BMI),
                        URIC_ACID = as.formula(~ AGE + SEX + RACE + BMI))
    betas_RENAL <- list(CREATININE = c(3.91, 0.006, 0.27, 0.03, 0.0125, -0.035, 0.02, 0.00085),  # log(Creatinine) μmol/L
                        BUN = c(0.825, 0.015, 0.18, 0.25, 0.11, 0.075, 0.1, 0.001),  # log(Blood Urea Nitrogen) mmol/L
                        URIC_ACID = c(-1.41, 0.0025, 0.15, 0.05, 0, 0, 0.02, 0.005))  # log(Uric Acid) mmol/L
    sigma_RENAL <- matrix(c(0.7288, 0.5345, 0.0677,
                            0.5345, 0.5965, 0.0946,
                            0.0677, 0.0946, 0.1432), byrow = T, nrow = 3)
    # LIPIDS
    formu_LIPIDS <- list(HDL = as.formula(~ AGE + SEX + RACE + BMI + SMOKE),
                         LDL = as.formula(paste("~ AGE + SEX + RACE + BMI + SMOKE + ", 
                                                paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "LDL", ]$Query, collapse = " + "))),
                         TG = as.formula(paste("~ AGE + SEX + RACE + BMI + SMOKE + ", 
                                               paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "LDL", ]$Query, collapse = " + "))))
    betas_LIPIDS <- list(HDL = c(1.5, 0.0006, -0.22, -0.05, -0.07, -0.025, -0.05, -0.005, 0.015, -0.05), # mmol/L
                         LDL = c(1, -0.01, -0.25, -0.35, -0.11, -0.15, -0.35, 0.005, 0.055, 0,
                                 transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "LDL"])), # inv-norm LDL 
                         TG = c(2.3, -0.011, 0.3, 0, 0.55, 0.31, 0, 0.02, -0.4, -0.20, 
                                transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "TG"]))) # mmol/L
    sigma_LIPIDS <- matrix(c(0.1832, 0.0808, -0.0970,
                             0.0808, 0.9949, 0.0925,
                             -0.0970, 0.0925, 1.0460), byrow = TRUE, nrow = 3)
    # HEMATOLOGY
    formu_HEMA <- list(WBC = as.formula(~ AGE + SEX + RACE + BMI),
                       RBC = as.formula(~ AGE + SEX + RACE + BMI),
                       Hb = as.formula(~ AGE + SEX + RACE + BMI),
                       HCT = as.formula(paste("~ AGE + SEX + RACE + BMI + ", 
                                              paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "HCT", ]$Query, collapse = " + "))),
                       PLATELET = as.formula(~ AGE + SEX + RACE + BMI),
                       PT  = as.formula(~ AGE + RACE + BMI))
    betas_HEMA <- list(WBC = c(9.45, -0.011, 0.15, -0.2, -0.225, -0.615, -0.2, 0.008), # * 10^9 cells/L
                       RBC = c(3.50, -0.0055, 0.20, -0.2, -0.018, 0, -0.2, 0.015), # * 10^9 cells/L
                       Hb = c(137.5, -0.25, 10.25, -5, -4.45, -2.55, -3, 0.185), # g/L
                       HCT = c(-0.30, -0.011, 0.5, 0, -0.2, -0.13, -0.2, 0.0115,
                               transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "HCT"])), # inv-norm HCT.
                       PLATELET = c(310, -0.40, -31.5, 17.55, -4.15, -15, 10, -0.125), # 10^3 /mcL
                       PT = c(1.5, 0.0035, -0.0985, -0.135, -0.0595, -0.22, -0.00365)) # seconds
    sigma_HEMA <- matrix(c(28.5981, 0.0899, -0.9600, 0.0499, 103.5507, -0.1987,
                           0.0899, 1.1079, 9.8312, 0.5539, 6.2844, 0.0460,
                           -0.9600, 9.8312, 397.8477, 19.7037,-111.6706, -2.7967,
                           0.0499, 0.5539, 19.7037, 1.0450,  -2.5055, -0.1198,
                           103.5507, 6.2844, -111.6706, -2.5055, 8993.4060, -6.1559,
                           -0.1987, 0.0460, -2.7967, -0.1198, -6.1559, 5.2838), byrow = TRUE, nrow = 6)
    
    # LIVER
    formu_LIVER <- list(ALT = as.formula(paste("~ AGE + SEX + RACE + BMI + ALC + ", 
                                               paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "ALT", ]$Query, collapse = " + "))),
                        AST = as.formula(~ AGE + SEX + RACE + BMI + ALC),
                        ALP = as.formula(paste("~ AGE + SEX + RACE + BMI + ALC + ", 
                                               paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "ALP", ]$Query, collapse = " + "))), 
                        GGT = as.formula(~ AGE + SEX + RACE + BMI + ALC),
                        BILIRUBIN = as.formula(~ AGE + SEX + RACE + BMI + ALC))
    betas_LIVER <- list(ALT = c(3.30, -0.0044, 0.2, -0.03, 0.06, 0.07, 0.05, 0.003, -0.02, 0.05,
                                transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "ALT"])),  #log(x + 1) U/L
                        AST = c(3.4, 0.0016, 0.108, -0.105, 0.05, 0.07, 0.1, -0.002, -0.08, 0.12), #log(x + 1) U/L
                        ALP = c(4.2, 0.0044, 0.045, 0.05, 0.025, -0.025, 0.05, -0.0008, -0.03, 0.06, 
                                transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "ALP"])),   #log(x + 1) U/L
                        GGT = c(2.95, 0.0068, 0.2675, 0.18, 0.14, 0.0097, 0.23, 0.00135, -0.2, 0.3), #log(x + 1) U/L
                        BILIRUBIN = c(1.65, 0.0032, 0.23, 0.1, -0.02, 0.08, 0.035, -0.001, -0.1, 0.1))  #log(x + 1) µmol/L
    sigma_LIVER <- matrix(c(0.7304, 0.5964, 0.1645, 0.5224, 0.2257,
                            0.5964, 0.8465, 0.1636, 0.4986, 0.2857,
                            0.1645, 0.1636, 0.2801, 0.3600, 0.1433,
                            0.5224, 0.4986, 0.3600, 1.1438, 0.3194,
                            0.2257, 0.2857, 0.1433, 0.3194, 0.5523), nrow = 5, byrow = T)
    # PROTEINS
    formu_PROTEIN <- list(ALBUMIN = as.formula(paste("~ AGE + SEX + RACE + GLUCOSE + CREATININE + ALT + ALP + AST + ",
                                                     paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "ALBUMIN", ]$Query, collapse = " + "))),
                          GLOBULIN = as.formula(~ AGE + SEX + RACE + GLUCOSE + CREATININE + ALP))
    betas_PROTEIN <- list(ALBUMIN = c(40, -0.08, 0.84, -1.5, -0.5, 0.5, 0.5, -0.588, -0.0037, 0.002, -0.012, -0.001,
                                      transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "ALBUMIN"])), # g/L
                          GLOBULIN = c(37.5, -0.012, -0.40, 3, 2.75, 1.5, 2, 0.6, 0.004, 0.008)) # g/L
    sigma_PROTEIN <- matrix(c(25.8312, -11.5855,
                              -11.5855, 46.7204), byrow = T, nrow = 2)
    
    # INFLAM
    formu_INFLAM <- list(FERRITIN = as.formula(~ AGE + SEX + RACE + SMOKE + ALT + ALP + AST),
                         CRP = as.formula(paste("~ AGE + SEX + RACE + ALT + ALP + AST + ",
                                                paste0(genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "CRP", ]$Query, collapse = " + "))))
    betas_INFLAM <- list(FERRITIN = c(3.05, 0.012, 0.57, 0.086, -0.026, 0.36, 0.36,
                                      0.23, 0.10, 0.0004, 0.0014, 0.0004),  # log μg/L
                         CRP = c(1.75, 0.0135, 0.15, 0.11, 0, 0.085, 0.2, -0.00035, 0.002, 0.00015,
                                 transform_betas(genoInfo$genoTrait$Beta[genoInfo$genoTrait$PHENO == "CRP"]))) #log mg/L
    sigma_INFLAM <- matrix(c(2.2360, 0.5156,
                             0.5156, 1.5107), nrow = 2, byrow = T)
    
    # PACK UP
    formu <- list(HEIGHT = formu_HEIGHT, SMOKE = formu_SMOKE, EXER = formu_EXER, 
                  ALC = formu_ALC, BMI = formu_BMI, MARRIAGE = formu_MARRIAGE,
                  VITALS = formu_VITALS, RENAL = formu_RENAL, LIPIDS = formu_LIPIDS,
                  HEMA = formu_HEMA, NUTRIENTS = formu_NUTRIENTS, LIVER = formu_LIVER,
                  DIABETE = formu_DIABETE, PROTEIN = formu_PROTEIN, INFLAM = formu_INFLAM)
    betas <- list(HEIGHT = betas_HEIGHT, SMOKE = betas_SMOKE, EXER = betas_EXER, 
                  ALC = betas_ALC, BMI = betas_BMI, MARRIAGE = betas_MARRIAGE,
                  VITALS = betas_VITALS, RENAL = betas_RENAL, LIPIDS = betas_LIPIDS,
                  HEMA = betas_HEMA, NUTRIENTS = betas_NUTRIENTS, LIVER = betas_LIVER,
                  DIABETE = betas_DIABETE, PROTEIN = betas_PROTEIN, INFLAM = betas_INFLAM)
    sigma <- list(HEIGHT = sigma_HEIGHT, BMI = sigma_BMI,
                  VITALS = sigma_VITALS, RENAL = sigma_RENAL, LIPIDS = sigma_LIPIDS,
                  HEMA = sigma_HEMA, NUTRIENTS = sigma_NUTRIENTS, LIVER = sigma_LIVER,
                  DIABETE = sigma_DIABETE, PROTEIN = sigma_PROTEIN, INFLAM = sigma_INFLAM)
    type <- list(HEIGHT = "Norm", SMOKE = "Cat", EXER = "Cat", 
                 ALC = "Cat", BMI = "Norm", MARRIAGE = "Cat",
                 VITALS = "MVNorm", RENAL = "MVNorm", LIPIDS = "MVNorm",
                 HEMA = "MVNorm", NUTRIENTS = "MVNorm", LIVER = "MVNorm",
                 DIABETE = "MVNorm", PROTEIN = "MVNorm", INFLAM = "MVNorm")
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



####### STARTING SIMULATION.  SAVING FILES ########
if(!dir.exists('./data/TRUE')){system('mkdir ./data/TRUE')}
replicate <- 1000
n <- 2e4
seed <- 12
for (i in 12:61){
  digit <- stringr::str_pad(i, 4, pad = 0)
  cat("Current:", digit, "\n")
  data <- generateData(n, seed)
  save(data, file = paste0("./data/TRUE/", digit, ".RData"))
  seed <- seed + 1
}











