#################################### Generate Population ####################################
lapply(c("LDlinkR", "dplyr", "purrr", "mvtnorm", "stringr", "tidyr"), require, character.only = T)
generateSurvivalData <- function(digit, seed){
  set.seed(seed)
  n <- 2e4
  data <- data.frame(ID = 1:n)
  data$AGE <- runif(n, 18, 85)
  # 0 For 45% Female, 1 For 55% Male
  data$SEX <- rbinom(n, 1, 0.55)
  # 50% EUR, 20% AFR, 10% EAS, 10% SAS, 10% AMR
  data$RACE <- sample(c("EUR", "AFR", "EAS", "SAS", "AMR"), 
                      size = n, replace = T, c(0.5, 0.2, 0.1, 0.1, 0.1)) 
  # Genotypes
  genoInfo <- loadGenotypeInfo()
  data <- simGenotypes(data, genoInfo)
  # SMOKE ~ AGE + SEX (Never Smoked, Current Smoker, Ex-Smoker)
  X <- model.matrix(~ I(AGE - 40) + SEX, data)
  beta_full <- rbind(
    c(-1, 0.03, 0.40),
    c(-1.0, 0.01, 0.40),
    c(-0.5, 0.02, -0.30)
  )
  eta <- X %*% t(beta_full)
  P <- exp(eta) / rowSums(exp(eta))
  data$SMOKE<- apply(P, 1, function(pk) sample(
    c("Never","Current","Ex-smoker"), 1, prob = pk
  ))
  # EXER ~ AGE + SEX (Low, Normal, High)
  design_EXER <- model.matrix(~ I(AGE - 40) + SEX, data)
  betas_EXER <- rbind(c(-1, -0.01, 0),
                       c(-3, 0, 0.8))
  lp_EXER <- exp(cbind(rep(0, n), design_EXER %*% t(betas_EXER)))
  probmat_EXER <- lp_EXER / rowSums(lp_EXER)
  cummat_EXER <- t(apply(probmat_EXER, 1, cumsum))
  data$EXER <- max.col(cummat_EXER >= runif(n), ties.method = "first")
  # ALC ~ AGE + SEX + SMOKE (None, Moderate, Heavy)
  design_ALC <- model.matrix(~ I(AGE - 20) + SEX + I(SMOKE == 1) + I(SMOKE == 2), data)
  betas_ALC <- rbind(c(-1, -0.02, 0.15, 0.5, 0.3),
                     c(-2, -0.03, 0.5, 1.0, 0.7))
  lp_ALC <- exp(cbind(rep(0, n), design_ALC %*% t(betas_ALC)))
  probmat_ALC <- lp_ALC / rowSums(lp_ALC)
  cummat_ALC <- t(apply(probmat_ALC, 1, cumsum))
  data$ALC <- max.col(cummat_ALC >= runif(n), ties.method = "first")
  # SUBS ~ AGE + SEX + SMOKE * ALC
  design_SUBS <- model.matrix(~ AGE + SEX + I(SMOKE == 2) + I(SMOKE == 3) +
                                I(ALC == 2) + I(ALC == 3) + I(SMOKE == 2) * I(ALC == 3), data)
  betas_SUBS <- rbind(c(-2, 0.05, 0.1, 0.5, 0.3, 0.4, 0.6, 0),
                      c(-4, 0.01, 0.2, 0.8, 0.5, 0.5, 1.0, 0.7), 
                      c(-5, 0.02, 0.3, 1.5, 0.8, 0.8, 1.5, 1.0))
  lp_SUBS <- exp(cbind(rep(0, n), design_SUBS %*% t(betas_SUBS)))
  probmat_SUBS <- lp_SUBS / rowSums(lp_SUBS)
  cummat_SUBS <- t(apply(probmat_SUBS, 1, cumsum))
  data$SUBS <- max.col(cummat_SUBS >= runif(n), ties.method = "first")
  # BMI ~ AGE + SEX + EXER + EXER * AGE + GENOs
  genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "BMI",]
  design_BMI <- model.matrix(~ I(AGE - 30) + SEX + I(EXER == 1) + I(EXER == 2) +
                               I(EXER == 3) + I(EXER == 3) * I(AGE - 30) + G_07, data)
  betas_BMI <- c(18, 0.3, 0.1, 0.6, 0.1, -0.2, -0.1, 0.2)
  data$BMI <- design_BMI %*% betas_BMI + rnorm(n, 0, 0.05)
  # MARRIAGE ~ AGE + BMI + SMOKE * ALC
  design_MARRIAGE <- model.matrix(~ I(AGE - 30) + BMI + I(SMOKE == 2) * I(ALC == 3), data)
  betas_MARRIAGE <- rbind(c(-2, 0.10, -0.002, 0, 0, 0.2), 
                          c(-5, 0.15, 0.003, 0.001, 0.1, 0.1))
  lp_MARRIAGE <- exp(cbind(rep(0, n), design_MARRIAGE %*% t(betas_MARRIAGE)))
  probmat_MARRIAGE <- lp_MARRIAGE / rowSums(lp_MARRIAGE)
  cummat_MARRIAGE <- t(apply(probmat_MARRIAGE, 1, cumsum))
  data$MARRIAGE <- max.col(cummat_MARRIAGE >= runif(n), ties.method = "first")
  # Simple Measurements
  # SBP ~ AGE + SEX + RACE + EXER + SMOKE + ALC + SUBS + BMI 
  design_SBP <- model.matrix(~ I(AGE - 60) + SEX + I(RACE == "EUR") + I(RACE == "AFR") +
                               I(EXER == 2) + I(EXER == 3) + 
                               I(SMOKE == 2) + I(ALC == 3) + 
                               I(SUBS == 4) + I(BMI - 25), data)
  betas_SBP <- c(130, 0.6, 4, 5, -2, -2, -4, 1, 0.5, 2, 1.2)
  data$SBP <- round(design_SBP %*% betas_SBP + rnorm(n, 0, 0.1))
  # DBP ~ AGE + SEX + RACE + EXER + SMKE + ALC + SUBS + BMI
  design_DBP <- model.matrix(~ I(AGE - 60) + SEX + I(RACE == "EUR") + I(RACE == "AFR") +
                               I(EXER == 2) + I(EXER == 3) + 
                               I(SMOKE == 2) + I(ALC == 3) + 
                               I(SUBS == 4) + I(BMI - 25), data)
  betas_DBP <- c(80, 0.3, 2, 3, -1.5, -1, -2, 0.5, 0.2, 1, 0.7)
  data$DBP <- round(design_DBP %*% betas_DBP + rnorm(n, 0, 0.1))
  # HYPERTENSION
  data$HYPERTENSION <- with(data, SBP >= 140 | DBP >= 90)
  
}

loadGenotypeInfo <- function(){
  rsids <- data.frame(refsnp_id = c("rs10811661", "rs11063069", "rs11708067", "rs17036101", "rs17584499",
                                    "rs340874", "rs4402960", "rs4607103", "rs7754840", "rs9300039",
                                    "rs5015480", "rs9465871", "rs4506565", "rs6718526", "rs358806"),
                      chr = c(9, 12, 3, 3, 9, 
                              1, 3, 3, 6, 11,
                              10, 6, 10, 2, 3))
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
  if (!file.exists("./data/genofreq.RData")){
    genoFreq <- vector("list", 150)
    m <- 1
    for (i in rsids$refsnp_id){
      for (j in race){
        mat <- retrieveFreq(i, j) %>%
          rename(allele = i) %>%
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
  if (!file.exists("./data/genocorr.RData")){
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
    genoTrait <- NULL
    for (j in race){
      trait <- LDtrait(snps = genoInfo$rsids$refsnp_id,
                       pop = j,
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
        DBP    = "Diastolic blood pressure",
        SBP    = "Systolic blood pressure",
        Med    = c("Medication use (drugs used in diabetes)", 
                   "Takes medication for Diabetes/sugar?"),
        FBG    = c("Fasting blood glucose", 
                   "Fasting plasma glucose", 
                   "Fasting glucose"),
        CRP    = c("C-reactive protein levels", 
                   "C-reactive protein levels (MTAG)"),
        HbA1c  = c("Glycated hemoglobin levels",
                   "Hemoglobin A1c (HbA1c, mean, inv-norm transformed)",
                   "Hemoglobin A1c (HbA1c, minimum, inv-norm transformed)",
                   "Hemoglobin A1c (HbA1c, maximum, inv-norm transformed)"),
        HEIGHT = "Height",
        CA     = "Calcium levels",
        TAG    = "Triglyceride levels",
        ALA    = c("Alanine levels (UKB data field 23460)", "Alanine levels"),
        ALT    = "Alanine aminotransferase levels",
        ALP    = "Serum alkaline phosphatase levels",
        ALBUMIN= "Serum albumin levels",
        INS_F  = "Fasting insulin"
      )
      all_mats <- imap(phenos, function(trait_names, pheno_code) {
        trait %>%
          filter(GWAS_Trait %in% trait_names) %>%
          group_by(Query) %>%
          summarise(Beta = mean(Beta, na.rm = TRUE), .groups = "drop") %>%
          mutate(RACE = j, PHENO = pheno_code)
      })
      result <- bind_rows(all_mats)
      genoTrait <- rbind(result, genoTrait)
    }
    save(genoTrait, file = "./data/genoTrait.RData")
  }else{
    load("./data/genoTrait.RData")
  }
  return (list(rsids = rsids, genoFreq = genoFreq, genoCorr = genoCorr, genoTrait = genoTrait))
}

loadFormulaInfo <- function(){
  if (!file.exists("./data/covarInfo.RData")){
    # SMOKE ~ AGE + SEX (Never Smoked, Current Smoker, Ex-Smoker)
    formu_SMOKE <- as.formula(~ I(AGE - 30) + SEX)
    betas_SMOKE <- rbind(c(-1, 0.01, 0.4),
                         c(-1.5, 0.02, 0.25))
    # EXER ~ AGE + SEX (Low, Normal, High)
    formu_EXER <- as.formula(~ I(AGE - 40) + SEX)
    betas_EXER <- rbind(c(-0.5, -0.01, 0),
                        c(-1, -0.03, -0.1))
    # ALC ~ AGE + SEX + SMOKE (None, Moderate, Heavy)
    formu_ALC <- as.formula(~ I(AGE - 20) + SEX + I(SMOKE == 2))
    betas_ALC <- rbind(c(0, 0.02, 0.15, 0.3),
                       c(-0.5, 0.03, 0.3, 0.3))
    # SUBS ~ AGE + SEX + SMOKE + ALC (Never Used, Used Before, Regular Used)
    formu_SUBS <- as.formula(~ I(AGE - 20) + SEX + 
                               I(SMOKE == 2) + I(ALC == 2) + I(ALC == 3))
    betas_SUBS <- rbind(c(-5, 0.05, 0.1, 0.25, 0.4, 0.3),
                        c(-4, 0.01, 0.15, 0.35, 0.4, 0.3))
    # BMI ~ AGE + SEX + EXER + EXER * AGE + GENOs
    genoInfo$genoTrait[genoInfo$genoTrait$PHENO == "BMI",]
    design_BMI <- model.matrix(~ I(AGE - 30) + SEX + I(EXER == 1) + I(EXER == 2) +
                                 I(EXER == 3) + I(EXER == 3) * I(AGE - 30) + G_07, data)
    betas_BMI <- c(18, 0.3, 0.1, 0.6, 0.1, -0.2, -0.1, 0.2)
    data$BMI <- design_BMI %*% betas_BMI + rnorm(n, 0, 0.05)
    # MARRIAGE ~ AGE + BMI + SMOKE * ALC
    design_MARRIAGE <- model.matrix(~ I(AGE - 30) + BMI + I(SMOKE == 2) * I(ALC == 3), data)
    betas_MARRIAGE <- rbind(c(-2, 0.10, -0.002, 0, 0, 0.2), 
                            c(-5, 0.15, 0.003, 0.001, 0.1, 0.1))
    lp_MARRIAGE <- exp(cbind(rep(0, n), design_MARRIAGE %*% t(betas_MARRIAGE)))
    probmat_MARRIAGE <- lp_MARRIAGE / rowSums(lp_MARRIAGE)
    cummat_MARRIAGE <- t(apply(probmat_MARRIAGE, 1, cumsum))
    data$MARRIAGE <- max.col(cummat_MARRIAGE >= runif(n), ties.method = "first")
    # Simple Measurements
    # SBP ~ AGE + SEX + RACE + EXER + SMOKE + ALC + SUBS + BMI 
    design_SBP <- model.matrix(~ I(AGE - 60) + SEX + I(RACE == "EUR") + I(RACE == "AFR") +
                                 I(EXER == 2) + I(EXER == 3) + 
                                 I(SMOKE == 2) + I(ALC == 3) + 
                                 I(SUBS == 4) + I(BMI - 25), data)
    betas_SBP <- c(130, 0.6, 4, 5, -2, -2, -4, 1, 0.5, 2, 1.2)
    data$SBP <- round(design_SBP %*% betas_SBP + rnorm(n, 0, 0.1))
    # DBP ~ AGE + SEX + RACE + EXER + SMKE + ALC + SUBS + BMI
    design_DBP <- model.matrix(~ I(AGE - 60) + SEX + I(RACE == "EUR") + I(RACE == "AFR") +
                                 I(EXER == 2) + I(EXER == 3) + 
                                 I(SMOKE == 2) + I(ALC == 3) + 
                                 I(SUBS == 4) + I(BMI - 25), data)
    betas_DBP <- c(80, 0.3, 2, 3, -1.5, -1, -2, 0.5, 0.2, 1, 0.7)
    data$DBP <- round(design_DBP %*% betas_DBP + rnorm(n, 0, 0.1))
    # HYPERTENSION
    data$HYPERTENSION <- with(data, SBP >= 140 | DBP >= 90)
  }
}

simCatCovs <- function(data, mm, betas){
  lp <- exp(cbind(rep(0, n), mm %*% t(betas)))
  probmat <- lp / rowSums(lp)
  cummat <- t(apply(probmat, 1, cumsum))
  result <- max.col(cummat_EXER >= runif(n), ties.method = "first")
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
          qbinom(U[, i], 2, probs[i])
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
