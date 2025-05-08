#################################### Generate Population ####################################
generateSurvivalData <- function(digit, seed){
  set.seed(seed)
  n <- 2e4
  data <- data.frame(ID = 1:n)
  data$AGE <- runif(n, 35, 85)
  # 0 For 45% Female, 1 For 55% Male
  data$SEX <- rbinom(n, 1, 0.55)
  # 50% Euro, 20% Black, 20% Asian, 10% Other
  data$RACE <- sample(1:4, size = n, replace = T, c(0.5, 0.2, 0.2, 0.1)) 
  # Genotypes
  
  
  # SMOKE ~ AGE + SEX
  design_SMOKE <- model.matrix(~ I(AGE - 40) + SEX, data)
  betas_SMOKE <- rbind(c(-1, 0.01, 0.4),
                       c(-1, 0.02, -0.3))
  lp_SMOKE <- exp(cbind(rep(0, n), design_SMOKE %*% t(betas_SMOKE)))
  probmat_SMOKE <- lp_SMOKE / rowSums(lp_SMOKE)
  cummat_SMOKE <- t(apply(probmat_SMOKE, 1, cumsum))
  data$SMOKE <- max.col(cummat_SMOKE >= runif(n), ties.method = "first")
  # EXER ~ AGE + SEX
  design_EXER <- model.matrix(~ I(AGE - 40) + SEX, data)
  betas_EXER <- rbind(c(-1, -0.01, 0),
                       c(-3, 0, 0.8))
  lp_EXER <- exp(cbind(rep(0, n), design_EXER %*% t(betas_EXER)))
  probmat_EXER <- lp_EXER / rowSums(lp_EXER)
  cummat_EXER <- t(apply(probmat_EXER, 1, cumsum))
  data$EXER <- max.col(cummat_EXER >= runif(n), ties.method = "first")
  # ALC ~ AGE + SEX + SMOKE
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
  # BMI ~ AGE + SEX + EXER + EXER * AGE + G_07
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
  design_SBP <- model.matrix(~ I(AGE - 60) + SEX + I(RACE == 1) + I(RACE == 2) +
                               I(EXER == 2) + I(EXER == 3) + I(SMOKE == 2) + I(ALC == 3) + I(SUBS == 4) + I(BMI - 25), data)
  betas_SBP <- c(130, 0.6, 4, 5, -2, -2, -4, 1, 0.5, 2, 1.2)
  data$SBP <- round(design_SBP %*% betas_SBP + rnorm(n, 0, 0.1))
  # DBP ~ AGE + SEX + RACE + EXER + SMKE + ALC + SUBS + BMI
  design_DBP <- model.matrix(~ I(AGE - 60) + SEX + I(RACE == 1) + I(RACE == 2) +
                               I(EXER == 2) + I(EXER == 3) + I(SMOKE == 2) + I(ALC == 3) + I(SUBS == 4) + I(BMI - 25), data)
  betas_DBP <- c(80, 0.3, 2, 3, -1.5, -1, -2, 0.5, 0.2, 1, 0.7)
  data$DBP <- round(design_DBP %*% betas_DBP + rnorm(n, 0, 0.1))
  # HYPERTENSION
  data$HYPERTENSION <- with(data, SBP >= 140 | DBP >= 90)
  
}


generateGenotypes <- function(N){
  
}

pkgs <- c("AnnotationHub", "VariantAnnotation", "sim1000G",
          "SeqArray", "stringr", "data.table")
rs_info <- data.table(chr = c(11, 3, 1, 6, 
                              3, 8, 3, 9, 
                              10, 11, 12, 17, 
                              12, 3, 3, 11, 
                              1, 19, 9, 22),
                      rsid = c("rs10830963", "rs1801282", "rs340874", "rs7756992", 
                               "rs4607103", "rs13266634", "rs11708067", "rs10811661",
                               "rs7903146", "rs5219", "rs2074356", "rs757210", 
                               "rs11063069", "rs17036101", "rs4402960", "rs1552224",
                               "rs1234315", "rs10401969", "rs17584499", "rs4821480"),
                      pos = c(92975544, 12351626, 213985913, 20679478,
                              64726228, 117172544, 123346931, 22134095,
                              112998590, 17388025, 112207597, 37736525,
                              4265207, 12236345, 185793899, 72722053,
                              173209324, 19296909, 8879118, 36299201))
fetch_region <- function(chr, start_bp, end_bp,
                         dest_dir = "./data/vcf", build = "hg19"){
  dir.create(dest_dir, showWarnings = FALSE)
  url <- sprintf(
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr%d.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz",
    chr
  )
  out_vcf <- file.path(dest_dir,
                       sprintf("chr%d.vcf.bgz", chr))
  print(out_vcf)
  if (!file.exists(out_vcf)) {
    rng   <- GenomicRanges::GRanges(chr, IRanges::IRanges(start_bp, end_bp))
    param <- VariantAnnotation::ScanVcfParam(which = rng)
    
    message(sprintf("➜  downloading chr%s:%s-%s …", chr, start_bp, end_bp))
    vcf <- VariantAnnotation::readVcf(
      Rsamtools::TabixFile(url),
      genome = build,
      param  = param
    )
    VariantAnnotation::writeVcf(vcf, out_vcf, index = TRUE)
  }
  out_vcf
}

windows <- rs_info[, .(
  start = pmax(pos - 5e5, 1),
  end   = pos + 5e5
), by = .(chr)]

vcf_paths <- windows[, fetch_region(chr, start, end), by = 1:nrow(windows)]$V1

unique_chrs <- unique(rs_info$chr)
maps <- setNames(
  lapply(unique_chrs, function(c)
    downloadGeneticMap(c)),
  unique_chrs
)

simulate_chr <- function(vcf_file, map_file, n_ind = 1000) {
  chr <- sub("chr(\\d+).vcf.bgz", "\\1", basename(vcf_file))
  vcf  <- readVCF(vcf_file)
  gmap_df <- fread(map_file[[chr]])
  gmap_df$Chromosome <- gsub("chr", "", gmap_df$Chromosome)
  startSimulation(vcf = vcf,
                  totalNumberOfIndividuals = 2*n_ind,
                  typeOfGeneticMap = "provided")
  ids <- generateUnrelatedIndividuals(N = n_ind)
  geno_mat <- retrieveGenotypes(ids)
  dt <- as.data.table(geno_mat)
  dt[, sample := seq_len(.N)]
  dt
}
genos <- lapply(vcf_paths, simulate_chr, maps, n_ind = 1000)

geno_dt <- Reduce(function(a, b) merge(a, b, by = "sample"), genos)
