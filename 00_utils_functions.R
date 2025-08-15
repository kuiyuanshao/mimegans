expit <- function(x){
  exp(x) / (1 + exp(x))
}

calcCICover <- function(true, lower, upper){
  return (true >= lower) & (true <= upper)
}

as.mids <- function(imp_list){
  imp_mids <- miceadds::datlist2mids(imp_list)
  return (imp_mids)
}


exactAllocation <- function(data, stratum_variable, target_variable, sample_size){
  strata_units <- as.data.frame(table(data[[stratum_variable]]))
  colnames(strata_units) <- c(stratum_variable, "count")
  conversion_functions <- list(
    numeric = "as.numeric",
    integer = "as.integer",
    character = "as.character",
    logical = "as.logical",
    factor = "as.factor"
  )
  strata_units[, 1] <-  do.call(conversion_functions[[class(data[[stratum_variable]])[1]]], list(strata_units[, 1]))
  
  data <- merge(data, strata_units, by = stratum_variable)
  Y_bars <- aggregate(as.formula(paste0(target_variable, " ~ ", stratum_variable)), data = data, FUN = function(x) sum(x) / length(x))
  colnames(Y_bars)[2] <- "Y_bars"
  data <- merge(data, Y_bars, by = stratum_variable)
  Ss <- aggregate(as.formula(paste0("(", target_variable, " - Y_bars", ")^2", " ~ ", stratum_variable)), 
                  data = data, FUN = function(x) sum(x) / (length(x) - 1))
  
  NS <- strata_units$count * sqrt(Ss[, 2])
  names(NS) <- Ss[, 1]
  NS <- NS[order(NS, decreasing = T)]
  # Type-II
  columns <- sample_size - 2 * nrow(Ss)
  priority <- matrix(0, nrow = columns, ncol = nrow(Ss))
  colnames(priority) <- names(NS)
  for (h in names(NS)){
    priority[, h] <- NS[[h]] / sqrt((2:(columns + 1)) * (3:(columns + 2)))
  }
  priority <- as.data.frame(priority)
  priority <- stack(priority)
  colnames(priority) <- c("value", stratum_variable)
  order_priority <- order(priority$value, decreasing = T)
  alloc <- (table(priority[[stratum_variable]][order_priority[1:columns]]) + 2)
  alloc <- alloc[names(table(data[[stratum_variable]]))]
  return (alloc)
}

match_types <- function(new_df, orig_df) {
  common <- intersect(names(orig_df), names(new_df))
  out <- new_df
  
  for (nm in common) {
    tmpl <- orig_df[[nm]]
    col <- out[[nm]]
    if (is.integer(tmpl))        out[[nm]] <- as.integer(col)
    else if (is.numeric(tmpl))   out[[nm]] <- as.numeric(col)
    else if (is.logical(tmpl))   out[[nm]] <- as.logical(col)
    else if (is.factor(tmpl)) {
      out[[nm]] <- factor(col,
                          levels = levels(tmpl),
                          ordered = is.ordered(tmpl))
    }
    else if (inherits(tmpl, "Date")) {
      out[[nm]] <- as.Date(col)
    } else if (inherits(tmpl, "POSIXct")) {
      tz <- attr(tmpl, "tzone")
      out[[nm]] <- as.POSIXct(col, tz = tz)
    }
    else {
      out[[nm]] <- as.character(col)
    }
  }
  out
}

reallocate <- function(samp){
  parts <- do.call(rbind, strsplit(as.character(samp$STRATA), "\\."))
  flag <- parts[,1]
  lvl1 <- as.integer(parts[,2])
  lvl2 <- as.integer(parts[,3])
  
  cnt   <- table(samp$STRATA)
  single <- names(cnt)[cnt == 1]       # lone strata
  multi <- names(cnt)[cnt > 1]        # viable targets
  
  nearest <- function(lbl){
    i <- which(samp$STRATA == lbl)[1]
    f <- flag[i]; x <- lvl1[i]; y <- lvl2[i]
    idx <- match(multi, samp$STRATA)
    cand <- multi[flag[idx] == f]    # same TRUE/FALSE block
    d <- abs(lvl1[idx[flag[idx]==f]] - x) + abs(lvl2[idx[flag[idx]==f]] - y)
    cand[which.min(d)]
  }
  
  map <- setNames(multi, multi)
  for(lbl in single) map[lbl] <- nearest(lbl)
  
  samp$STRATA <- map[samp$STRATA]
  samp$fpc <- cnt[samp$STRATA]
  
  return (samp)
}

data_info_srs <- list(weight_var = "W",
                      cat_vars = c("SEX", "RACE", "SMOKE", "EXER", "ALC", "INSURANCE", "REGION", 
                                   "URBAN", "INCOME", "MARRIAGE", 
                                   "rs10811661", "rs7756992", "rs11708067", "rs17036101", "rs17584499",
                                   "rs1111875", "rs4402960", "rs4607103", "rs7754840", "rs9300039",
                                   "rs5015480", "rs9465871", "rs4506565", "rs5219", "rs358806", 
                                   "HYPERTENSION", 
                                   "SMOKE_STAR", "ALC_STAR", "EXER_STAR", "INCOME_STAR",
                                   "rs10811661_STAR", "rs7756992_STAR", "rs11708067_STAR", "rs17036101_STAR", "rs17584499_STAR",
                                   "rs1111875_STAR", "rs4402960_STAR", "rs4607103_STAR", "rs7754840_STAR", "rs9300039_STAR",
                                   "rs5015480_STAR", "rs9465871_STAR", "rs4506565_STAR", "rs5219_STAR", "rs358806_STAR",
                                   "EVENT", "EVENT_STAR", "R"),
                      num_vars = c("X", "ID", "AGE", "EDU", "HEIGHT", "BMI", "MED_Count",
                                   "Creatinine", "eGFR", "Urea", "Potassium", "Sodium",
                                   "Chloride", "Bicarbonate", "Calcium", "Magnesium", "Phosphate",
                                   "Triglyceride", "HDL", "LDL", "Hb", "HCT",
                                   "RBC", "WBC", "Platelet", "MCV", "RDW",
                                   "Neutrophils", "Lymphocytes", "Monocytes", "Eosinophils", "Basophils",
                                   "Na_INTAKE", "K_INTAKE", "KCAL_INTAKE", "PROTEIN_INTAKE", "AST",
                                   "ALT", "ALP", "GGT", "Bilirubin", "Albumin",
                                   "Globulin", "Protein", "Glucose", "F_Glucose", "HbA1c",
                                   "Insulin", "Ferritin", "SBP", "Temperature", "HR",
                                   "SpO2", "WEIGHT", "EDU_STAR", "Na_INTAKE_STAR", "K_INTAKE_STAR",
                                   "KCAL_INTAKE_STAR", "PROTEIN_INTAKE_STAR", "Glucose_STAR", "F_Glucose_STAR", "HbA1c_STAR",
                                   "C", "C_STAR", "T_I", "T_I_STAR"),
                      phase2_vars = c("SMOKE", "ALC", "EXER", "INCOME", "EDU", 
                                      "Na_INTAKE", "K_INTAKE", "KCAL_INTAKE", "PROTEIN_INTAKE", 
                                      "rs10811661", "rs7756992", "rs11708067", "rs17036101", "rs17584499",
                                      "rs1111875", "rs4402960", "rs4607103", "rs7754840", "rs9300039",
                                      "rs5015480", "rs9465871", "rs4506565", "rs5219", "rs358806",
                                      "Glucose", "F_Glucose", "HbA1c", "T_I", "EVENT", "C"),
                      phase1_vars = c("SMOKE_STAR", "ALC_STAR", "EXER_STAR", "INCOME_STAR", "EDU_STAR", 
                                      "Na_INTAKE_STAR", "K_INTAKE_STAR", "KCAL_INTAKE_STAR", "PROTEIN_INTAKE_STAR", 
                                      "rs10811661_STAR", "rs7756992_STAR", "rs11708067_STAR", "rs17036101_STAR", "rs17584499_STAR",
                                      "rs1111875_STAR", "rs4402960_STAR", "rs4607103_STAR", "rs7754840_STAR", "rs9300039_STAR",
                                      "rs5015480_STAR", "rs9465871_STAR", "rs4506565_STAR", "rs5219_STAR", "rs358806_STAR",
                                      "Glucose_STAR", "F_Glucose_STAR", "HbA1c_STAR", "T_I_STAR", "EVENT_STAR", "C_STAR"))

data_info_balance <- list(weight_var = "W",
                          cat_vars = c("SEX", "RACE", "SMOKE", "EXER", "ALC", "INSURANCE", "REGION", 
                                       "URBAN", "INCOME", "MARRIAGE", 
                                       "rs10811661", "rs7756992", "rs11708067", "rs17036101", "rs17584499",
                                       "rs1111875", "rs4402960", "rs4607103", "rs7754840", "rs9300039",
                                       "rs5015480", "rs9465871", "rs4506565", "rs5219", "rs358806", 
                                       "HYPERTENSION", 
                                       "SMOKE_STAR", "ALC_STAR", "EXER_STAR", "INCOME_STAR",
                                       "rs10811661_STAR", "rs7756992_STAR", "rs11708067_STAR", "rs17036101_STAR", "rs17584499_STAR",
                                       "rs1111875_STAR", "rs4402960_STAR", "rs4607103_STAR", "rs7754840_STAR", "rs9300039_STAR",
                                       "rs5015480_STAR", "rs9465871_STAR", "rs4506565_STAR", "rs5219_STAR", "rs358806_STAR",
                                       "EVENT", "EVENT_STAR", "STRATA", "R"),
                          num_vars = c("X", "ID", "AGE", "EDU", "HEIGHT", "BMI", "MED_Count",
                                       "Creatinine", "eGFR", "Urea", "Potassium", "Sodium",
                                       "Chloride", "Bicarbonate", "Calcium", "Magnesium", "Phosphate",
                                       "Triglyceride", "HDL", "LDL", "Hb", "HCT",
                                       "RBC", "WBC", "Platelet", "MCV", "RDW",
                                       "Neutrophils", "Lymphocytes", "Monocytes", "Eosinophils", "Basophils",
                                       "Na_INTAKE", "K_INTAKE", "KCAL_INTAKE", "PROTEIN_INTAKE", "AST",
                                       "ALT", "ALP", "GGT", "Bilirubin", "Albumin",
                                       "Globulin", "Protein", "Glucose", "F_Glucose", "HbA1c",
                                       "Insulin", "Ferritin", "SBP", "Temperature", "HR",
                                       "SpO2", "WEIGHT", "EDU_STAR", "Na_INTAKE_STAR", "K_INTAKE_STAR",
                                       "KCAL_INTAKE_STAR", "PROTEIN_INTAKE_STAR", "Glucose_STAR", "F_Glucose_STAR", "HbA1c_STAR",
                                       "C", "C_STAR", "T_I", "T_I_STAR"),
                          phase2_vars = c("SMOKE", "ALC", "EXER", "INCOME", "EDU", 
                                          "Na_INTAKE", "K_INTAKE", "KCAL_INTAKE", "PROTEIN_INTAKE", 
                                          "rs10811661", "rs7756992", "rs11708067", "rs17036101", "rs17584499",
                                          "rs1111875", "rs4402960", "rs4607103", "rs7754840", "rs9300039",
                                          "rs5015480", "rs9465871", "rs4506565", "rs5219", "rs358806",
                                          "Glucose", "F_Glucose", "HbA1c", "T_I", "EVENT", "C"),
                          phase1_vars = c("SMOKE_STAR", "ALC_STAR", "EXER_STAR", "INCOME_STAR", "EDU_STAR", 
                                          "Na_INTAKE_STAR", "K_INTAKE_STAR", "KCAL_INTAKE_STAR", "PROTEIN_INTAKE_STAR", 
                                          "rs10811661_STAR", "rs7756992_STAR", "rs11708067_STAR", "rs17036101_STAR", "rs17584499_STAR",
                                          "rs1111875_STAR", "rs4402960_STAR", "rs4607103_STAR", "rs7754840_STAR", "rs9300039_STAR",
                                          "rs5015480_STAR", "rs9465871_STAR", "rs4506565_STAR", "rs5219_STAR", "rs358806_STAR",
                                          "Glucose_STAR", "F_Glucose_STAR", "HbA1c_STAR", "T_I_STAR", "EVENT_STAR", "C_STAR"))

data_info_neyman <- list(weight_var = "W",
                         cat_vars = c("SEX", "RACE", "SMOKE", "EXER", "ALC", "INSURANCE", "REGION", 
                                      "URBAN", "INCOME", "MARRIAGE", 
                                      "rs10811661", "rs7756992", "rs11708067", "rs17036101", "rs17584499",
                                      "rs1111875", "rs4402960", "rs4607103", "rs7754840", "rs9300039",
                                      "rs5015480", "rs9465871", "rs4506565", "rs5219", "rs358806", 
                                      "HYPERTENSION", 
                                      "SMOKE_STAR", "ALC_STAR", "EXER_STAR", "INCOME_STAR",
                                      "rs10811661_STAR", "rs7756992_STAR", "rs11708067_STAR", "rs17036101_STAR", "rs17584499_STAR",
                                      "rs1111875_STAR", "rs4402960_STAR", "rs4607103_STAR", "rs7754840_STAR", "rs9300039_STAR",
                                      "rs5015480_STAR", "rs9465871_STAR", "rs4506565_STAR", "rs5219_STAR", "rs358806_STAR",
                                      "EVENT", "EVENT_STAR", "STRATA", "R"),
                         num_vars = c("X", "ID", "AGE", "EDU", "HEIGHT", "BMI", "MED_Count",
                                      "Creatinine", "eGFR", "Urea", "Potassium", "Sodium",
                                      "Chloride", "Bicarbonate", "Calcium", "Magnesium", "Phosphate",
                                      "Triglyceride", "HDL", "LDL", "Hb", "HCT",
                                      "RBC", "WBC", "Platelet", "MCV", "RDW",
                                      "Neutrophils", "Lymphocytes", "Monocytes", "Eosinophils", "Basophils",
                                      "Na_INTAKE", "K_INTAKE", "KCAL_INTAKE", "PROTEIN_INTAKE", "AST",
                                      "ALT", "ALP", "GGT", "Bilirubin", "Albumin",
                                      "Globulin", "Protein", "Glucose", "F_Glucose", "HbA1c",
                                      "Insulin", "Ferritin", "SBP", "Temperature", "HR",
                                      "SpO2", "WEIGHT", "EDU_STAR", "Na_INTAKE_STAR", "K_INTAKE_STAR",
                                      "KCAL_INTAKE_STAR", "PROTEIN_INTAKE_STAR", "Glucose_STAR", "F_Glucose_STAR", "HbA1c_STAR",
                                      "C", "C_STAR", "T_I", "T_I_STAR"),
                         phase2_vars = c("SMOKE", "ALC", "EXER", "INCOME", "EDU", 
                                         "Na_INTAKE", "K_INTAKE", "KCAL_INTAKE", "PROTEIN_INTAKE", 
                                         "rs10811661", "rs7756992", "rs11708067", "rs17036101", "rs17584499",
                                         "rs1111875", "rs4402960", "rs4607103", "rs7754840", "rs9300039",
                                         "rs5015480", "rs9465871", "rs4506565", "rs5219", "rs358806",
                                         "Glucose", "F_Glucose", "HbA1c", "T_I", "EVENT", "C"),
                         phase1_vars = c("SMOKE_STAR", "ALC_STAR", "EXER_STAR", "INCOME_STAR", "EDU_STAR", 
                                         "Na_INTAKE_STAR", "K_INTAKE_STAR", "KCAL_INTAKE_STAR", "PROTEIN_INTAKE_STAR", 
                                         "rs10811661_STAR", "rs7756992_STAR", "rs11708067_STAR", "rs17036101_STAR", "rs17584499_STAR",
                                         "rs1111875_STAR", "rs4402960_STAR", "rs4607103_STAR", "rs7754840_STAR", "rs9300039_STAR",
                                         "rs5015480_STAR", "rs9465871_STAR", "rs4506565_STAR", "rs5219_STAR", "rs358806_STAR",
                                         "Glucose_STAR", "F_Glucose_STAR", "HbA1c_STAR", "T_I_STAR", "EVENT_STAR", "C_STAR"))