

calibrateFun <- function(samp){
  twophase_des <- twophase(id = list(~1, ~1), strata = list(NULL, ~STRATA), 
                           subset = ~as.logical(R), data = samp)
  modimp.HbA1c <- svyglm(HbA1c ~ HbA1c_STAR + AGE + SEX + RACE + BMI + SMOKE_STAR + SBP +
                           GLUCOSE_STAR + F_GLUCOSE_STAR + INSULIN_STAR + INSURANCE + T_I_STAR, 
                         family = "gaussian", design = twophase_des)
  samp$HbA1c_impute <- as.vector(predict(modimp.HbA1c, newdata = samp, 
                                         type = "response", se.fit = FALSE))
  modimp.SMOKE <- svy_vglm(SMOKE ~ SMOKE_STAR + AGE + SEX + RACE + EXER_STAR + ALC_STAR +
                             BMI + GLUCOSE_STAR + F_GLUCOSE_STAR + HbA1c_STAR + SBP + DBP + PULSE + 
                             HDL + LDL + TG + FERRITIN + INSURANCE + T_I_STAR, 
                           family = "multinomial", design = twophase_des)
  samp$SMOKE_impute <- as.vector(predict(modimp.SMOKE, newdata = samp, 
                                         type = "response", se.fit = FALSE))
  modimp.rs4506565 <- svy_vglm(rs4506565 ~ rs4506565_STAR + T_I_STAR + 
                                 AGE + SEX + RACE + INSURANCE + HbA1c_STAR + 
                                 SMOKE_STAR + EVENT_STAR, 
                               family = "multinomial", design = twophase_des)
  samp$rs4506565_impute <- as.vector(predict(modimp.rs4506565, newdata = samp, 
                                             type = "response", se.fit = FALSE))
  modimp.EVENT <- svyglm(EVENT ~ EVENT_STAR + HbA1c_STAR + rs4506565_STAR + 
                           AGE + SEX + INSURANCE + RACE + BMI + SMOKE_STAR + T_I_STAR, 
                         family = "binomial", design = twophase_des)
  samp$EVENT_impute <- as.vector(predict(modimp.EVENT, newdata = samp, 
                                         type = "response", se.fit = FALSE))
  
  phase1model_imp <- coxph(Surv(T_I_STAR, EVENT_impute) ~ I((HbA1c_impute - 50) / 5) + 
                             rs4506565_impute + I((AGE - 50) / 5) + SEX + INSURANCE + 
                             RACE + I(BMI / 5) + SMOKE_impute, data = samp)
  inffun_imp <- residuals(phase1model_imp, type = "dfbeta")
  colnames(inffun_imp) <- paste0("if", 1:ncol(inffun_imp))
  
  twophase_des_imp <- twophase(id = list(~1, ~1), strata = list(NULL, ~STRATA), 
                               subset = ~as.logical(R), data = cbind(samp, inffun_imp))
  califormu <- make.formula(colnames(inffun_imp)) 
  cali_twophase_imp <- calibrate(twophase_des_imp, califormu, phase = 2, calfun = "raking")
  
  rakingest <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                       rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                       RACE + I(BMI / 5) + EXER, data = samp)
  
  return (rakingest)
}