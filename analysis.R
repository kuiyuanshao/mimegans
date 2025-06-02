source("00_utils_functions.R")

wdir <- "./simulations/megans/"
attempts <- c("mlp.mlp_32", "mlp.attn_31", "attn.mlp_22", "attn.attn_21", "pmm")

load("/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalData/Output/htn_parameters.RData")
load("/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalData/Output/sbp_parameters.RData")

load("/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalData/RData/TargetPopulationData.RData")

glmci.lm <- NULL
ci.bn <- NULL
lm.df <- NULL
bn.df <- NULL

for (i in 1:58){
  digit <- stringr::str_pad(i, 4, pad = 0)
  cat("Current:", digit)
  load(paste0("/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalData/Output/NutritionalData_", digit, ".RData"))
  true.lm <- glm(sbp ~ c_age + c_bmi + c_ln_na_true + high_chol + usborn +
                   female + bkg_o + bkg_pr, family = gaussian(), data = pop)
  true.bn <- glm(hypertension ~ c_age + c_bmi + c_ln_na_true + high_chol + usborn +
                   female + bkg_o + bkg_pr, family = binomial(), data = pop)
  curr_res.lm <- c(coef(true.lm)[4], sqrt(vcov(true.lm)[4, 4]), "TRUE")
  curr_res.bn <- c(coef(true.bn)[4], sqrt(vcov(true.bn)[4, 4]), "TRUE")
  names(curr_res.lm) <- c("coef", "se", "method")
  names(curr_res.bn) <- c("coef", "se", "method")
  lm.df <- rbind(lm.df, curr_res.lm)
  bn.df <- rbind(bn.df, curr_res.bn)
  
  for (j in attempts){
    load(paste0(wdir, j, "/", digit, ".RData"))
    imp.mids <- as.mids(megans_imp$imputation)
    fit.lm <- with(data = imp.mids, 
                   exp = glm(sbp ~ c_age + c_bmi + c_ln_na_true + high_chol + usborn +
                               female + bkg_pr + bkg_o, family = gaussian()))
    fit.bn <- with(data = imp.mids, 
                   exp = glm(hypertension ~ c_age + c_bmi + c_ln_na_true + high_chol + usborn +
                               female + bkg_pr + bkg_o, family = binomial()))
    pooled.lm <- mice::pool(fit.lm)
    pooled.bn <- mice::pool(fit.bn)
    sumry.lm <- summary(pooled.lm, conf.int = TRUE)
    sumry.bn <- summary(pooled.bn, conf.int = TRUE)
    ci.lm.cov <- calcCICover(sbp_parameters$coefficients$Estimate, sumry.lm$`2.5 %`, sumry.lm$`97.5 %`)
    ci.bn.cov <- calcCICover(htn_parameters$Estimate, sumry.bn$`2.5 %`, sumry.bn$`97.5 %`)
    
    ci.lm <- rbind(ci.lm, c(ci.lm.cov, j))
    ci.bn <- rbind(ci.bn, c(ci.bn.cov, j))
    
    curr_res.lm <- c(sumry.lm$estimate[4], sumry.lm$std.error[4], j)
    curr_res.bn <- c(sumry.bn$estimate[4], sumry.bn$std.error[4], j)
    names(curr_res.lm) <- c("coef", "se", "method")
    names(curr_res.bn) <- c("coef", "se", "method")
    lm.df <- rbind(lm.df, curr_res.lm)
    bn.df <- rbind(bn.df, curr_res.bn)
  }
}
save(lm.df, bn.df, ci.lm, ci.bn, file = "test.RData")
library(ggplot2)
load("test.RData")
ggplot(lm.df) + 
  geom_boxplot(aes(x = method, y = as.numeric(coef))) + 
  geom_hline(aes(yintercept = sbp_parameters$coefficients$Estimate[4]), 
             linetype = "dashed", color = "black")

ggplot(lm.df) + 
  geom_boxplot(aes(x = method, y = as.numeric(se)))

ggplot(bn.df) + 
  geom_boxplot(aes(x = method, y = as.numeric(coef))) +
  geom_hline(aes(yintercept = htn_parameters$Estimate[4]), 
             linetype = "dashed", color = "black")

ggplot(bn.df) + 
  geom_boxplot(aes(x = method, y = as.numeric(se))) + 
  geom_hline(aes(yintercept = sqrt(vcov(true.bn)[4, 4])), 
             linetype = "dashed", color = "black")

ci.lm <- as.data.frame(ci.lm)
ci.bn <- as.data.frame(ci.bn)
library(dplyr)

ci.lm[ci.lm$V10 == "mlp.mlp_32", -10] %>% 
  mutate(across(everything(), as.logical)) %>%
  colMeans()

ci.bn[ci.bn$V10 == "mlp.mlp_32", -10] %>% 
  mutate(across(everything(), as.logical)) %>%
  colMeans()
####
ci.lm[ci.lm$V10 == "mlp.attn_31", -10] %>% 
  mutate(across(everything(), as.logical)) %>%
  colMeans()

ci.bn[ci.bn$V10 == "mlp.attn_31", -10] %>% 
  mutate(across(everything(), as.logical)) %>%
  colMeans()
####
ci.lm[ci.lm$V10 == "pmm", -10] %>% 
  mutate(across(everything(), as.logical)) %>%
  colMeans()

ci.bn[ci.bn$V10 == "pmm", -10] %>% 
  mutate(across(everything(), as.logical)) %>%
  colMeans()

# two variances
# 500 replicates, median of the 500 beta hats, bias = median - truth value.
# beta hats - bias to re centre it.

# 


