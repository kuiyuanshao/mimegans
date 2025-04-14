pacman::p_load("survey", "readxl", "stringr", "dplyr", "purrr", "ggplot2", "mice", "progress")

source("./megans/mmer.impute.cwgangp.R")
source("./megans/normalizing.R")
source("./megans/encoding.R")
source("./megans/utils.R")
source("./megans/sample.batches.R")
source("./megans/generate.impute.R")
source("./megans/networks.R")
source("./megans/generators.R")
source("./megans/discriminators.R")

if(!dir.exists('./simulations')){system('mkdir ./simulations')}
if(!dir.exists('./simulations/megans')){system('mkdir ./simulations/megans')}

if(!dir.exists('./simulations/megans/attn_g_1_3')){system('mkdir ./simulations/megans/attn_g_1_3')}

for (i in 1:20){
  cat("Iteration:", i, "\n")
  digit <- str_pad(i, nchar(4444), pad=0)
  data_nut <- read.csv(paste0("./data/SRS_", digit, ".csv"))
  data_nut$X <- NULL
  data_nut$id <- NULL
  data_nut$R <- NULL
  data_info = list(phase1_vars = c("c_ln_na_bio1", "c_ln_k_bio1", "c_ln_kcal_bio1", "c_ln_protein_bio1"), 
                   phase2_vars = c("c_ln_na_true", "c_ln_k_true", "c_ln_kcal_true", "c_ln_protein_true"), 
                   weight_var = "W",
                   cat_vars = c("hypertension", "bkg_pr", "bkg_o", "female", 
                                "high_chol", "usborn", "idx"),
                   num_vars = names(data_nut)[!names(data_nut) %in% c("W", "hypertension", 
                                                                      "bkg_pr", "bkg_o", "female", 
                                                                      "high_chol", "usborn", "idx")])
  
  megans_imp.attn <- mmer.impute.cwgangp(data_nut, m = 5, num.normalizing = "mode", cat.encoding = "onehot", 
                                         device = "cpu", epochs = 10000, 
                                         params = list(gamma = 1, scaling = 1, n_g_layers = 1, 
                                                       n_d_layers = 1, pac = 5, 
                                                       type_g = "attn", type_d = "attn"), 
                                         data_info = data_info, save.step = 1000)
  save(megans_imp.attn, file = paste0("./simulations/megans/attn_g_1_3/", digit, ".RData"))
}

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
  
  var.1 <- 1/length(imp) * colSums(m_vars.1) +  (length(imp) + 1) * apply(m_coefs.1, 2, var) / length(imp)
  #var.1.2 <- (length(imp) + 1) * apply(m_coefs.1, 2, var) / length(imp)
  var.2 <- 1/length(imp) * colSums(m_vars.2) + (length(imp) + 1) * apply(m_coefs.2, 2, var) / length(imp)
  #var.2.2 <- (length(imp) + 1) * apply(m_coefs.2, 2, var) / length(imp)
  return (list(coef = list(colMeans(m_coefs.1), colMeans(m_coefs.2)), var = list(var.1, var.2)))
}

n <- 20
result_df.1 <- vector("list", n)
result_df.2 <- vector("list", n)
m <- 1

for (i in 1:n){
  digit <- str_pad(i, nchar(4444), pad=0)
  
  if (!file.exists(paste0("./simulations/megans/mlp_without_pac/", digit, ".RData"))){
    next
  }
  
  load(paste0("/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalData/Output/NutritionalData_", digit, ".RData"))
  curr_sample <- read.csv(paste0("/nesi/project/uoa03789/PhD/SamplingDesigns/NutritionalData/NutritionalSample/SRS/SRS_", digit, ".csv"))
  
  
  true.1 <- glm(hypertension ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn +
                  female + bkg_o + bkg_pr, family = binomial(), data = pop)
  true.2 <- glm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn +
                  female + bkg_o + bkg_pr, family = gaussian(), data = pop)
  complete.1 <- glm(hypertension ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn +
                      female + bkg_o + bkg_pr, family = binomial(), data = curr_sample)
  complete.2 <- glm(sbp ~ c_ln_na_true + c_age + c_bmi + high_chol + usborn +
                      female + bkg_o + bkg_pr, family = gaussian(), data = curr_sample)
  
  load(paste0("./simulations/megans/mlp_with_pac/", digit, ".RData"))
  imp_coefs_vars.gans.pac <- find_coef_var(imp = megans_imp.mlp$step_result[[1]])
  load(paste0("./simulations/megans/mlp_without_pac/", digit, ".RData"))
  imp_coefs_vars.gans.unpac <- find_coef_var(imp = megans_imp.mlp$step_result[[1]])
  
  load(paste0("./simulations/megans/attn_3_1/", digit, ".RData"))
  imp_coefs_vars.gans.attn_31 <- find_coef_var(imp = megans_imp.attn$step_result[[1]])
  load(paste0("./simulations/megans/attn_5_3/", digit, ".RData"))
  imp_coefs_vars.gans.attn_53 <- find_coef_var(imp = megans_imp.attn$step_result[[1]])
  
  curr_res.1 <- data.frame(TRUE.Est = coef(true.1),
                           COMPL.Est = coef(complete.1),
                           GANS_PAC.imp.Est = imp_coefs_vars.gans.pac$coef[[1]],
                           GANS_UNPAC.imp.Est = imp_coefs_vars.gans.unpac$coef[[1]],
                           GANS_ATTN31.imp.Est = imp_coefs_vars.gans.attn_31$coef[[1]],
                           GANS_ATTN53.imp.Est = imp_coefs_vars.gans.attn_53$coef[[1]],
                           
                           TRUE.Var = diag(vcov(true.1)),
                           COMPL.Var = diag(vcov(complete.1)),
                           GANS_PAC.imp.Var = imp_coefs_vars.gans.pac$var[[1]],
                           GANS_UNPAC.imp.Var = imp_coefs_vars.gans.unpac$var[[1]],
                           GANS_ATTN31.imp.Var = imp_coefs_vars.gans.attn_31$var[[1]],
                           GANS_ATTN53.imp.Var = imp_coefs_vars.gans.attn_53$var[[1]],
                           
                           DIGIT = digit)
  
  curr_res.2 <- data.frame(TRUE.Est = coef(true.2),
                           COMPL.Est = coef(complete.2),
                           GANS_PAC.imp.Est = imp_coefs_vars.gans.pac$coef[[2]],
                           GANS_UNPAC.imp.Est = imp_coefs_vars.gans.unpac$coef[[2]],
                           GANS_ATTN31.imp.Est = imp_coefs_vars.gans.attn_31$coef[[2]],
                           GANS_ATTN53.imp.Est = imp_coefs_vars.gans.attn_53$coef[[2]],
                           
                           TRUE.Var = diag(vcov(true.2)),
                           COMPL.Var = diag(vcov(complete.2)),
                           GANS_PAC.imp.Var = imp_coefs_vars.gans.pac$var[[2]],
                           GANS_UNPAC.imp.Var = imp_coefs_vars.gans.unpac$var[[2]],
                           GANS_ATTN31.imp.Var = imp_coefs_vars.gans.attn_31$var[[2]],
                           GANS_ATTN53.imp.Var = imp_coefs_vars.gans.attn_53$var[[2]],
                           
                           DIGIT = digit)
  result_df.1[[m]] <- curr_res.1
  result_df.2[[m]] <- curr_res.2
  m <- m + 1
}

pacman::p_load("ggplot2", "tidyr", "dplyr", "RColorBrewer", "ggh4x")
#### MODEL-BASED RESULTS
combined_df.1 <- bind_rows(result_df.1) %>%
  filter(grepl("^c_ln_na_true", rownames(.))) %>%
  pivot_longer(
    cols = 1:12,
    names_to = c("METHOD", "TYPE"),
    names_pattern = "^(.*)\\.(Est|Var)$"
  )

combined_df.2 <- bind_rows(result_df.2) %>%
  filter(grepl("^c_ln_na_true", rownames(.))) %>%
  pivot_longer(
    cols = 1:12,
    names_to = c("METHOD", "TYPE"),
    names_pattern = "^(.*)\\.(Est|Var)$"
  )

means.1 <- combined_df.1 %>%
  dplyr::filter(METHOD == "TRUE") %>%
  aggregate(value ~ TYPE, data = ., FUN = mean)

means.2 <- combined_df.2 %>%
  dplyr::filter(METHOD == "TRUE") %>%
  aggregate(value ~ TYPE, data = ., FUN = mean)




ggplot(combined_df.1) +
  geom_boxplot(aes(x = factor(METHOD, levels = c("TRUE", "COMPL", "GANS_PAC.imp", "GANS_UNPAC.imp", "GANS_ATTN31.imp", "GANS_ATTN53.imp")),
                   y = value)) +
  geom_hline(data = means.1, aes(yintercept = value), linetype = "dashed", color = "black") +
  facet_wrap(~TYPE, scales = "free", ncol = 1,
             labeller = labeller(TYPE = c(Est = "Coefficient", Var = "Variance"))) +
  theme_minimal() +
  labs(x = "Methods", y = "Estimate", colour = "Sampling Designs") +
  theme(axis.title.x = element_text(family = "Georgia"),
        axis.title.y = element_text(family = "Georgia"),
        axis.text.x = element_text(family = "Georgia"),
        axis.text.y = element_text(family = "Georgia"),
        legend.title = element_text(family = "Georgia"),
        legend.text = element_text(family = "Georgia"),
        strip.text = element_text(family = "Georgia")) +
  facetted_pos_scales(y = list(TYPE == "Est" ~ scale_y_continuous(limits = c(0, 2.5)),
                               TYPE == "Var" ~ scale_y_continuous(limits = c(0, 0.09))))

ggsave("Imputation_logistic_boxplot.png", width = 10, height = 10, limitsize = F)

ggplot(combined_df.2) +
  geom_boxplot(aes(x = factor(METHOD, levels = c("TRUE", "COMPL", "GANS_PAC.imp", "GANS_UNPAC.imp", "GANS_ATTN31.imp", "GANS_ATTN53.imp")),
                   y = value)) +
  geom_hline(data = means.2, aes(yintercept = value), linetype = "dashed", color = "black") +
  facet_wrap(~TYPE, scales = "free", ncol = 1,
             labeller = labeller(TYPE = c(Est = "Coefficient", Var = "Variance"))) +
  theme_minimal() +
  labs(x = "Methods", y = "Estimate", colour = "Sampling Designs") +
  theme(axis.title.x = element_text(family = "Georgia"),
        axis.title.y = element_text(family = "Georgia"),
        axis.text.x = element_text(family = "Georgia"),
        axis.text.y = element_text(family = "Georgia"),
        legend.title = element_text(family = "Georgia"),
        legend.text = element_text(family = "Georgia"),
        strip.text = element_text(family = "Georgia")) +
  facetted_pos_scales(y = list(TYPE == "Est" ~ scale_y_continuous(limits = c(23, 35)),
                               TYPE == "Var" ~ scale_y_continuous(limits = c(0, 1))))

ggsave("Imputation_gaussian_boxplot.png", width = 10, height = 10, limitsize = F)
