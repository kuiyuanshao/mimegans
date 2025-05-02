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

if(!dir.exists('./simulations/megans/mlp.mlp_32')){system('mkdir ./simulations/megans/mlp.mlp_32')}
if(!dir.exists('./simulations/megans/mlp.attn_31')){system('mkdir ./simulations/megans/mlp.attn_31')}
if(!dir.exists('./simulations/megans/attn.mlp_22')){system('mkdir ./simulations/megans/attn.mlp_22')}
if(!dir.exists('./simulations/megans/attn.attn_21')){system('mkdir ./simulations/megans/attn.attn_21')}

for (i in 1:100){
  cat("Iteration:", i, "\n")
  digit <- str_pad(i, nchar(4444), pad=0)
  data_nut <- read.csv(paste0("./data/SRS_", digit, ".csv"))
  data_nut$X <- NULL
  data_nut$id <- NULL
  data_nut$R <- NULL
  data_info = list(weight_var = "W",
                   cat_vars = c("hypertension", "bkg_pr", "bkg_o", "female", 
                                "high_chol", "usborn", "idx"),
                   num_vars = names(data_nut)[!names(data_nut) %in% c("hypertension", 
                                                                      "bkg_pr", "bkg_o", "female", 
                                                                      "high_chol", "usborn", "idx")])
  megans_imp <- mmer.impute.cwgangp(data_nut, m = 20, num.normalizing = "mode", cat.encoding = "onehot", 
                                    device = "cpu", epochs = 10000, 
                                    params = list(n_g_layers = 3, n_d_layers = 2, 
                                                  type_g = "mlp", type_d = "mlp"), 
                                    data_info = data_info, save.step = 1000)
  save(megans_imp, file = paste0("./simulations/megans/mlp.mlp_32/", digit, ".RData"))
  megans_imp <- mmer.impute.cwgangp(data_nut, m = 20, num.normalizing = "mode", cat.encoding = "onehot", 
                                    device = "cpu", epochs = 10000, 
                                    params = list(n_g_layers = 3, n_d_layers = 1,
                                                  token_bias = F, token_learn = T,
                                                  type_g = "mlp", type_d = "attn"), 
                                    data_info = data_info, save.step = 1000)
  save(megans_imp, file = paste0("./simulations/megans/mlp.attn_31/", digit, ".RData"))
  megans_imp <- mmer.impute.cwgangp(data_nut, m = 20, num.normalizing = "mode", cat.encoding = "onehot", 
                                    device = "cpu", epochs = 10000, 
                                    params = list(n_g_layers = 2, n_d_layers = 2,
                                                  token_bias = F, token_learn = T,
                                                  type_g = "attn", type_d = "mlp"), 
                                    data_info = data_info, save.step = 1000)
  save(megans_imp, file = paste0("./simulations/megans/attn.mlp_22/", digit, ".RData"))
  megans_imp <- mmer.impute.cwgangp(data_nut, m = 20, num.normalizing = "mode", cat.encoding = "onehot", 
                                    device = "cpu", epochs = 10000, 
                                    params = list(n_g_layers = 2, n_d_layers = 1,
                                                  token_bias = F, token_learn = T,
                                                  type_g = "attn", type_d = "attn"), 
                                    data_info = data_info, save.step = 1000)
  save(megans_imp, file = paste0("./simulations/megans/attn.attn_21/", digit, ".RData"))
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
  var_between.1 <- apply(m_coefs.1, 2, var)
  var_between.2 <- apply(m_coefs.2, 2, var)
  var.1 <- 1/length(imp) * colSums(m_vars.1) + (length(imp) + 1) / length(imp) * var_between.1
  var.2 <- 1/length(imp) * colSums(m_vars.2) + (length(imp) + 1) / length(imp) * var_between.2
  
  lambda.1 <- (var_between.1 + var_between.1 / length(imp)) / var.1
  lambda.2 <- (var_between.2 + var_between.2 / length(imp)) / var.2
  df_old.1 <- (length(imp) - 1) / (lambda.1 ^ 2)
  df_old.2 <- (length(imp) - 1) / (lambda.2 ^ 2)
  df_obs.1 <- (imp_mod.1$df.residual + 1) / (imp_mod.1$df.residual + 3) * imp_mod.1$df.residual * (1 - lambda.1)
  df_obs.2 <- (imp_mod.2$df.residual + 1) / (imp_mod.2$df.residual + 3) * imp_mod.2$df.residual * (1 - lambda.2)
  
  df_adj.1 <- (df_old.1 * df_obs.1) / (df_old.1 + df_obs.1)
  df_adj.2 <- (df_old.2 * df_obs.2) / (df_old.2 + df_obs.2)
  
  return (list(coef = list(colMeans(m_coefs.1), colMeans(m_coefs.2)), var = list(var.1, var.2),
               df_adj = list(df_adj.1, df_adj.2)))
}


CI_coverage <- function(output, true_coef.1, true_coef.2){
  lower.1 <- output$coef[[1]] - qt(0.975, output$df_adj[[1]]) * sqrt(output$var[[1]])
  upper.1 <- output$coef[[1]] + qt(0.975, output$df_adj[[1]]) * sqrt(output$var[[1]])
  
  lower.2 <- output$coef[[2]] - qt(0.975, output$df_adj[[2]]) * sqrt(output$var[[2]])
  upper.2 <- output$coef[[2]] + qt(0.975, output$df_adj[[2]]) * sqrt(output$var[[2]])
  
  logistic <- (true_coef.1 >= lower.1) & (true_coef.1 <= upper.1)
  linear <- (true_coef.2 >= lower.2) & (true_coef.2 <= upper.2)
  return (list(logistic, linear))
}

n <- 20
result_df.1 <- vector("list", n)
result_df.2 <- vector("list", n)
m <- 1
CI_coverage_df.mice <- vector("list", n)
CI_coverage_df.gans <- vector("list", n)
for (i in 1:n){
  digit <- str_pad(i, nchar(4444), pad=0)
  
  if (!file.exists(paste0("./simulations/megans/attn_projd_3_1/", digit, ".RData"))){
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
  
  load(paste0("./simulations/megans/attn_projd_3_1/", digit, ".RData"))
  imp_coefs_vars.gans.pac <- find_coef_var(imp = megans_imp.attn$imputation)

  load(paste0("/nesi/project/uoa03789/PhD/SamplingDesigns/WeightsDesign/Test/testMICE/pmm/MICE_IMPUTE_", digit, ".RData"))
  imp_coefs_vars.mice <- find_coef_var(imp = imputed_data_list)
  
  curr_res.1 <- data.frame(TRUE.Est = coef(true.1),
                           COMPL.Est = coef(complete.1),
                           MICE.imp.Est = imp_coefs_vars.mice$coef[[1]],
                           GANS.imp.Est = imp_coefs_vars.gans.pac$coef[[1]],

                           
                           TRUE.Var = diag(vcov(true.1)),
                           COMPL.Var = diag(vcov(complete.1)),
                           MICE.imp.Var = imp_coefs_vars.mice$var[[1]],
                           GANS.imp.Var = imp_coefs_vars.gans.pac$var[[1]],
                           
                           DIGIT = digit)
  
  curr_res.2 <- data.frame(TRUE.Est = coef(true.2),
                           COMPL.Est = coef(complete.2),
                           MICE.imp.Est = imp_coefs_vars.mice$coef[[2]],
                           GANS.imp.Est = imp_coefs_vars.gans.pac$coef[[2]],
                           
                           TRUE.Var = diag(vcov(true.2)),
                           COMPL.Var = diag(vcov(complete.2)),
                           MICE.imp.Var = imp_coefs_vars.mice$var[[2]],
                           GANS.imp.Var = imp_coefs_vars.gans.pac$var[[2]],
                           
                           DIGIT = digit)
  result_df.1[[m]] <- curr_res.1
  result_df.2[[m]] <- curr_res.2
  
  mice_coverage <- CI_coverage(imp_coefs_vars.mice, coef(true.1), coef(true.2))
  gans_coverage <- CI_coverage(imp_coefs_vars.gans.pac, coef(true.1), coef(true.2))
  
  CI_coverage_df.mice[[m]] <- data.frame(logistic = mice_coverage[[1]], linear = mice_coverage[[2]])
  CI_coverage_df.gans[[m]] <- data.frame(logistic = gans_coverage[[1]], linear = gans_coverage[[2]])
  
  m <- m + 1
}

rowMeans(bind_cols(CI_coverage_df.mice)[, seq(2, 40, 2)])
rowMeans(bind_cols(CI_coverage_df.mice)[, seq(1, 40, 2)])

rowMeans(bind_cols(CI_coverage_df.gans)[, seq(2, 40, 2)])
rowMeans(bind_cols(CI_coverage_df.gans)[, seq(1, 40, 2)])

pacman::p_load("ggplot2", "tidyr", "dplyr", "RColorBrewer", "ggh4x")
#### MODEL-BASED RESULTS
combined_df.1 <- bind_rows(result_df.1) %>%
  filter(grepl("^c_ln_na_true", rownames(.))) %>%
  pivot_longer(
    cols = 1:8,
    names_to = c("METHOD", "TYPE"),
    names_pattern = "^(.*)\\.(Est|Var)$"
  )

combined_df.2 <- bind_rows(result_df.2) %>%
  filter(grepl("^c_ln_na_true", rownames(.))) %>%
  pivot_longer(
    cols = 1:8,
    names_to = c("METHOD", "TYPE"),
    names_pattern = "^(.*)\\.(Est|Var)$"
  )

means.1 <- combined_df.1 %>%
  dplyr::filter(METHOD == "TRUE") %>%
  aggregate(value ~ TYPE, data = ., FUN = mean)

means.2 <- combined_df.2 %>%
  dplyr::filter(METHOD == "TRUE") %>%
  aggregate(value ~ TYPE, data = ., FUN = mean)


combined_df.1 %>%
  filter(TYPE == "Est") %>% 
  group_by(METHOD) %>%
  summarise(rss = sum((value - means.1$value[1])^2))

combined_df.2 %>%
  filter(TYPE == "Est") %>% 
  group_by(METHOD) %>%
  summarise(rss = sum((value - means.2$value[1])^2))


ggplot(combined_df.1) +
  geom_boxplot(aes(x = factor(METHOD, levels = c("TRUE", "COMPL", "MICE.imp", "GANS.imp")),
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
  geom_boxplot(aes(x = factor(METHOD, levels = c("TRUE", "COMPL", "MICE.imp", "GANS.imp")),
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
