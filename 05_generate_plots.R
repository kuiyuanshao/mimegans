lapply(c("ggplot2", "dplyr", "tidyr", "RColorBrewer", "ggh4x"), require, character.only = T)
load("./simulations/results.RData")
resultCoeff_long <- resultCoeff %>% 
  pivot_longer(
    cols = 1:14,
    names_to = "Covariate", 
    values_to = "Coefficient"
  ) %>%
  mutate(Coefficient = as.numeric(Coefficient), 
         Method = factor(Method, levels = c("TRUE", "ME", "COMPLETE", "MIMEGANS", "MICE", "MIXGB", "RAKING")),
         `Sampling Design` = factor(Design, levels = c("SRS", "BALANCE", "NEYMAN")),
         Covariate = factor(Covariate, levels = names(resultCoeff)[1:14], labels = 
                              c("HbA1c", "rs4506565 1", "rs4506565 2", "AGE", "eGFR", "SEX TRUE", "INSURANCE TRUE", 
                                "RACE AFR", "RACE AMR", "RACE SAS", "RACE EAS", "BMI", "SMOKE 2", "SMOKE 3")))

means.coef <- resultCoeff_long %>% 
  filter(Method == "TRUE") %>%
  select(-c("Design", "Method", "ID")) %>% 
  group_by(Covariate) %>%
  summarise(mean = mean(Coefficient))


range_coef <- list(Covariate == "HbA1c" ~ scale_y_continuous(limits = c(means.coef$mean[1] - 0.15, means.coef$mean[1] + 0.15)),
                   Covariate == "rs4506565 1" ~ scale_y_continuous(limits = c(means.coef$mean[2] - 0.25, means.coef$mean[2] + 0.25)),
                   Covariate == "rs4506565 2" ~ scale_y_continuous(limits = c(means.coef$mean[3] - 0.25, means.coef$mean[3] + 0.25)),
                   Covariate == "AGE" ~ scale_y_continuous(limits = c(means.coef$mean[4] - 0.1, means.coef$mean[4] + 0.1)),
                   Covariate == "eGFR" ~ scale_y_continuous(limits = c(means.coef$mean[5] - 0.1, means.coef$mean[5] + 0.1)),
                   Covariate == "SEX TRUE" ~ scale_y_continuous(limits = c(means.coef$mean[6] - 0.2, means.coef$mean[6] + 0.2)),
                   Covariate == "INSURANCE TRUE" ~ scale_y_continuous(limits = c(means.coef$mean[7] - 0.2, means.coef$mean[7] + 0.2)),
                   Covariate == "RACE AFR" ~ scale_y_continuous(limits = c(means.coef$mean[8] - 0.25, means.coef$mean[8] + 0.25)),
                   Covariate == "RACE AMR" ~ scale_y_continuous(limits = c(means.coef$mean[9] - 0.25, means.coef$mean[9] + 0.25)),
                   Covariate == "RACE SAS" ~ scale_y_continuous(limits = c(means.coef$mean[10] - 0.25, means.coef$mean[10] + 0.25)),
                   Covariate == "RACE EAS" ~ scale_y_continuous(limits = c(means.coef$mean[11] - 0.25, means.coef$mean[11] + 0.25)),
                   Covariate == "BMI" ~ scale_y_continuous(limits = c(means.coef$mean[12] - 0.25, means.coef$mean[12] + 0.25)),
                   Covariate == "SMOKE 2" ~ scale_y_continuous(limits = c(means.coef$mean[13] - 0.25, means.coef$mean[13] + 0.25)),
                   Covariate == "SMOKE 3" ~ scale_y_continuous(limits = c(means.coef$mean[14] - 0.25, means.coef$mean[14] + 0.25)))

ggplot(resultCoeff_long) + 
  geom_boxplot(aes(x = Method, 
                   y = Coefficient,
                   colour = `Sampling Design`)) + 
  geom_hline(data = means.coef, aes(yintercept = mean), lty = 2) + 
  facet_wrap(~ Covariate, scales = "free") + 
  theme_minimal() + 
  theme(axis.title.x = element_text(family = "Georgia"),
        axis.title.y = element_text(family = "Georgia"),
        axis.text.x = element_text(family = "Georgia"),
        axis.text.y = element_text(family = "Georgia"),
        legend.title = element_text(family = "Georgia"),
        legend.text = element_text(family = "Georgia"),
        strip.text = element_text(family = "Georgia")) + 
  scale_colour_manual(
    values = c("SRS" = "red", "BALANCE" = "green", "NEYMAN" = "blue", "NA" = "black"),
    breaks = c("SRS", "BALANCE", "NEYMAN")) + 
  facetted_pos_scales(y = range_coef)

ggsave("./simulations/Imputation_Coeff_Boxplot2.png", width = 30, height = 10, limitsize = F)

ggplot(resultStdError) + 
  geom_boxplot(aes(x = factor(Method, levels = c("TRUE", "ME", "COMPLETE", "MIMEGANS", "MICE", "MIXGB", "RAKING")), 
                   y = as.numeric(`I((HbA1c - 50)/5)`),
                   colour = factor(Design, levels = c("SRS", "BALANCE", "NEYMAN")))) + 
  theme_minimal() + 
  labs(x = "Methods", y = "Standard Errors") + 
  theme(axis.title.x = element_text(family = "Georgia"),
        axis.title.y = element_text(family = "Georgia"),
        axis.text.x = element_text(family = "Georgia"),
        axis.text.y = element_text(family = "Georgia"),
        legend.title = element_text(family = "Georgia"),
        legend.text = element_text(family = "Georgia"),
        strip.text = element_text(family = "Georgia")) + 
  scale_colour_manual(
    name = "Sampling Design",
    values = c("SRS" = "red", "BALANCE" = "green", "NEYMAN" = "blue", "NA" = "black"),
    breaks = c("SRS", "BALANCE", "NEYMAN")
  )
