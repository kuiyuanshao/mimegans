lapply(c("ggplot2", "dplyr", "tidyr", "RColorBrewer", "ggh4x"), require, character.only = T)

ggplot(resultCoeff) + 
  geom_boxplot(aes(x = factor(Method, levels = c("me", "complete", "mice", "megans")), 
                   y = as.numeric(`I((HbA1c - 50)/5)`))) + 
  theme_minimal() + 
  labs(x = "Methods", y = "Coefficient") + 
  theme(axis.title.x = element_text(family = "Georgia"),
        axis.title.y = element_text(family = "Georgia"),
        axis.text.x = element_text(family = "Georgia"),
        axis.text.y = element_text(family = "Georgia"),
        legend.title = element_text(family = "Georgia"),
        legend.text = element_text(family = "Georgia"),
        strip.text = element_text(family = "Georgia")) +
  scale_x_discrete(labels = c("me" = "Measurement Error",
                              "complete" = "Complete-Case",
                              "mice" = "MICE", 
                              "megans" = "GANs")) 

ggplot(resultStdError) + 
  geom_boxplot(aes(x = factor(Method, levels = c("true", "me", "complete", "mice", "megans")), 
                   y = as.numeric(`I((HbA1c - 50)/5)`))) + 
  theme_minimal() + 
  labs(x = "Methods", y = "Standard Errors") + 
  theme(axis.title.x = element_text(family = "Georgia"),
        axis.title.y = element_text(family = "Georgia"),
        axis.text.x = element_text(family = "Georgia"),
        axis.text.y = element_text(family = "Georgia"),
        legend.title = element_text(family = "Georgia"),
        legend.text = element_text(family = "Georgia"),
        strip.text = element_text(family = "Georgia")) +
  scale_x_discrete(labels = c("true" = "TRUE",
                              "me" = "Measurement Error",
                              "complete" = "Complete-Case",
                              "mice" = "MICE", 
                              "megans" = "GANs")) 
