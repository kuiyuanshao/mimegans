lapply(c("ggplot2", "dplyr", "tidyr", "RColorBrewer", "ggh4x", "survival", "survminer", "broom"), require, character.only = T)
source("00_utils_functions.R")

digit = "0026"
load(paste0("./simulations/Balance/megans/", digit, ".RData"))
load(paste0("./data/Complete/", digit, ".RData"))

cox.true <- coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                    rs4506565 + I((AGE - 50) / 5) + SEX + INSURANCE + 
                    RACE + I(BMI / 5) + SMOKE, data = data)
megans_imp$imputation <- lapply(megans_imp$imputation, function(dat){
  match_types(dat, data)
})
imp.mids <- as.mids(megans_imp$imputation)
fit <- with(data = imp.mids, 
            exp = coxph(Surv(T_I, EVENT) ~ I((HbA1c - 50) / 5) + 
                          rs4506565 + I((AGE - 50) / 5) + 
                          SEX + INSURANCE + 
                          RACE + I(BMI / 5) + SMOKE))
pooled <- mice::pool(fit)
sumry <- summary(pooled, conf.int = TRUE)
round(exp(sumry$estimate) - exp(coef(cox.true)), 4)
sumry$std.error
sqrt(diag(vcov(cox.true)))
ggplot(megans_imp$imputation[[4]]) + 
  geom_density(aes(x = T_I), colour = "red") +
  geom_density(aes(x = T_I), data = data)

ggplot(megans_imp$imputation[[1]]) + 
  geom_density(aes(x = HbA1c), colour = "red") +
  geom_density(aes(x = HbA1c), data = data)

ggplot(resultCoeff) + 
  geom_boxplot(aes(x = factor(Method, levels = c("me", "complete", "mice", "megans")), 
                   y = as.numeric(`I((HbA1c - 50)/5)`))) + 
  theme_minimal() + 
  labs(x = "Methods", y = "Coefficient Bias") + 
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
                              "megans" = "GANs")) +
  geom_hline(yintercept = 0, colour = "red", lty = "dashed")

resultCoeff %>% group_by(Method) %>%
  summarise(sqrt(mean(as.numeric(`I((HbA1c - 50)/5)`) ^ 2)))

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

resultStdError


load("./data/Complete/0007.RData")
samp <- read.csv("./data/Sample/Balance/0007.csv")
load("./simulations/Balance/mice/0007.RData")
load("./simulations/Balance/megans/0007.RData")
samp <- match_types(samp, data)
# mice_imp <- mice::complete(mice_imp, "all")
mice_imp <- lapply(mice_imp, function(dat){
  match_types(dat, data)
})
megans_imp$imputation <- lapply(megans_imp$imputation, function(dat){
  match_types(dat, data)
})

true_fit <- survfit(Surv(T_I, EVENT) ~ 1, 
                    type = "kaplan-meier", data = data)
me_fit <- survfit(Surv(T_I_STAR, EVENT_STAR) ~ 1, 
                  type = "kaplan-meier", data = data)
complcase_fit <- survfit(Surv(T_I, EVENT) ~ 1, 
                         type = "kaplan-meier", data = samp)

mice_fits <- map(mice_imp, ~ survfit(Surv(T_I, EVENT) ~ 1, data = .x))
megans_fits <- map(megans_imp$imputation, ~ survfit(Surv(T_I, EVENT) ~ 1, data = .x))
time_grid <- seq(0, 24.001, by = 0.1)

true_sumry <- summary(true_fit, times = time_grid, extend = TRUE)
true_df <- tibble(time = true_sumry$time,
                  estimate = true_sumry$surv,
                  std.error = true_sumry$std.err,
                  conf.low = true_sumry$lower,
                  conf.high = true_sumry$upper,
                  method = "TRUE")
me_sumry <- summary(me_fit, times = time_grid, extend = TRUE)
me_df <- tibble(time = me_sumry$time,
                estimate = me_sumry$surv,
                std.error = me_sumry$std.err,
                conf.low = me_sumry$lower,
                conf.high = me_sumry$upper,
                method = "Measurement Error") 
complcase_sumry <- summary(complcase_fit, times = time_grid, extend = TRUE)
complcase_df <- tibble(time = complcase_sumry$time,
                       estimate = complcase_sumry$surv,
                       std.error = complcase_sumry$std.err,
                       conf.low = complcase_sumry$lower,
                       conf.high = complcase_sumry$upper,
                       method = "Complete-Case")

mice_df <- map2_dfr(mice_fits, seq_along(mice_fits), function(fit, i) {
  s <- summary(fit, times = time_grid, extend = TRUE)
  tibble(
    .imp = i,
    time = s$time,
    surv = s$surv,
    std.err = s$std.err
  )}) %>%
  group_by(time) %>%
  summarize(
    B = var(surv), 
    estimate = mean(surv),
    W = mean(std.err^2),
    std.error = sqrt(W + (1 + 1/20) * B),
    conf.low = estimate - 1.96 * std.error,
    conf.high = estimate + 1.96 * std.error,
    .groups = "drop"
  ) %>%
  select(time, estimate, std.error, conf.low, conf.high) %>%
  mutate(method = "MICE")

megans_df <- map2_dfr(megans_fits, seq_along(megans_fits), function(fit, i) {
  s <- summary(fit, times = time_grid, extend = TRUE)
  tibble(
    .imp = i,
    time = s$time,
    surv = s$surv,
    std.err = s$std.err
  )}) %>%
  group_by(time) %>%
  summarize(
    B = var(surv), 
    estimate = mean(surv),
    W = mean(std.err^2),
    std.error = sqrt(W + (1 + 1/5) * B),
    conf.low = estimate - 1.96 * std.error,
    conf.high = estimate + 1.96 * std.error,
    .groups = "drop"
  ) %>%
  select(time, estimate, std.error, conf.low, conf.high) %>%
  mutate(method = "GANs")


plot_df <- rbind(true_df, megans_df)

ggplot(plot_df, aes(time, 1 - estimate, colour = method, fill = method)) +
  geom_step(linewidth = 0.3, alpha = 0.5) +
  geom_ribbon(aes(ymin = 1 - conf.low, ymax = 1 - conf.high,
                  colour = method),
              alpha = 0.1) +
  labs(x = "Time", y = "Cumulative Risk",
       colour = NULL, fill = NULL) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

print(p)

mice_imp <- mice_imp$`1`
mice_imp$Type <- "MICE"
megans_imp <- megans_imp$imputation[[1]]
megans_imp$Type <- "GANs"
data$Type <- "TRUE"
df <- rbind(data %>% select(T_I, Type), 
            megans_imp %>% select(T_I, Type))
ggplot(df) + 
  geom_density(aes(x = T_I, colour = Type)) + 
  geom_density(aes(x = T_I_STAR), data = data)
  xlab("Time") + theme_minimal()

ggplot() + 
  geom_density(aes(x = T_I_STAR), data = samp %>% filter(R == 1), colour = "blue") + 
  geom_density(aes(x = T_I), data = samp, colour = "red")

ggplot() + 
  geom_density(aes(x = T_I_STAR - T_I), data = samp)
# mice_km_fit <- lapply(seq_along(mice_imp), function(i) {
#   d <- mice_imp[[i]]
#   fit <- survfit(Surv(T_I, EVENT) ~ 1, data = d)
#   tidy(fit) |> 
#     mutate(.imp = i)
# })
# mice_km_fit <- do.call(rbind, mice_km_fit)
# megans_km_fit <- lapply(seq_along(megans_imp$imputation), function(i) {
#   d <- megans_imp$imputation[[i]]
#   fit <- survfit(Surv(T_I, EVENT) ~ 1, data = d)
#   tidy(fit) |> 
#     mutate(.imp = i)
# })
# megans_km_fit <- do.call(rbind, megans_km_fit)
# 
# mice_km_fit <- mice_km_fit %>%
#   group_by(time) %>%
#   summarize(
#     m = n_distinct(.imp), 
#     n.risk = round(mean(n.risk)),
#     n.event = round(mean(n.event)),
#     n.censor = round(mean(n.censor)),
#     B = var(estimate), 
#     estimate = mean(estimate),
#     W = mean(std.error^2),
#     std.error = sqrt(W + (1 + 1/m) * B),
#     conf.low = estimate - 1.96 * std.error,
#     conf.high = estimate + 1.96 * std.error,
#     .groups = "drop"
#   ) %>%
#   select(time, n.risk, n.event, n.censor,
#          estimate, std.error, conf.low, conf.high) %>%
#   mutate(method = "MICE")
# 
# megans_km_fit <- megans_km_fit %>%
#   group_by(time) %>%
#   summarize(
#     m = n_distinct(.imp), 
#     n.risk = round(mean(n.risk)),
#     n.event = round(mean(n.event)),
#     n.censor = round(mean(n.censor)),
#     B = var(estimate), 
#     estimate = mean(estimate),
#     W = mean(std.error^2),
#     std.error = sqrt(W + (1 + 1/m) * B),
#     conf.low = estimate - 1.96 * std.error,
#     conf.high = estimate + 1.96 * std.error,
#     .groups = "drop"
#   ) %>%
#   select(time, n.risk, n.event, n.censor,
#          estimate, std.error, conf.low, conf.high) %>%
#   mutate(method = "GANs")
# 
# true_km_fit <- tidy(true_km_fit) |>
#   mutate(method = "TRUE")
# me_km_fit <- tidy(me_km_fit) |>
#   mutate(method = "Measurement Error")
# complcase_km_fit <- tidy(complcase_km_fit) |>
#   mutate(method = "Complete Case")
