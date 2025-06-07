library(xgboost)
library(ParBayesianOptimization)
param_tune <- function(data, params){
  cv_results <- mixgb_cv(data = data, nrounds = 1000, xgb.params = params, verbose = FALSE)
  
}


param_to_tune <- list(eta = c(0.01, 0.3),
                      max_depth = c(3, 12),
                      min_child_weight = c(1, 10),
                      subsample = c(0.5, 1),
                      colsample_bytree = c(0.5, 1),
                      gamma = c(0, 5))


params <- list(max_depth = 6, subsample = 0.7, eta = 0.3)
cleandata <- data_clean(data)
cv_results <- mixgb_cv(data = cleandata, nrounds = 100, xgb.params = params, verbose = FALSE)
imputed_data_list <- mixgb(cleandata, m = 20, pmm.type = "auto", maxit = 50, 
                           nrounds = cv_results$best.nrounds, xgb.params = params, verbose = F)