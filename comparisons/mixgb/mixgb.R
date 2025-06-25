library(xgboost)
library(ParBayesianOptimization)

# nrounds 50

params <- list(max_depth = 6, subsample = 0.7, eta = 0.3)
cleandata <- data_clean(data)
cv_results <- mixgb_cv(data = cleandata, nrounds = 100, xgb.params = params, verbose = FALSE)
imputed_data_list <- mixgb(cleandata, m = 20, pmm.type = "auto", maxit = 50, 
                           nrounds = cv_results$best.nrounds, xgb.params = params, verbose = F)