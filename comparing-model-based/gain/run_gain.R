library(caret)
run_gain <- function(data, cat_vars){
  dummy <- dummyVars(paste0(" ~ ", cat_vars, collapse = "+", 
                            data = data))
  onehot_encoded <- data.frame(predict(dummy, newdata = data))
  gain_imp <- gain(onehot_encoded, device = "cpu", 
                   batch_size = 128, hint_rate = 0.9, 
                   alpha = 10, beta = 1, n = 10000)
  decoded <- as.data.frame(predict(dummy, gain_imp, inverse = T))
  return (decoded)
}