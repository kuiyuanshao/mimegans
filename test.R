lapply(c("dplyr", "stringr", "torch", "survival", "mclust"), require, character.only = T)
lapply(paste0("./mimegans/", list.files("./mimegans")), source)
source("00_utils_functions.R")

load("NutritionalData_0001.RData")
for (i in 1:10){
  digit <- stringr::str_pad(i, 4, pad = 0)
  load(paste0("./debug/", digit, ".RData"))
  
  
}