expit <- function(x){
  exp(x) / (1 + exp(x))
}

calcCICover <- function(true, lower, upper){
  return (true >= lower) & (true <= upper)
}

as.mids <- function(imp_list){
  imp_mids <- miceadds::datlist2mids(imp_list)
  return (imp_mids)
}

exactAllocation <- function(data, stratum_variable, target_variable, sample_size){
  strata_units <- as.data.frame(table(data[[stratum_variable]]))
  colnames(strata_units) <- c(stratum_variable, "count")
  conversion_functions <- list(
    numeric = "as.numeric",
    integer = "as.integer",
    character = "as.character",
    logical = "as.logical",
    factor = "as.factor"
  )
  strata_units[, 1] <-  do.call(conversion_functions[[class(data[[stratum_variable]])[1]]], list(strata_units[, 1]))
  
  data <- merge(data, strata_units, by = stratum_variable)
  Y_bars <- aggregate(as.formula(paste0(target_variable, " ~ ", stratum_variable)), data = data, FUN = function(x) sum(x) / length(x))
  colnames(Y_bars)[2] <- "Y_bars"
  data <- merge(data, Y_bars, by = stratum_variable)
  Ss <- aggregate(as.formula(paste0("(", target_variable, " - Y_bars", ")^2", " ~ ", stratum_variable)), 
                  data = data, FUN = function(x) sum(x) / (length(x) - 1))
  
  NS <- strata_units$count * sqrt(Ss[, 2])
  names(NS) <- Ss[, 1]
  NS <- NS[order(NS, decreasing = T)]
  # Type-II
  columns <- sample_size - 2 * nrow(Ss)
  priority <- matrix(0, nrow = columns, ncol = nrow(Ss))
  colnames(priority) <- names(NS)
  for (h in names(NS)){
    priority[, h] <- NS[[h]] / sqrt((2:(columns + 1)) * (3:(columns + 2)))
  }
  priority <- as.data.frame(priority)
  priority <- stack(priority)
  colnames(priority) <- c("value", stratum_variable)
  order_priority <- order(priority$value, decreasing = T)
  alloc <- (table(priority[[stratum_variable]][order_priority[1:columns]]) + 2)
  alloc <- alloc[order(as.integer(names(alloc)))]
  return (alloc)
}


