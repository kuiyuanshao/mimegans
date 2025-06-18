#################################### Generate Sample ####################################
lapply(c("BalancedSampling", "dplyr"), require, character.only = T)
source("00_utils_functions.R")
invCll <- function(t) 1-exp(-exp(t))
balSample <- function(obsY,obsD,obsZ,r0,tauCut,pamela=FALSE) {
  
  tmpY <- obsY
  tmpD <- obsD
  
  indOver <- which(tmpY>max(tauCut))
  tmpY[indOver] <- max(tauCut)
  tmpD[indOver] <- 0
  
  obsMat <- matrix(c(tmpY,tmpD,obsZ),nrow=length(tmpY),ncol=3)
  names(obsMat) <- NULL
  
  stratY <- as.numeric(names(table(tmpY)))
  stratD <- as.numeric(names(table(tmpD)))
  stratZ <- as.numeric(names(table(obsZ)))
  
  indStrat <- list()
  tableStrat <- c()
  l <- 1
  for (i in stratY) {
    for (j in stratD) {
      for (k in stratZ) {
        
        tmp <- which(apply(obsMat,1,'identical',y=c(i,j,k))==TRUE)
        tableStrat <- rbind(tableStrat,c(i,j,k,length(tmp)))
        indStrat[[l]] <- tmp
        l <- l+1
      }
    }
  }
  
  nonZeroStrat <- which(tableStrat[,4]!=0)
  numSam <- round(length(obsY)*(1-r0))
  numSamStrat <- numSam/length(nonZeroStrat)
  
  # nFloor <- floor(numSamStrat)
  # nCeiling <- ceiling(numSamStrat)
  # pSam <- nCeiling - numSamStrat
  # tmpSam <- sample(c(nFloor,nCeiling),length(nonZeroStrat),replace=TRUE,prob=c(pSam,1-pSam))
  
  indPhase2Bal <- c()
  openStrat <- c()
  w <- c()
  for (l in 1:length(nonZeroStrat)) {
    tmp <- which(apply(obsMat,1,'identical',y=tableStrat[nonZeroStrat[l],1:3])==TRUE)
    
    # if (length(tmp)<tmpSam[l]) {
    if (length(tmp)<numSamStrat) {
      
      if (pamela==FALSE) {
        indPhase2Bal <- c(indPhase2Bal,tmp)
        w[tmp] <- 1
        openStrat[nonZeroStrat[l]] <- 0
        # w[tmp] <- length(tmp)/length(obsY)
      } else {
        if (tableStrat[nonZeroStrat[l],1]<max(tauCut) & tableStrat[nonZeroStrat[l],2]==0) {
          if (length(tmp)<=1) {
            indPhase2Bal <- c(indPhase2Bal,tmp)
          } else {
            indPhase2Bal <- c(indPhase2Bal,sample(tmp,min(numSamStrat/2,length(tmp))))
          }
          w[tmp] <- min(numSamStrat/2,length(tmp))/length(tmp)
          openStrat[nonZeroStrat[l]] <- 0
        } else {
          indPhase2Bal <- c(indPhase2Bal,tmp)
          w[tmp] <- 1
          openStrat[nonZeroStrat[l]] <- 0
        }
      }
      
    } else {
      
      if (pamela==FALSE) {
        indPhase2Bal <- c(indPhase2Bal,sample(tmp,numSamStrat))
        w[tmp] <- numSamStrat/length(tmp)
        openStrat[nonZeroStrat[l]] <- 1
      } else {
        if (tableStrat[nonZeroStrat[l],1]<max(tauCut) & tableStrat[nonZeroStrat[l],2]==0) {
          if (length(tmp)<=1) {
            indPhase2Bal <- c(indPhase2Bal,tmp)
          } else {
            indPhase2Bal <- c(indPhase2Bal,sample(tmp,numSamStrat/2))
          }
          w[tmp] <- numSamStrat/2/length(tmp)
          openStrat[nonZeroStrat[l]] <- 0
        } else {
          indPhase2Bal <- c(indPhase2Bal,sample(tmp,numSamStrat))
          w[tmp] <- numSamStrat/length(tmp)
          openStrat[nonZeroStrat[l]] <- 1
        }
      }
      # w[tmp] <- numSamStrat/length(obsY)
      
      # indPhase2Bal <- c(indPhase2Bal,sample(tmp,tmpSam[l]))
      # w[tmp] <- tmpSam[l]/length(tmp)
      
    }  
  }
  
  indPhase2Bal <- sort(indPhase2Bal)
  indPhase1Bal <- (1:length(obsY))[-indPhase2Bal]
  
  length(indPhase2Bal)
  
  indOpen <- which(openStrat==1)
  numSamOpenStrat <- (numSam-length(indPhase2Bal))/length(indOpen)
  for (l in 1:length(indOpen)) {
    
    tmp <- which(apply(obsMat[indPhase1Bal,],1,'identical',y=tableStrat[indOpen[l],1:3])==TRUE)
    
    if (length(tmp)<numSamOpenStrat) {
      
      # print(c(indOpen[l],length(tmp),length(tmp)))
      
      indPhase2Bal <- c(indPhase2Bal,indPhase1Bal[tmp])
      
    } else {
      
      # print(c(indOpen[l],length(tmp),numSamOpenStrat))
      
      indPhase2Bal <- c(indPhase2Bal,sample(indPhase1Bal[tmp],numSamOpenStrat))
      
    }
    
  }
  
  indPhase2Bal <- sort(indPhase2Bal)
  indPhase1Bal <- (1:length(obsY))[-indPhase2Bal]
  
  length(indPhase2Bal)
  
  # randomly distribute remaining slots for validation 
  if (length(indPhase2Bal)<numSam) {
    
    if (pamela==FALSE) {
      numRes <- numSam-length(indPhase2Bal)
      ind0 <- sample(indPhase1Bal,numRes,prob=w[indPhase1Bal])
      
      # ind0 <- order(w0[indPhase1Bal])[1:numRes]
      # ind0 <- indPhase1Bal[ind0]
      
      indPhase2Bal <- sort(c(indPhase2Bal,ind0))
      indPhase1Bal <- (1:length(obsY))[-indPhase2Bal]
    } else {
      tmp <- which(obsMat[indPhase1Bal,1]<max(tauCut) & obsMat[indPhase1Bal,2]==0)
      
      numRes <- numSam-length(indPhase2Bal)
      ind0 <- sample(indPhase1Bal[-tmp],numRes,prob=w[indPhase1Bal[-tmp]])
      
      # ind0 <- order(w0[indPhase1Bal])[1:numRes]
      # ind0 <- indPhase1Bal[ind0]
      
      indPhase2Bal <- sort(c(indPhase2Bal,ind0))
      indPhase1Bal <- (1:length(obsY))[-indPhase2Bal]
    }
    
  }
  length(indPhase2Bal)
  
  obsFreq <- c()
  for (i in stratY) {
    for (j in stratD) {
      for (k in stratZ) {
        
        tmp <- which(apply(obsMat,1,'identical',y=c(i,j,k))==TRUE)
        tmp1 <- which(apply(obsMat[indPhase2Bal,],1,'identical',y=c(i,j,k))==TRUE)
        obsFreq <- c(obsFreq,length(tmp1))
        
        w[tmp] <- length(tmp1)/length(tmp)
        # w[tmp] <- length(tmp0)/length(obsY)
        
      }
    }
  }
  
  # w[which(w<0.01)] <- 0.01
  
  tableStrat <- cbind(tableStrat,obsFreq)
  colnames(tableStrat) <- c('Y','D','Z','sizeStrat','sizeSample')
  tableStrat <- tableStrat[,c(1,2,3,5,4)]
  
  return(list(indPhase1Bal=indPhase1Bal,
              indPhase2Bal=indPhase2Bal,
              wt=1/w,
              tableStrat=tableStrat,
              indStrat=indStrat))
}
msFitNew <- function(obsY,obsD,obsZ,obsX,indPhase,tauCut,tol=0.001,wt=NULL,link) {
  #### Code Retrieved From: https://github.com/kyungheehan/mean-score/blob/master/source-functions-github.R
  N <- length(obsY)
  
  indPhase1 <- indPhase[[1]]
  indPhase2 <- indPhase[[2]]
  
  if (is.null(wt)) {
    
    wt <- 1/(length(indPhase2)/(length(indPhase1) + length(indPhase2)))
    wt <- rep(wt,length(obsY))
    
    # wt[indPhase1] <- 1/(length(indPhase2)/(length(indPhase1) + length(indPhase2)))
    # wt[indPhase2] <- 1/(length(indPhase2)/(length(indPhase1) + length(indPhase2)))
  }
  
  valWt <- wt[indPhase2]
  nonvalWt <- wt[indPhase1]
  
  tmpY <- obsY
  tmpD <- obsD
  
  indOver <- which(tmpY>max(tauCut))
  tmpY[indOver] <- max(tauCut)
  tmpD[indOver] <- 0
  
  obsMat <- matrix(c(tmpY,tmpD,obsZ),nrow=length(tmpY),ncol=3)
  phase1Mat <- obsMat[indPhase1,]
  phase2Mat <- obsMat[indPhase2,]
  names(obsMat) <- NULL
  
  stratY <- as.numeric(names(table(tmpY)))
  stratD <- as.numeric(names(table(tmpD)))
  stratZ <- as.numeric(names(table(obsZ)))
  
  indStrat <- indStrat1 <- indStrat2 <- list()
  tableStrat <- c()
  l <- 1
  for (i in stratY) {
    for (j in stratD) {
      for (k in stratZ) {
        
        tmp <- which(apply(obsMat,1,'identical',y=c(i,j,k))==TRUE)
        tmp1 <- which(apply(phase1Mat,1,'identical',y=c(i,j,k))==TRUE)
        tmp2 <- which(apply(phase2Mat,1,'identical',y=c(i,j,k))==TRUE)
        
        tableStrat <- rbind(tableStrat,c(i,j,k,length(tmp1),length(tmp2)))
        
        indStrat1[[l]] <- indPhase1[tmp1]
        indStrat2[[l]] <- indPhase2[tmp2]
        indStrat[[l]] <- tmp
        l <- l+1
      }
    }
  }
  nonZeroStrat <- which((tableStrat[,4]+tableStrat[,5])!=0)
  
  # indStrat <- indStrat[nonZeroStrat]
  indStrat1 <- indStrat1[nonZeroStrat]
  indStrat2 <- indStrat2[nonZeroStrat]
  
  tableStrat <- tableStrat[nonZeroStrat,]
  
  
  valX <- as.matrix(obsX[indPhase2,])
  nonvalX <- as.matrix(obsX[indPhase1,])
  
  valZ <- as.matrix(obsZ[indPhase2,])
  nonvalZ <- as.matrix(obsZ[indPhase1,])
  
  valY <- ceiling(obsY[indPhase2])
  nonvalY <- ceiling(obsY[indPhase1])
  
  # pseudo outervations
  D <- matrix(0,nrow=length(obsY),ncol=length(tauCut))
  for (i in 1:length(obsY)) {
    for (j in 1:length(tauCut)) {
      if ((tmpY[i]) > (tauCut[j]-1) & (tmpY[i]) <= tauCut[j] & tmpD[i]==1) {
        D[i,j] <- 1
      }
    }
  }
  colnames(D) <- tauCut
  
  # cbind(valY,D[indPhase2,],obsY[indPhase2])[1:50,]
  
  valY[which(valY>max(tauCut))] <- max(tauCut)
  nonvalY[which(nonvalY>max(tauCut))] <- max(tauCut)
  
  valD <- D[indPhase2,]
  nonvalD <- D[indPhase1,]
  
  # cbind(valY,valD,obsY[indPhase2],obsD[indPhase2])[1:50,]
  
  combY <- c(valY,nonvalY)
  combD <- rbind(valD,nonvalD)
  combZ <- rbind(valZ,nonvalZ)
  combX <- rbind(valX,nonvalX)
  combWt <- c(valWt,nonvalWt)
  
  combY[indPhase1] <- nonvalY
  combY[indPhase2] <- valY
  
  combD[indPhase1,] <- nonvalD
  combD[indPhase2,] <- valD
  
  combZ[indPhase1,] <- nonvalZ
  combZ[indPhase2,] <- valZ
  
  combX[indPhase1,] <- nonvalX
  combX[indPhase2,] <- valX
  
  combWt[indPhase1] <- nonvalWt
  combWt[indPhase2] <- valWt
  
  ## new version
  
  # initial estimates
  msAlpha <- cpAlpha <- ipwAlpha <- fullAlpha <- rep(0,length(tauCut))
  msBeta <- cpBeta <- ipwBeta <- fullBeta <- rep(0,ncol(obsX))
  
  coefMs <- coefCp <- coefIpw <- coefFull<- c(msAlpha,msBeta)
  
  eps <- 10
  iter <- 1
  while (eps>tol) {
    eps0 <- eps
    tmp1Alpha <- coefMs[1:length(tauCut)]
    tmp1Beta <- coefMs[-(1:length(tauCut))]
    
    tmp2Alpha <- coefCp[1:length(tauCut)]
    tmp2Beta <- coefCp[-(1:length(tauCut))]
    
    tmp3Alpha <- coefFull[1:length(tauCut)]
    tmp3Beta <- coefFull[-(1:length(tauCut))]
    
    tmp4Alpha <- coefIpw[1:length(tauCut)]
    tmp4Beta <- coefIpw[-(1:length(tauCut))]
    
    
    ## score and information
    scoreA <- scoreACp <- scoreAIpw <- scoreAFull <- rep(0,length(tauCut))
    scoreB <- scoreBCp <- scoreBIpw <- scoreBFull <- rep(0,ncol(valX))
    
    hessAA <- hessAACp <- hessAAIpw <- hessAAFull <- matrix(0,nrow=length(tauCut),ncol=length(tauCut))
    hessAB <- hessABCp <- hessABIpw <- hessABFull <- matrix(0,nrow=length(tauCut),ncol=ncol(valX))
    hessBB <- hessBBCp <- hessBBIpw <- hessBBFull <- matrix(0,nrow=ncol(valX),ncol=ncol(valX))
    
    
    for (l in 1:nrow(tableStrat)) {
      
      I1 <- tableStrat[l,4]
      I2 <- tableStrat[l,5]
      
      for (i in indStrat2[[l]]) {
        
        for (j in 1:min(combY[i],max(tauCut))) {
          
          if (link=='logit') {
            
            mu1 <- invLogit(tmp1Alpha[j]+c(t(tmp1Beta)%*%combX[i,]))
            mu2 <- invLogit(tmp2Alpha[j]+c(t(tmp2Beta)%*%combX[i,]))
            mu3 <- invLogit(tmp3Alpha[j]+c(t(tmp3Beta)%*%combX[i,]))
            mu4 <- invLogit(tmp4Alpha[j]+c(t(tmp4Beta)%*%combX[i,]))
            
            # mean score (phase 2 + auxiliary phase 1)
            scoreA[j] <- scoreA[j] + (1+I1/I2)*(combD[i,j] - mu1)
            hessAA[j,j] <- hessAA[j,j] - (1+I1/I2)*mu1*(1-mu1)
            hessAB[j,] <- hessAB[j,] - (1+I1/I2)*mu1*(1-mu1)*combX[i,]
            
            scoreB <- scoreB + (1+I1/I2)*(combD[i,j] - mu1)*combX[i,]
            hessBB <- hessBB - (1+I1/I2)*mu1*(1-mu1)*combX[i,]%*%t(combX[i,])
            
            # complete (phase 2 only)
            scoreACp[j] <- scoreACp[j] + (combD[i,j] - mu2)
            hessAACp[j,j] <- hessAACp[j,j] - mu2*(1-mu2)
            hessABCp[j,] <- hessABCp[j,] - mu2*(1-mu2)*combX[i,]
            
            scoreBCp <- scoreBCp + (combD[i,j] - mu2)*combX[i,]
            hessBBCp <- hessBBCp - mu2*(1-mu2)*combX[i,]%*%t(combX[i,])
            
            # full cohorts
            scoreAFull[j] <- scoreAFull[j] + (combD[i,j] - mu3)
            hessAAFull[j,j] <- hessAAFull[j,j] - mu3*(1-mu3)
            hessABFull[j,] <- hessABFull[j,] - mu3*(1-mu3)*combX[i,]
            
            scoreBFull <- scoreBFull + (combD[i,j] - mu3)*combX[i,]
            hessBBFull <- hessBBFull - mu3*(1-mu3)*combX[i,]%*%t(combX[i,])
            
            # IPW
            scoreAIpw[j] <- scoreAIpw[j] + (combD[i,j] - mu4)*combWt[i]
            hessAAIpw[j,j] <- hessAAIpw[j,j] - mu4*(1-mu4)*combWt[i]
            hessABIpw[j,] <- hessABIpw[j,] - mu4*(1-mu4)*combX[i,]*combWt[i]
            
            scoreBIpw <- scoreBIpw + (combD[i,j] - mu4)*combX[i,]*combWt[i]
            hessBBIpw <- hessBBIpw - mu4*(1-mu4)*combX[i,]%*%t(combX[i,])*combWt[i] 
            
          } else if (link=='cloglog') {
            cll1 <- tmp1Alpha[j]+c(t(tmp1Beta)%*%combX[i,])
            cll2 <- tmp2Alpha[j]+c(t(tmp2Beta)%*%combX[i,])
            cll3 <- tmp3Alpha[j]+c(t(tmp3Beta)%*%combX[i,])
            cll4 <- tmp4Alpha[j]+c(t(tmp4Beta)%*%combX[i,]) 
            
            mu1 <- invCll(cll1)
            mu2 <- invCll(cll2)
            mu3 <- invCll(cll3)
            mu4 <- invCll(cll4)
            
            # mean score (phase 2 + auxiliary phase 1)
            tmp1 <- combD[i,j]*exp(cll1)/mu1 - exp(cll1)
            tmp2 <- combD[i,j]*exp(cll1)/mu1*(1-exp(cll1-exp(cll1))/mu1) - exp(cll1)
            
            scoreA[j] <- scoreA[j] + (1+I1/I2)*tmp1
            hessAA[j,j] <- hessAA[j,j] + (1+I1/I2)*tmp2
            hessAB[j,] <- hessAB[j,] + (1+I1/I2)*tmp2*combX[i,]
            
            scoreB <- scoreB + (1+I1/I2)*tmp1*combX[i,]
            hessBB <- hessBB + (1+I1/I2)*tmp2*combX[i,]%*%t(combX[i,])
            
            # complete (phase 2 only)
            tmp1 <- combD[i,j]*exp(cll2)/mu2 - exp(cll2)
            tmp2 <- combD[i,j]*exp(cll2)/mu2*(1-exp(cll2-exp(cll2))/mu2) - exp(cll2)
            
            scoreACp[j] <- scoreACp[j] + tmp1
            hessAACp[j,j] <- hessAACp[j,j] + tmp2
            hessABCp[j,] <- hessABCp[j,] + tmp2*combX[i,]
            
            scoreBCp <- scoreBCp + tmp1*combX[i,]
            hessBBCp <- hessBBCp + tmp2*combX[i,]%*%t(combX[i,])
            
            # full cohorts
            tmp1 <- combD[i,j]*exp(cll3)/mu3 - exp(cll3)
            tmp2 <- combD[i,j]*exp(cll3)/mu3*(1-exp(cll3-exp(cll3))/mu3) - exp(cll3)
            
            scoreAFull[j] <- scoreAFull[j] + tmp1
            hessAAFull[j,j] <- hessAAFull[j,j] + tmp2
            hessABFull[j,] <- hessABFull[j,] + tmp2*combX[i,]
            
            scoreBFull <- scoreBFull + tmp1*combX[i,]
            hessBBFull <- hessBBFull + tmp2*combX[i,]%*%t(combX[i,])
            
            # IPW
            tmp1 <- combD[i,j]*exp(cll4)/mu4 - exp(cll4)
            tmp2 <- combD[i,j]*exp(cll4)/mu4*(1-exp(cll4-exp(cll4))/mu4) - exp(cll4)
            
            scoreAIpw[j] <- scoreAIpw[j] + tmp1*combWt[i]
            hessAAIpw[j,j] <- hessAAIpw[j,j] + tmp2*combWt[i]
            hessABIpw[j,] <- hessABIpw[j,] + tmp2*combX[i,]*combWt[i]
            
            scoreBIpw <- scoreBIpw + tmp1*combX[i,]*combWt[i]
            hessBBIpw <- hessBBIpw + tmp2*combX[i,]%*%t(combX[i,])*combWt[i] 
            
          }
          
          
        }
      }
      
      # extra updates for full cohorts
      for (i in indStrat1[[l]]) {
        
        for (j in 1:min(combY[i],max(tauCut))) {
          
          if (link=='logit') {
            
            mu3 <- invLogit(tmp3Alpha[j]+c(t(tmp3Beta)%*%combX[i,]))
            
            # full cohorts
            scoreAFull[j] <- scoreAFull[j] + (combD[i,j] - mu3)
            hessAAFull[j,j] <- hessAAFull[j,j] - mu3*(1-mu3)
            hessABFull[j,] <- hessABFull[j,] - mu3*(1-mu3)*combX[i,]
            
            scoreBFull <- scoreBFull + (combD[i,j] - mu3)*combX[i,]
            hessBBFull <- hessBBFull - mu3*(1-mu3)*combX[i,]%*%t(combX[i,])
            
          } else if (link=='cloglog') {
            
            cll3 <- tmp3Alpha[j]+c(t(tmp3Beta)%*%combX[i,])
            mu3 <- invCll(cll3)
            
            # full cohorts
            tmp1 <- combD[i,j]*exp(cll3)/mu3 - exp(cll3)
            tmp2 <- combD[i,j]*exp(cll3)/mu3*(1-exp(cll3-exp(cll3))/mu3) - exp(cll3)
            
            scoreAFull[j] <- scoreAFull[j] + tmp1
            hessAAFull[j,j] <- hessAAFull[j,j] + tmp2
            hessABFull[j,] <- hessABFull[j,] + tmp2*combX[i,]
            
            scoreBFull <- scoreBFull + tmp1*combX[i,]
            hessBBFull <- hessBBFull + tmp2*combX[i,]%*%t(combX[i,])
            
          }
        }
      }
      
    }
    
    # mean score
    score <- c(scoreA,scoreB)
    
    hess1 <- cbind(hessAA,hessAB)
    hess2 <- cbind(t(hessAB),hessBB)
    hess <- rbind(hess1,hess2)
    
    # complete
    scoreCp <- c(scoreACp,scoreBCp)
    
    hess1Cp <- cbind(hessAACp,hessABCp)
    hess2Cp <- cbind(t(hessABCp),hessBBCp)
    hessCp <- rbind(hess1Cp,hess2Cp)
    
    # full cohorts
    scoreFull <- c(scoreAFull,scoreBFull)
    
    hess1Full <- cbind(hessAAFull,hessABFull)
    hess2Full <- cbind(t(hessABFull),hessBBFull)
    hessFull <- rbind(hess1Full,hess2Full)
    
    # IPW
    scoreIpw <- c(scoreAIpw,scoreBIpw)
    
    hess1Ipw <- cbind(hessAAIpw,hessABIpw)
    hess2Ipw <- cbind(t(hessABIpw),hessBBIpw)
    hessIpw <- rbind(hess1Ipw,hess2Ipw)
    
    
    # Newton-Raphson
    coefMs <- coefMs - ginv(hess)%*%score
    coefCp <- coefCp - ginv(hessCp)%*%scoreCp
    coefFull <- coefFull - ginv(hessFull)%*%scoreFull
    coefIpw <- coefIpw - ginv(hessIpw)%*%scoreIpw
    
    epsMs <- max(abs(coefMs-c(tmp1Alpha,tmp1Beta)))
    epsCp <- max(abs(coefCp-c(tmp2Alpha,tmp2Beta)))
    epsFull <- max(abs(coefFull-c(tmp3Alpha,tmp3Beta)))
    epsIpw <- max(abs(coefIpw-c(tmp4Alpha,tmp4Beta)))
    
    
    epsAll <- c(epsMs,epsCp,epsIpw,epsFull)
    eps <- max(epsAll)
    
    iter <- iter+1
    # if (eps-eps0 > 0) {
    if (iter > 20) {
      print('The algorithm may not converge')
      break()
    }
    
  }
  return(list(alphaMs=tmp1Alpha, betaMs=tmp1Beta,
              alphaCp=tmp2Alpha, betaCp=tmp2Beta,
              alphaFull=tmp3Alpha, betaFull=tmp3Beta,
              alphaIpw=tmp4Alpha, betaIpw=tmp4Beta,
              infoFull=-hessFull/length(obsY),
              infoMs  =-hess/length(obsY),
              infoIpw =-hessIpw/length(obsY),
              infoCp  =-hessCp/length(valY),
              indStrat = indStrat,
              tableStrat = tableStrat))
}
msDesignNew <- function(obsY,obsD,obsZ,obsX,indPhase,tauCut,r,alphaTmp,betaTmp,infoTmp,pilotTable,indStrat,link){ 
  #### Code Retrieved From: https://github.com/kyungheehan/mean-score/blob/master/source-functions-github.R
  N <- length(obsY)
  
  tmpY <- obsY
  tmpD <- obsD
  
  indOver <- which(tmpY>max(tauCut))
  tmpY[indOver] <- max(tauCut)
  tmpD[indOver] <- 0
  
  
  indPhase1 <- indPhase[[1]]
  indPhase2 <- indPhase[[2]]
  
  valX <- as.matrix(obsX[indPhase2,])
  nonvalX <- as.matrix(obsX[indPhase1,])
  
  valZ <- as.matrix(obsZ[indPhase2,])
  nonvalZ <- as.matrix(obsZ[indPhase1,])
  
  valY <- ceiling(obsY[indPhase2])
  nonvalY <- ceiling(obsY[indPhase1])
  
  # pseudo outervations
  D <- matrix(0,nrow=length(obsY),ncol=length(tauCut))
  for (i in 1:length(obsY)) {
    for (j in 1:length(tauCut)) {
      if ((tmpY[i]) > (tauCut[j]-1) & (tmpY[i]) <= tauCut[j] & tmpD[i]==1) {
        D[i,j] <- 1
      }
    }
  }
  colnames(D) <- tauCut
  
  # cbind(valY,D[indPhase2,],obsY[indPhase2])[1:50,]
  
  valY[which(valY>max(tauCut))] <- max(tauCut)
  nonvalY[which(nonvalY>max(tauCut))] <- max(tauCut)
  
  valD <- D[indPhase2,]
  nonvalD <- D[indPhase1,]
  
  # cbind(valY,valD,outY[indPhase2],outD[indPhase2])[1:50,]
  
  combY <- c(valY,nonvalY)
  combD <- rbind(valD,nonvalD)
  combZ <- rbind(valZ,nonvalZ)
  combX <- rbind(valX,nonvalX)
  
  combY[indPhase1] <- nonvalY
  combY[indPhase2] <- valY
  
  combD[indPhase1,] <- nonvalD
  combD[indPhase2,] <- valD
  
  combZ[indPhase1,] <- nonvalZ
  combZ[indPhase2,] <- valZ
  
  combX[indPhase1,] <- nonvalX
  combX[indPhase2,] <- valX
  
  
  # optimally distributed sampling design
  optTmp <- optDist(n2yz=NULL,pilotTable,indStrat,
                    combY,combX,combD,indPhase,tauCut,r,alphaTmp,betaTmp,infoTmp,link=link)
  n2Ayz <- optTmp$n2Ayz
  n2Byz0 <- n2ByzOpt <- optTmp$n2Byz
  
  round(cbind(n2Byz0,n2Ayz,pilotTable[,5]),3)
  apply(n2Byz0,2,'sum');sum(n2Ayz)

  iter <- 1
  while (nrow(which(n2Byz0<0,arr.ind=TRUE))>0) {
    iter <- iter+1
    n2yz <- list(A=n2Ayz,B=n2Byz0)
    optTmp <- optDist(n2yz,pilotTable,indStrat,combY,combX,combD,
                      indPhase,tauCut,r,alphaTmp,betaTmp,infoTmp,link=link)
    n2Byz0 <- optTmp$n2Byz
    apply(round(n2Byz0,3),2,'sum');sum(n2Ayz)
    print(iter)
  }
  
  apply(round(n2Byz0,1),2,'sum') + sum(n2Ayz)
  which(n2Byz0<0,arr.ind=TRUE)
  n2Byz <- floor(n2Byz0)
  tmp <- n2Byz0 - n2Byz
  
  for (lc in 1:ncol(n2Byz)) {
    nRemain <- (1-r)*N - sum(n2Byz[,lc]) - sum(n2Ayz)
    if (nRemain > 0) {
      indAdd <- order(tmp[,lc],decreasing=TRUE)[1:nRemain]
      tmp[indAdd,lc] <- 1
      tmp[which(tmp[,lc]<1),lc] <- 0
    }
  }
  n2Byz <- n2Byz + tmp
  
  apply(round(n2Byz,1),2,'sum') + sum(n2Ayz)
  which(n2Byz<0,arr.ind=TRUE)
  
  round(cbind(n2Byz,n2Ayz,pilotTable[,5]),3)
  
  
  # design - McIsaac and Cook
  indSample <- prob <- matrix(0,nrow=nrow(nonvalZ),ncol=ncol(infoTmp))
  for (lc in 1:ncol(n2Byz)) {
    
    for (lr in 1:nrow(n2Byz)) {
      
      if (n2Byz[lr,lc]!=0) {
        
        ind2Byz <- intersect(indStrat[[lr]],indPhase1)
        indTmp <- sample(ind2Byz,min(n2Byz[lr,lc],length(ind2Byz)))
        
        indSample[match(indTmp,indPhase1),lc] <- 1
        
        prob[match(ind2Byz,indPhase1),lc] <- n2Byz0[lr,lc]/length(ind2Byz)
        
      }
    }
    
  }
  
  return(list(indSample=indSample,
              prob=prob,
              n2Ayz=n2Ayz,
              n2ByzOpt=n2ByzOpt,
              n2Byz0=n2Byz0,
              n2Byz=n2Byz))
  
}
optDist <- function(n2yz=NULL,pilotTable,indStrat,combY,combX,combD,indPhase,tauCut,r,alphaTmp,betaTmp,infoTmp,link) {
  #### Code Retrieved From: https://github.com/kyungheehan/mean-score/blob/master/source-functions-github.R
  indPhase1 <- indPhase[[1]]
  indPhase2 <- indPhase[[2]]
  
  if (is.null(n2yz)) {
    
    # optimally distributed sampling design
    numer <- list()
    denom <- 0
    n2Ayz <- rep(0,nrow(pilotTable))
    for (l in 1:nrow(pilotTable)) {
      
      if (pilotTable[l,5]==0) {
        
        numer[[l]] <- rep(0,length(alphaTmp)+length(betaTmp))
        
      } else {
        
        Nyz <- pilotTable[l,5]
        Pyz <- Nyz/N
        
        n2Ayz[l] <- pilotTable[l,4]
        ind2Ayz <- intersect(indStrat[[l]],indPhase2)
        
        scoreStrat <- c()
        
        for (i in ind2Ayz) {
          
          scoreA <-rep(0,length(tauCut))
          scoreB <-rep(0,ncol(combX))
          
          for (j in 1:min(combY[i],max(tauCut))) {
            
            if (link=='logit') {
              
              mu <- invLogit(alphaTmp[j]+c(t(betaTmp)%*%combX[i,]))
              # mu <- invLogit(trueAlpha[j]+c(t(trueBeta)%*%combX[i,]))
              
              scoreA[j] <- combD[i,j] - mu
              scoreB <- scoreB + (combD[i,j] - mu)*combX[i,]
              
            } else if (link=='cloglog') {
              
              cll <- alphaTmp[j]+c(t(betaTmp)%*%combX[i,])
              mu <- invCll(cll)
              # mu <- invLogit(trueAlpha[j]+c(t(trueBeta)%*%combX[i,]))
              
              tmp1 <- combD[i,j]*exp(cll)/mu - exp(cll)
              # tmp2 <- combD[i,j]*exp(cll)/mu*(1-exp(cll-exp(cll)))/mu - exp(cll)
              
              scoreA[j] <- tmp1
              scoreB <- scoreB + tmp1*combX[i,]
              
            }
            
          }
          
          score <- c(scoreA,scoreB)
          scoreStrat <- rbind(scoreStrat,score)
          
        }
        
        if (length(ind2Ayz)<2) {
          V <- matrix(0,nrow=(length(trueAlpha)+length(trueBeta)),ncol=(length(trueAlpha)+length(trueBeta)))
        } else {
          V <- var(scoreStrat)
        }
        
        # round(V,3)
        # round(A,3)
        
        # tmp0 <- ginv(A)%*%V%*%ginv(A)
        tmp0 <- ginv(infoTmp)%*%V%*%ginv(infoTmp)
        tmp0Svd <- svd(tmp0)
        
        svdU <- tmp0Svd$u
        svdD <- tmp0Svd$d
        svdV <- tmp0Svd$v
        
        tmp <- svdU%*%diag(sqrt(svdD))%*%t(svdV)
        
        numer[[l]] <- (1-r)*N*sqrt(Nyz)*sqrt(Pyz)*diag(tmp)
        denom <- denom + sqrt(Nyz)*sqrt(Pyz)*diag(tmp)
        
      }
    }
    
    n2Byz <- matrix(0,nrow=nrow(pilotTable),ncol=(length(alphaTmp)+length(betaTmp)))
    for (l in 1:nrow(pilotTable)) {
      if (pilotTable[l,5]!=0) {
        n2Byz[l,] <- numer[[l]]/denom - n2Ayz[l]
      }
    }
    apply(n2Byz,2,'sum')
    
  } else {
    
    # re-distributed design
    n2Ayz <- n2yz$A
    n2Byz0 <- n2Byz <- n2yz$B
    for (lc in 1:ncol(n2Byz)) {
      
      numer <- rep(NA,nrow(pilotTable))
      denom <- 0
      # n2Ayz <- rep(0,nrow(pilotTable))
      for (lr in 1:nrow(pilotTable)) {
        
        if (n2Byz0[lr,lc] <= 0) {
          
          # print(c(1,lr))
          
          n2Byz[lr,lc] <- 0 
          next()
          
        } else if (n2Byz0[lr,lc] > (pilotTable[lr,5] - n2Ayz[lr])) {
          
          # print(c(2,lr))
          
          n2Byz[lr,lc] <- pilotTable[lr,5] - n2Ayz[lr] 
          next()
          
        } else {
          
          # print(c(3,lr))
          
          Nyz <- pilotTable[lr,5]
          Pyz <- Nyz/N
          
          # n2Ayz[lr] <- pilotTable[lr,4]
          ind2Ayz <- intersect(indStrat[[lr]],indPhase2)
          
          scoreStrat <- c()
          
          for (i in ind2Ayz) {
            
            scoreA <-rep(0,length(tauCut))
            scoreB <-rep(0,ncol(combX))
            
            for (j in 1:min(combY[i],max(tauCut))) {
              
              if (link=='logit') {
                
                mu <- invLogit(alphaTmp[j]+c(t(betaTmp)%*%combX[i,]))
                # mu <- invLogit(trueAlpha[j]+c(t(trueBeta)%*%combX[i,]))
                
                scoreA[j] <- combD[i,j] - mu
                scoreB <- scoreB + (combD[i,j] - mu)*combX[i,]
                
              } else if (link=='cloglog') {
                
                cll <- alphaTmp[j]+c(t(betaTmp)%*%combX[i,])
                mu <- invCll(cll)
                # mu <- invLogit(trueAlpha[j]+c(t(trueBeta)%*%combX[i,]))
                
                tmp1 <- combD[i,j]*exp(cll)/mu - exp(cll)
                # tmp2 <- combD[i,j]*exp(cll)/mu*(1-exp(cll-exp(cll)))/mu - exp(cll)
                
                scoreA[j] <- tmp1
                scoreB <- scoreB + tmp1*combX[i,]
                
              }
              
            }
            
            score <- c(scoreA,scoreB)
            scoreStrat <- rbind(scoreStrat,score)
            
          }
          
          if (length(ind2Ayz)<2) {
            V <- matrix(0,nrow=(length(trueAlpha)+length(trueBeta)),ncol=(length(trueAlpha)+length(trueBeta)))
          } else {
            V <- var(scoreStrat)
          }
          
          tmp0 <- ginv(infoTmp)%*%V%*%ginv(infoTmp)
          tmp0Svd <- svd(tmp0)
          
          svdU <- tmp0Svd$u
          svdD <- tmp0Svd$d
          svdV <- tmp0Svd$v
          
          tmp <- svdU%*%diag(sqrt(svdD))%*%t(svdV)
          
          ind1 <- which(n2Byz0[,lc] <= 0)
          ind2 <- which(n2Byz0[,lc] > (pilotTable[,5] - n2Ayz))
          
          nAdj <- (1-r)*N - sum(n2Ayz[ind1]) - sum(pilotTable[ind2,5])#-n2Ayz[ind2])
          
          # print(c(nAdj,sum(n2Ayz[ind1]),sum(pilotTable[ind2,5]-n2Ayz[ind2])))
          
          numer[lr] <- nAdj*sqrt(Nyz)*sqrt(Pyz)*diag(tmp)[lc]
          denom <- denom + sqrt(Nyz)*sqrt(Pyz)*diag(tmp)[lc]
          
        }
        
      }
      # sum(numer/denom)
      # sum(n2Ayz[which(n2Byz0[,lc]<=0)])
      
      
      # ind1 <- which(n2Byz0[,lc] >= 0)
      # ind2 <- which(n2Byz0[,lc] <= (pilotTable[,5] - n2Ayz))
      # ind3 <- which(pilotTable[,5]!=0)
      # 
      # ind <- unique(sort(c(ind1,ind2,ind3)))
      
      for (lr in which(is.na(numer)==FALSE)) {
        n2Byz[lr,lc] <- numer[lr]/denom - n2Ayz[lr]
      }
      # n2Byz[,lc]
      # sum(n2Byz[,lc])
      # cbind(n2Ayz,n2Byz[,lc])
      # sum(n2Ayz)
      # sum(n2Byz[,lc])
      
      asdf <- cbind(numer/denom,n2Ayz,n2Byz0[,lc],n2Byz[,lc],pilotTable[,5])
      colnames(asdf) <- c('design','n2Ayz','n2Byz0','n2Byz','total')
      asdf
      sum(n2Byz[,lc])
      
    }
    apply(n2Byz,2,'sum') + sum(n2Ayz)
    # which(n2Byz<0,arr.ind=TRUE)
    
  }
  
  return(list(n2Ayz=n2Ayz,n2Byz=n2Byz))
  
}
generateSample <- function(data, proportion, seed){
  set.seed(seed)
  nRow <- N <- nrow(data)
  n_phase2 <- n <- round(nRow * proportion) 
  p2vars <- c("SMOKE", "ALC", "EXER", "INCOME", "EDU", 
              "Na_INTAKE", "K_INTAKE", "KCAL_INTAKE", "PROTEIN_INTAKE", 
              "rs10811661", "rs7756992", "rs11708067", "rs17036101", "rs17584499",
              "rs1111875", "rs4402960", "rs4607103", "rs7754840", "rs9300039",
              "rs5015480", "rs9465871", "rs4506565", "rs5219", "rs358806",
              "GLUCOSE", "F_GLUCOSE", "HbA1c", "INSULIN", "T_I", "EVENT", "C")
  # Simple Random Sampling
  srs_ind <- sample(nRow, n_phase2)
  samp_srs <- data %>%
    dplyr::mutate(R = ifelse(1:nRow %in% srs_ind, 1, 0),
                  W = 1,
                  across(all_of(p2vars), ~ ifelse(R == 0, NA, .)))
  # Balanced Sampling
  time_cut <- cut(data$T_I_STAR, breaks = c(-Inf, seq(3, 24, by = 3), Inf), 
                  labels = 1:8) # Cut by every 3 months
  #hba1c_cut <- as.numeric(cut(data$HbA1c_STAR, breaks = c(-Inf, 64, 75, Inf), 
  #                 labels = 1:3))
  hba1c_cut <- as.numeric(data$URBAN)
  strata <- interaction(data$EVENT_STAR, time_cut, hba1c_cut, drop = TRUE)
  k <- nlevels(strata)
  per_strat <- floor(n_phase2 / k)
  ids_by_str <- split(seq_len(nRow), strata)
  balanced_ind <- unlist(lapply(names(ids_by_str), function(i){
    if (table(strata)[i] < per_strat){
      return (ids_by_str[[i]]) # Sample everyone if insufficient 
    }else{
      return (sample(ids_by_str[[i]], per_strat))
    }
  }))
  openStrata <- names(table(strata)[table(strata) > per_strat])
  remaining_per_strat <- ceiling((n_phase2 - length(balanced_ind)) / length(openStrata))
  remaining_ind <- unlist(lapply(openStrata, function(i){
    sample(ids_by_str[[i]][!(ids_by_str[[i]] %in% balanced_ind)], remaining_per_strat)
  }))[1:(n_phase2 - length(balanced_ind))]
  balanced_ind <- c(balanced_ind, remaining_ind)
  balanced_weights <- table(strata) / table(strata[balanced_ind])
  samp_balance <- data %>%
    dplyr::mutate(R = ifelse(1:nRow %in% balanced_ind, 1, 0),
                  STRATA = strata, 
                  W = case_when(!!!lapply(names(balanced_weights), function(value){
                    expr(STRATA == !!value ~ !!balanced_weights[[value]])
                  })),
                  across(all_of(p2vars), ~ ifelse(R == 0, NA, .)))
  # Adaptive Sampling
  # Han K, Lumley T, Shepherd BE, Shaw PA. Two-phase analysis and study design for survival models with error-prone exposures.
  obsY <- data$T_I_STAR
  obsD <- as.numeric(data$EVENT_STAR)
  obsZ <- as.matrix(as.numeric(data$URBAN))
  obsX <- model.matrix(~ I((HbA1c_STAR - 50) / 5) + rs4506565_STAR + I((AGE - 50) / 5) + SEX + INSURANCE + 
                         RACE + I(BMI / 5) + EXER_STAR, data = data)[, -1]
  r0 <- 0.95
  r <- 0.9
  tauCut <- 1:24
  trueAlpha <- rep(0, 24)
  trueBeta <- rep(0, ncol(obsX))
  pilotSam <- balSample(obsY, obsD, obsZ,
                        r0, tauCut, pamela=TRUE)
  pilotWt <- pilotSam$wt
  pilotTable <- pilotSam$tableStrat
  indStratPilot <- pilotSam$indStrat
  indPhasePilot <- pilotSam[c(1,2)] 
  indPhase1Pilot <- indPhasePilot[[1]]
  indPhase2Pilot <- indPhasePilot[[2]]
  fit <- msFitNew(obsY, obsD, obsZ, obsX, 
                  indPhase = indPhasePilot, 
                  tauCut = tauCut, tol = 0.001,
                  wt = pilotWt, link = "cloglog")
  alphaTmp <- fit$alphaIpw; betaTmp <- fit$betaIpw
  infoTmp <- fit$infoIpw; pilotTable <- fit$tableStrat
  indStrat <- fit$indStrat
  design <- msDesignNew(obsY, obsD, obsZ, obsX, 
                        indPhase = indPhasePilot,
                        tauCut = tauCut, r = r, 
                        alphaTmp = alphaTmp, betaTmp = betaTmp,
                        infoTmp = infoTmp, pilotTable = pilotTable,
                        indStrat = indStrat, link = "cloglog")
  indSample <- data.frame(design$indSample)
  prob <- data.frame(design$prob)
  indPhase2Opt <- sort(c(which(samp_pilot$R == 1), 
                         which(samp_pilot$R == 0)[which(indSample[, 9 + 1]==1)]))
  indPhase1Opt <- which(samp_pilot$R == 0)[which(indSample[, 9 + 1] == 0)]
  
  return (list(samp_srs = samp_srs,
               samp_balance = samp_balance,
               samp_neyman = samp_neyman))
}

####### STARTING SIMULATION.  SAVING FILES ########
if(!dir.exists('./data/Sample')){system('mkdir ./data/Sample')}
if(!dir.exists('./data/Sample/SRS')){system('mkdir ./data/Sample/SRS')}
if(!dir.exists('./data/Sample/Balance')){system('mkdir ./data/Sample/Balance')}
if(!dir.exists('./data/Sample/Neyman')){system('mkdir ./data/Sample/Neyman')}
replicate <- 1
seed <- 1
for (i in 1:replicate){
  digit <- stringr::str_pad(i, 4, pad = 0)
  cat("Current:", digit, "\n")
  load(paste0("./data/Complete/", digit, ".RData"))
  samp_result <- generateSample(data, 0.05, seed)
  write.csv(samp_result$samp_balance, 
            file = paste0("./data/Sample/Balance/", digit, ".csv"))
  write.csv(samp_result$samp_neyman, 
            file = paste0("./data/Sample/Neyman/", digit, ".csv"))
  seed <- seed + 1
}


