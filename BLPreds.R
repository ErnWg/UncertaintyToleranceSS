#Functions for Bayesian Learning of rewards and stimulation probabilities 

#Beta Distribution Model for Shocks
sim_bayes <- function(nSubjects,nRounds,nTrials,nBandits,choice,shock){
  #initialise alpha and beta counts
  alpha <- array(1, dim=c(nSubjects, nRounds, nTrials, nBandits))
  beta <- array(1, dim=c(nSubjects, nRounds, nTrials, nBandits))
  
  for (s in 1:nSubjects){
    for (rd in 1:nRounds){
      for (t in 1:(nTrials-1)){
        alpha[s,rd,t+1,] = alpha[s,rd,t,]
        beta[s,rd,t+1,] = beta[s,rd,t,]
        if (choice[s,rd,t]>0){ #check for missing trials
          if (shock[s,rd,t] == 1){
            alpha[s,rd,t+1,choice[s,rd,t]] = alpha[s,rd,t+1,choice[s,rd,t]] + 1
          }
          else if(shock[s,rd,t] == 0){
            beta[s,rd,t+1,choice[s,rd,t]] = beta[s,rd,t+1,choice[s,rd,t]] + 1
          }
        }
      }
    }
  }
  meanBayes = alpha/(alpha + beta)
  varBayes = sqrt((alpha* beta)/(((alpha+beta)^2)*(alpha+beta+1)))
  entropy = meanBayes*log(meanBayes)-(1-meanBayes)*log(1-meanBayes);
  entropy <- ifelse(meanBayes > 0, -meanBayes * log2(meanBayes), 0)
  return(list(meanBayes=meanBayes, varBayes=varBayes, alpha=alpha, beta=beta, entropy=entropy))
}


#Kalman Filtering of Rewards. Here, the choice matrix doesnt matter since its complete feedback
#Noise arg is sigma of reward distribution

KFcomplete <- function(nSubjects,nRounds,nTrials,nBandits,optA,optB,feedbackA,feedbackB,initR0,noise){
  #initialise prior reward and uncertainty
  #Now I also want to compute a familiarity bias
  Rmu <- array(initR0, dim=c(nSubjects,nRounds,nTrials,nBandits))
  Rsig <- array(sqrt(noise^2 * 20), dim=c(nSubjects,nRounds,nTrials,nBandits))
  left.Rmu <- array(initR0,dim=c(nSubjects,nRounds,nTrials))
  right.Rmu <- array(initR0,dim=c(nSubjects,nRounds,nTrials))

  for (s in 1:nSubjects){
    for (rd in 1:nRounds){
      for (t in 1:(nTrials-1)){
        #Idx vector. This is reset to zero after each trial
        wB <- rep(0,nBandits)
        feedback.vec <- rep(0,nBandits)
        wB[optA[s,rd,t]] = 1
        wB[optB[s,rd,t]] = 1
        feedback.vec[optA[s,rd,t]] = feedbackA[s,rd,t]
        feedback.vec[optB[s,rd,t]] = feedbackB[s,rd,t]
        
        #Update rewards. This is vectorised
        kgain = wB * (Rsig[s,rd,t,]^2 / (Rsig[s,rd,t,]^2 + noise^2))
        Rmu[s,rd,t+1,] = Rmu[s,rd,t,] + kgain * (feedback.vec - Rmu[s,rd,t,])
        Rsig[s,rd,t+1,] = sqrt((1-kgain) * (Rsig[s,rd,t,]^2))
        left.Rmu[s,rd,t+1] = Rmu[s,rd,t+1,optA[s,rd,t+1]]
        right.Rmu[s,rd,t+1] = Rmu[s,rd,t+1,optB[s,rd,t+1]]
      }
    }
  }
  return(list(Rmu=Rmu,Rsig=Rsig,left.Rmu=left.Rmu,right.Rmu=right.Rmu))
}

KFsingle <- function(mu0,rewards,var){
  nObs <- length(rewards)
  Rmu <- rep(mu0,nObs)
  Rsig <- rep(sqrt(var^2 * 20),nObs)
  
  for (t in 1:(nObs-1)){
    kgain = Rsig[t]^2 / (Rsig[t]^2 + var^2)
    Rmu[t+1] = Rmu[t] + kgain*(rewards[t] - Rmu[t])
    Rsig[t+1] = sqrt((1-kgain) * (Rsig[t]^2))
  }
  
  #Scale 0-1
  Rmu.scaled = (Rmu - min(Rmu))/(max(Rmu)-min(Rmu))
  Rsig.scaled = (Rsig - min(Rsig))/(max(Rsig)-min(Rsig))
  
  return(list(Rmu=Rmu.scaled,Rsig=Rsig.scaled))
}


#Testing 
set.seed(29061996)
observations <- rnorm(60,10,2.5)
initR <- 5

#tmp<-KFsingle(initR,observations,2.5)
coeff <- c(1,10,20,30,50)

Rmu <- array(0,dim=c(5,60))
Rsig <- array(0,dim=c(5,60))
for (i in 1:length(coeff)){
  modelPreds <- KFsingle(initR*coeff[i],observations*coeff[i],2.5)
  Rmu[i,] <- modelPreds$Rmu
  Rsig[i,] <- modelPreds$Rsig
}
