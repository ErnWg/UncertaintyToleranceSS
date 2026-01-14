rm(list = ls())
options(mc.cores = parallel::detectCores())

# Load packages
packages <- c("tidyverse", "data.table", "jsonlite", "abind", "rstan")
invisible(lapply(packages, require, character.only = TRUE))

# Define data file paths
files <- list(
  data1 = file.path(getwd(), "data/MainExpData.json")
)


process_json <- function(filepath) {
  JsonData <- jsonlite::fromJSON(filepath, flatten = FALSE)
  # Extract experiment data
  ls.rawExpData <- lapply(JsonData$experimentData, jsonlite::fromJSON, flatten = FALSE)
  # Convert to dataframe
  rawExpData <- purrr::map_df(ls.rawExpData, ~ as.data.frame(t(.)))
  # Add meta
  rawExpData$ID <- JsonData$id
  rawExpData$reward <- JsonData$reward
  rawExpData$sessionCode <- JsonData$assignmentID
  rawExpData
}

# Process and append all files
rawExpData <- files %>%
  purrr::map_dfr(process_json) %>%
  dplyr::mutate(
    session = dplyr::case_when(
      sessionCode == "6850248c81fe16d78830a145" ~ 1L,
      sessionCode == "6867becc9dd2733d52bd3037" ~ 2L,
      TRUE ~ NA_integer_
    )
  )

# Export csv for bonus payment 
bonusPay <- rawExpData %>%
  dplyr::filter(reward > 0) %>%
  dplyr::select(Prolific_ID = ID, Bonus = reward)
# write.csv(bonusPay, file.path(getwd(), "data", "bonusPay.csv"), row.names = FALSE, quote = FALSE)

# Check for missing questionnaire data 
missingQuestionnaire <- rawExpData %>%
  dplyr::filter(purrr::map_lgl(`SSS-V`, ~ is.null(.) || all(is.na(.))))

n.missingQuestionnaire <- nrow(missingQuestionnaire)

# Remove missing questionnaire rows
rawExpData <- rawExpData %>%
  dplyr::filter(!purrr::map_lgl(`SSS-V`, ~ is.null(.) || all(is.na(.))))

####################### QA check ##############################################
# IMC check -- must equal c(0,1)
ls.IMC <- lapply(rawExpData$IMC, function(x) as.integer(unlist(x)))
include.IMC <- vapply(ls.IMC, function(x) identical(x, c(0L, 1L)), logical(1))
ExpData <- rawExpData[include.IMC, ]

IMCfail.Data <- rawExpData[!include.IMC, ]

# Calculate nSub after exclusion
n.IMCfail <- sum(!include.IMC)

nSub <- nrow(ExpData)

###############################################################################
# Bandit parameters (identical across subjects)
extract_bandit_params <- function(env_data) {
  dplyr::bind_rows(unlist(env_data, recursive = FALSE)) %>% 
    dplyr::select(!name) %>% dplyr::distinct() %>%
    dplyr::arrange(index)
}

bandit_params_list <- purrr::map(ExpData$images_environments, extract_bandit_params)

if (length(bandit_params_list) > 1) {
  identical_flags <- purrr::map2_lgl(
    bandit_params_list[-length(bandit_params_list)],
    bandit_params_list[-1],
    ~ identical(.x, .y)
  )
  which(!identical_flags) + 1  # mismatches (if any)
}

banditParams <- bandit_params_list[[1]]

nRd <- 4L
nT  <- 60L
nB  <- 6L

phase2_start <- 31L

# Get matrices of behavioural/demographic data
choiceMat     <- array(NA_integer_, dim = c(nSub, nRd, nT))
opt.left      <- array(NA_integer_, dim = c(nSub, nRd, nT))
opt.right     <- array(NA_integer_, dim = c(nSub, nRd, nT))
feedback.left <- array(NA_integer_, dim = c(nSub, nRd, nT))
feedback.right<- array(NA_integer_, dim = c(nSub, nRd, nT))
stim          <- array(NA_integer_, dim = c(nSub, nRd, nT))
scores        <- array(NA_integer_, dim = c(nSub, nRd, nT))
rt            <- array(NA_real_,    dim = c(nSub, nRd, nT))
stimLiking    <- array(NA_real_,    dim = c(nSub, nRd, nB, 3))
stimValence   <- array(NA_integer_, dim = c(nSub, nRd))
stimArousal   <- array(NA_integer_, dim = c(nSub, nRd))

gender  <- rep((NA_integer_), nSub)
age     <- rep((NA_integer_), nSub)
session <- rep((NA_integer_), nSub)
id      <- rep((NA_integer_), nSub)

for (s in seq_len(nSub)){
  # choices are 0-based with 99=miss in raw; map to 1..nB, miss->0
  choice0 <- aperm(array(unlist(ExpData$selectedRoundIndex[s]), dim = c(nT, nRd)))
  missed  <- choice0 >= 99L
  cm      <- choice0 + 1L
  cm[missed] <- 0L
  choiceMat[s,,] <- cm
  
  # options are 0-based in raw → 1..nB
  opt.left[s,,]      <- aperm(array(unlist(ExpData$leftroundIndex[s]),  dim = c(nT, nRd))) + 1L
  opt.right[s,,]     <- aperm(array(unlist(ExpData$rightroundIndex[s]), dim = c(nT, nRd))) + 1L
  
  feedback.left[s,,] <- aperm(array(unlist(ExpData$roundScoresLeft[s]),  dim = c(nT, nRd)))
  feedback.right[s,,]<- aperm(array(unlist(ExpData$roundScoresRight[s]), dim = c(nT, nRd)))
  scores[s,,]        <- aperm(array(unlist(ExpData$roundScores[s]),      dim = c(nT, nRd)))
  rt[s,,]            <- aperm(array(unlist(ExpData$time_round[s]),       dim = c(nT, nRd)))
  
  age[s]    <- as.integer(unlist(ExpData$personalInformation[s])[1])
  gender[s] <- as.integer(unlist(ExpData$personalInformation[s])[2] == "Male")
  stimValence[s,] <- as.integer(array(unlist(ExpData$pleasant_scale[s])))
  stimArousal[s,] <- as.integer(array(unlist(ExpData$wakefulness_scale[s])))
  session[s] <- as.integer(unlist(ExpData$session[s])[1])
  id[s]      <- as.integer(unlist(ExpData$ID[s])[1])
  
  stim_p2 <- aperm(array(unlist(ExpData$roundStimuli[s]),
                         dim = c(nT/2, nRd)))   # -> [Round × nT/2]
  stim[s, , phase2_start:nT] <- stim_p2         # write into phase-2 trials only
}

# Binarise choice: 1=left, 2=right, 0=miss; ensure only valid matches
validChoice <- (choiceMat == opt.left) | (choiceMat == opt.right)
bin.choiceMat <- (choiceMat == opt.right) + 1L
bin.choiceMat[!validChoice] <- 0L



session_df <- data.frame(Subject = seq_len(nSub), id = id, session = session)

nSub      <- dim(scores)[1]
nRd       <- dim(scores)[2]
n_trials  <- dim(scores)[3]
phase2_start <- 31L

# Duplicate valence and arousal ratings
valence3d <- array(stimValence, dim = c(nSub, nRd, nT))
arousal3d <- array(stimArousal, dim = c(nSub, nRd, nT))

# Reward-bias summaries (unchanged)
choice_df <- expand.grid(Subject = seq_len(nSub), Round = seq_len(nRd), Trial = seq_len(n_trials)) %>%
  dplyr::mutate(
    choice = as.vector(choiceMat), 
    bin.choiceMat = as.vector(bin.choiceMat),
    phase = dplyr::if_else(Trial <= 30L, 1L, 2L),
    opt.left = as.vector(opt.left),
    opt.right = as.vector(opt.right),
    feedback.left = as.vector(feedback.left),
    feedback.right = as.vector(feedback.right),
    reward = as.vector(scores),
    rt = as.vector(rt),
    stim = as.vector(stim),
    valence =  as.vector(valence3d),
    arousal = as.vector(arousal3d)
    
  ) %>% dplyr::left_join(
    session_df %>% dplyr::select(Subject, id) %>% dplyr::distinct(),
    by = "Subject"
  ) %>%
  dplyr::relocate(id, .after = Subject)

pTable <- choice_df %>%
  dplyr::group_by(Subject, phase, choice) %>%
  dplyr::summarise(count = dplyr::n(), .groups = "drop_last") %>%
  dplyr::mutate(prop = count / sum(count)) %>%
  dplyr::ungroup() %>%
  dplyr::group_by(Subject, phase, choice) %>%
  dplyr::summarise(prop = mean(prop), .groups = "drop") %>%
  tidyr::pivot_wider(names_from = choice, values_from = prop, names_prefix = "V") %>%
  dplyr::mutate(across(starts_with("V"), ~ tidyr::replace_na(.x, 0))) %>%
  dplyr::left_join(session_df, by = "Subject")

rewardChoice <- pTable %>%
  #dplyr::filter(phase == 2L) %>% 
  dplyr::mutate(Low  = (V1 + V3 + V5), High = (V2 + V4 + V6), rewardBias = High - Low) %>%
  dplyr::group_by(Subject, session, id, phase) %>%
  dplyr::summarise(meanHighChoice = mean(High), .groups = "drop")

pTable <- pTable %>% left_join(rewardChoice, by = c("Subject","session","phase","id"))

# ---- Helper to compute per-phase performance (misses count as 0) ----
phase_perf <- function(idx) {
  # best achievable reward per trial (element-wise max of left vs right)
  best_trial <- pmax(
    feedback.left[,,idx, drop = FALSE],
    feedback.right[,,idx, drop = FALSE]
  )
  # Sum across trials for each Subject x Round
  maxScore <- apply(best_trial, c(1, 2), sum, na.rm = TRUE)
  subScore <- apply(scores[,,idx, drop = FALSE], c(1, 2), sum, na.rm = TRUE)
  
  ratio <- subScore / maxScore
  ratio[!is.finite(ratio)] <- NA_real_  # 0/0, Inf
  rowMeans(ratio, na.rm = TRUE)         # mean across rounds per Subject
}

# Function to compute reward bias for unequal reward trials
prop_best_hvlv <- function(choice_df,
                           low  = c(1L, 3L, 5L),
                           high = c(2L, 4L, 6L),
                           idx = NULL,                 # optional: restrict to specific trials (e.g., 1:30)
                           keys_extra = character()    # optional: e.g., c("session","id")
) {
  keys <- c("Subject", "phase", keys_extra)
  
  df <- choice_df
  if (!is.null(idx)) df <- dplyr::filter(df, .data$Trial %in% idx)
  
  hv_lv_df <- df %>%
    dplyr::filter(
      ( .data$opt.left %in% low  & .data$opt.right %in% high ) |
        ( .data$opt.left %in% high & .data$opt.right %in% low  )
    ) %>%
    dplyr::mutate(best_choice = .data$choice %in% high)
  
  # Trial-weighted (pooled)
  trial_weighted <- hv_lv_df %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(keys))) %>%
    dplyr::summarise(
      propBest_HvLv = mean(best_choice, na.rm = TRUE),
      n_HvLv_trials = dplyr::n(),
      .groups = "drop"
    )
  
  # Equal-round (each Round contributes equally)
  eq_rounds <- hv_lv_df %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(c(keys, "Round")))) %>%
    dplyr::summarise(prop_r = mean(best_choice, na.rm = TRUE), .groups = "drop") %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(keys))) %>%
    dplyr::summarise(propBest_HvLv_eqRounds = mean(prop_r, na.rm = TRUE), .groups = "drop")
  
  dplyr::left_join(trial_weighted, eq_rounds, by = keys)
}

# Phase 1 & 2 performance 
idx_p1 <- 1:(phase2_start - 1L)
idx_p2 <- phase2_start:n_trials

meanPerf_p1 <- phase_perf(idx_p1)
meanPerf_p2 <- phase_perf(idx_p2)

perf_df <- bind_rows(
  tibble(Subject = seq_len(nSub), phase = 1L, meanPerf = meanPerf_p1),
  tibble(Subject = seq_len(nSub), phase = 2L, meanPerf = meanPerf_p2)
)

best_hvlv <- prop_best_hvlv(choice_df)

pTable <- pTable %>%
  dplyr::left_join(perf_df, by = c("Subject", "phase"))  %>%
  dplyr::left_join(best_hvlv, by = c("Subject","phase"))


# Quick visuals
hist(meanPerf_p1)  # Phase-1 performance distribution
plot(meanPerf_p1, meanPerf_p2) 

# Learning model predictions 
source("BLPreds.R")
sigma_reward <- sqrt(mean(banditParams$rewardLevel$variance))
RLearning <- KFcomplete(nSub, nRd, nT, nB, opt.left, opt.right, feedback.left, feedback.right,
                        mean(banditParams$rewardLevel$mean), sigma_reward)

stim_p2 <- stim[, , phase2_start:nT, drop = FALSE]
SLearning <- sim_bayes(nSub, nRd, nT/2, nB, choiceMat[,,31:60], stim_p2) # Phase 2 only
Rmu <- RLearning$Rmu
Rsig <- RLearning$Rsig
left.Rmu <- RLearning$left.Rmu
right.Rmu <- RLearning$right.Rmu

# Scale BL predictions for Stan reward inputs (Smu/Ssig are computed inside Stan)
scale01 <- function(x){(x - min(x)) / (max(x) - min(x))}
scaled.Rmu <- scale01(Rmu)
scaled.Rsig <- scale01(Rsig)
scaled.Smu <- abind::abind(array(0, dim = c(nSub, nRd, 30, nB)), scale01(SLearning$meanBayes), along = 3)
scaled.Ssig <- abind::abind(array(0, dim = c(nSub, nRd, 30, nB)), scale01(SLearning$varBayes), along = 3)


save(scaled.Rmu,scaled.Rsig,scaled.Smu,scaled.Ssig, file="data/modelPreds.RData")


## Compute heuristics

# Familiarity: 1 for bandit chosen at t-1
banditSticky <- array(0L, dim = c(nSub, nRd, nT, nB))
for (s in seq_len(nSub)) for (r in seq_len(nRd)) for (t in 1:(nT - 1)) {
  chosen <- choiceMat[s, r, t]
  if (!is.na(chosen) && chosen > 0L) banditSticky[s, r, t + 1, chosen] <- 1L
}

banditSticky <- scale01(banditSticky)


motorSticky <- array(NA, dim = c(nSub, nRd, nT))
for (s in 1:nSub) {
  for (rd in 1:nRd) {
    for (t in 1:(nT - 1)) {  # Prevent t+1 overflow
      if (bin.choiceMat[s, rd, t] == 2) {
        motorSticky[s, rd, t + 1] <- 0.5  # right
      } else if (bin.choiceMat[s, rd, t] == 1) {
        motorSticky[s, rd, t + 1] <- -0.5  # left
      } else {
        motorSticky[s, rd, t + 1] <- 0  # missed
      }
    }
    # Optionally: first trial has no previous motor info
    motorSticky[s, rd, 1] <- 0
  }
}

pick_option <- function(arr4, opt_idx3) {
  # arr4: [Subject, Round, Trial, Bandit]
  # opt_idx3: [Subject, Round, Trial] integer in 1..nB
  stopifnot(all(dim(arr4)[1:3] == dim(opt_idx3)))
  out <- array(NA_real_, dim = dim(opt_idx3))
  for (b in seq_len(dim(arr4)[4])) {
    mask <- (opt_idx3 == b)
    out[mask] <- arr4[,,,b][mask]
  }
  out
}

Rmu_left  <- pick_option(scaled.Rmu,  opt.left)
Rmu_right  <- pick_option(scaled.Rmu,  opt.right)
Smu_left  <- pick_option(scaled.Smu,  opt.left)
Smu_right <- pick_option(scaled.Smu,  opt.right)
Ssig_left <- pick_option(scaled.Ssig, opt.left)
Ssig_right<- pick_option(scaled.Ssig, opt.right)

banditSticky_left <- pick_option(banditSticky,  opt.left)
banditSticky_right <- pick_option(banditSticky,  opt.right)

choice_df <- choice_df %>%
  dplyr::mutate(
    Rmu_left = as.vector(Rmu_left),
    Rmu_right = as.vector(Rmu_right),
    Smu_left   = as.vector(Smu_left),
    Smu_right  = as.vector(Smu_right),
    Ssig_left  = as.vector(Ssig_left),
    Ssig_right = as.vector(Ssig_right),
    # convenient contrasts:
    dRmu = Rmu_right - Rmu_left,
    dSmu  = Smu_right - Smu_left,
    dSsig = Ssig_right - Ssig_left,
    #Value free heuristics
    dMotorSticky = as.vector(motorSticky),
    dBanditSticky = as.vector(banditSticky_right) - as.vector(banditSticky_left) 
  )



write_csv(pTable, "data/choiceFrequencyByPhaseRaw.csv")
saveRDS(pTable, "data/choiceFrequencyByPhaseRaw.rds")

write_csv(choice_df, "data/choiceDataRaw.csv")
saveRDS(choice_df, "data/choiceDataRaw.rds")


# StanData Objects for fitting (match Stan exactly)
phase2_start <- 31L
stim_expanded <- array(99L, dim = c(nSub, nRd, nT))
stim_expanded[,,phase2_start:nT] <- stim

stanData <- list(
  nSub = nSub, nRd = nRd, nT = nT, nB = nB,
  opt1 = opt.left, opt2 = opt.right,
  choice = bin.choiceMat,                    # 0/1/2
  stim = stim_expanded,                      # 0/1 in phase 2; 99 elsewhere
  phase2_start = phase2_start,
  Rmu = scaled.Rmu,                          # Stan centers: Rmu_c = Rmu - 0.5
  Rsig = scaled.Rsig
)

save(stanData, file = "data/stanData.RData")

# Extract Questionnaire Data 
# SSS
scores.SSS <- scores.SSS.DIS <- scores.SSS.BS <- scores.SSS.TAS <- scores.SSS.ES <- vector()
# BIS
scores.BIS <- scores.BIS.attentional <- scores.BIS.motor <- scores.BIS.nonplan <- vector()
# Others
scores.OCIR <- scores.DASS.D <- scores.DASS.A <- scores.DASS.S <- scores.ICAR <- vector()
scores.DAST <- scores.AUDIT <- scores.addiction <- vector()

questionnaireMat <- array(NA, dim = c(nSub, 109))

for (s in seq_len(nSub)){
  # SSS
  SSS <- as.integer(unlist(ExpData$`SSS-V`[s]))
  reverseIDX.SSS <- c(1,29,32,36,5,8,24,34,39,3,16,17,23,28,6,9,14,18,22)
  SSS[reverseIDX.SSS] <- (SSS[reverseIDX.SSS]*-1) + 1
  scores.SSS[s]      <- sum(SSS)
  scores.SSS.DIS[s]  <- sum(SSS[c(1,12,13,25,29,30,32,33,35,36)])
  scores.SSS.BS[s]   <- sum(SSS[c(2,5,7,8,15,24,27,31,34,39)])
  scores.SSS.TAS[s]  <- sum(SSS[c(3,11,16,17,20,21,23,28,38,40)])
  scores.SSS.ES[s]   <- sum(SSS[c(4,6,9,10,14,18,19,22,26,37)])
  
  # BIS (translate to 1..4)
  BIS <- as.integer(unlist(ExpData$BIS[s])) + 1
  reverseIDX.BIS <- c(1,7,8,10,12,13,15,29,9,20,30)
  BIS[reverseIDX.BIS] <- (BIS[reverseIDX.BIS]*-1) + 5
  scores.BIS[s]             <- sum(BIS)
  scores.BIS.attentional[s] <- sum(BIS[c(5,6,9,11,20,24,26,28)])
  scores.BIS.motor[s]       <- sum(BIS[c(2,3,4,16,17,19,21,22,23,25,30)])
  scores.BIS.nonplan[s]     <- sum(BIS[c(1,7,8,10,12,13,14,15,18,27,29)])
  
  # OCIR
  OCIR <- as.integer(unlist(ExpData$OCIR[s]))
  scores.OCIR[s] <- sum(OCIR)
  
  # DASS
  DASS <- as.integer(unlist(ExpData$DASS[s]))
  scores.DASS.D[s] <- sum(DASS[c(3,5,10,13,16,17,21)])*2
  scores.DASS.A[s] <- sum(DASS[c(2,4,7,9,15,19,20)])*2
  scores.DASS.S[s] <- sum(DASS[c(1,6,8,11,12,14,18)])*2
  
  # ICAR
  ICAR <- as.integer(unlist(ExpData$ICAR[s]))
  correctResp.ICAR <- c(4,4,4,6,6,3,4,4,5,2,2,4,3,2,6,7)
  scores.ICAR[s] <- sum(ICAR == correctResp.ICAR)
  
  # AUDIT
  AUDIT <- as.integer(unlist(ExpData$AUDIT[s]))
  scores.AUDIT[s] <- sum(AUDIT)
  
  # DAST (reverse Q3)
  DAST <- as.integer(unlist(ExpData$DAST[s]))
  reverseIDX.DAST <- 3
  DAST[reverseIDX.DAST] <- (DAST[reverseIDX.DAST]*-1) + 1
  scores.DAST[s] <- sum(DAST)
  
  # Overall addiction (mean of scaled AUDIT + DAST)
  scores.addiction[s] <- mean(c(scores.AUDIT[s]/40, scores.DAST[s]/10))
  
  # Matrix for factor analysis
  questionnaireMat[s,] <- c(SSS, BIS, DASS, OCIR)
}

# Score dataframes
df.Questionnaire <- data.frame(Subject =seq_len(nSub),
                               id = id,
                               SSS = scores.SSS,
                               BIS = scores.BIS,
                               DASS.D = scores.DASS.D,
                               DASS.A = scores.DASS.A,
                               DASS.S = scores.DASS.S,
                               OCIR = scores.OCIR,
                               AUDIT = scores.AUDIT,
                               DAST = scores.DAST,
                               IQ = scores.ICAR,
                               age = age,
                               gender = gender,
                               session = ExpData$session,
                               #subscores
                               SSS.TAS = scores.SSS.TAS, SSS.ES = scores.SSS.ES,
                               SSS.DIS = scores.SSS.DIS, SSS.BS = scores.SSS.BS,
                               BIS.attentional = scores.BIS.attentional,
                               BIS.motor = scores.BIS.motor,
                               BIS.nonplan = scores.BIS.nonplan)

write_csv(df.Questionnaire, "data/questionnaireScoresRaw.csv")
saveRDS(df.Questionnaire, file = "data/questionnaireScoresRaw.RDS")

# Subscales
df.Qsubscores <- data.frame(SSS.TAS = scores.SSS.TAS, SSS.ES = scores.SSS.ES,
                            SSS.DIS = scores.SSS.DIS, SSS.BS = scores.SSS.BS,
                            BIS.attentional = scores.BIS.attentional,
                            BIS.motor = scores.BIS.motor,
                            BIS.nonplan = scores.BIS.nonplan,
                            DASS.D = scores.DASS.D,
                            DASS.A = scores.DASS.A,
                            DASS.S = scores.DASS.S,
                            OCIR = scores.OCIR,
                            AUDIT = scores.AUDIT,
                            DAST = scores.DAST,
                            session = ExpData$session) %>%
  scale()

saveRDS(df.Qsubscores, file = "data/questionnaireSubScores.RDS")


