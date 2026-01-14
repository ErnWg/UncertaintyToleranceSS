
# Script to simulate data
library(dplyr)
library(purrr)
library(tidyr)

set.seed(1234)

# Config experiment
nSub    <- 100          # subjects per dataset
nRd     <- 4           # rounds (matches your description)
nT      <- 60          # trials per round (30 Phase 1 + 30 Phase 2)
phase2_start <- 31
nB      <- 6           # 6 objects/bandits per round

# reward structure
mu_high <- 10
mu_low  <- -10
sigma_reward <- 2.5    # SD for reward noise

stim_probs_levels <- c(0.15, 0.50, 0.85)


# 6 objects -> all 15 pairs, each 4 times (2x in Phase 1, 2x in Phase 2)

build_round_design <- function() {
  pairs <- combn(1:nB, 2)           # 2 x 15
  nPairs <- ncol(pairs)
  
  # Phase 1: each pair twice -> 30 trials
  idx_p1 <- rep(1:nPairs, each = 2)
  idx_p1 <- sample(idx_p1)          # random order
  
  # Phase 2: each pair twice -> 30 trials
  idx_p2 <- rep(1:nPairs, each = 2)
  idx_p2 <- sample(idx_p2)
  
  opt1 <- integer(nT)
  opt2 <- integer(nT)
  
  # Fill Phase 1 (1..30)
  for (t in 1:30) {
    pr <- pairs[, idx_p1[t]]
    if (runif(1) < 0.5) {
      opt1[t] <- pr[1]; opt2[t] <- pr[2]
    } else {
      opt1[t] <- pr[2]; opt2[t] <- pr[1]
    }
  }
  
  # Fill Phase 2 (31..60)
  for (k in 1:30) {
    pr <- pairs[, idx_p2[k]]
    t  <- 30 + k
    if (runif(1) < 0.5) {
      opt1[t] <- pr[1]; opt2[t] <- pr[2]
    } else {
      opt1[t] <- pr[2]; opt2[t] <- pr[1]
    }
  }
  
  list(opt1 = opt1, opt2 = opt2)
}

# Assign reward means (high/low) and stim probabilities (0.15/0.5/0.85)
# orthogonally across 6 bandits: 3 x 2 factorial all combos used once.
assign_bandit_properties <- function() {
  # all combinations of reward_level x p_stim
  grid <- expand.grid(
    reward = c("low", "high"),
    pstim  = stim_probs_levels
  )
  grid <- grid[sample(1:nrow(grid)), ]  # shuffle assignment to bandits
  
  mu_reward <- ifelse(grid$reward == "high",
                      rnorm(nB, mu_high, sigma_reward),
                      rnorm(nB, mu_low,  sigma_reward))
  
  p_stim <- grid$pstim
  
  list(mu_reward = mu_reward, p_stim = p_stim)
}

# Kalman filter for rewards (full feedback so both options are updated)
kalman_full_feedback <- function(nSub, nRd, nT, nB,
                                 opt1, opt2,
                                 rew1, rew2,
                                 initR0 = 0,
                                 noise = sigma_reward) {
  
  Rmu  <- array(initR0, dim = c(nSub, nRd, nT, nB))
  Rsig <- array(sqrt(noise^2 * 20), dim = c(nSub, nRd, nT, nB))
  
  for (s in 1:nSub) {
    for (rd in 1:nRd) {
      for (t in 1:(nT - 1)) {
        o1 <- opt1[s, rd, t]
        o2 <- opt2[s, rd, t]
        
        wB <- rep(0, nB)
        fb <- rep(0, nB)
        
        wB[o1] <- 1
        wB[o2] <- 1
        
        fb[o1] <- rew1[s, rd, t]
        fb[o2] <- rew2[s, rd, t]
        
        Kgain <- wB * (Rsig[s, rd, t, ]^2 /
                         (Rsig[s, rd, t, ]^2 + noise^2))
        
        Rmu[s, rd, t + 1, ]  <-
          Rmu[s, rd, t, ] + Kgain * (fb - Rmu[s, rd, t, ])
        
        Rsig[s, rd, t + 1, ] <-
          sqrt((1 - Kgain) * (Rsig[s, rd, t, ]^2))
      }
    }
  }
  
  list(Rmu = Rmu, Rsig = Rsig)
}

scale_01 <- function(x) {
  rng <- range(as.numeric(x), na.rm = TRUE)
  if (diff(rng) == 0) {
    x[] <- 0.5
  } else {
    x <- (x - rng[1]) / (rng[2] - rng[1])
  }
  x
}

# Specify Model params to include 
model_features <- list(
  Reward = list(theta = FALSE, omega = FALSE,
                motor_sticky = FALSE, bandit_sticky = FALSE),
  Theta = list(theta = TRUE,  omega = FALSE,
               motor_sticky = FALSE, bandit_sticky = FALSE),
  Omega = list(theta = FALSE, omega = TRUE,
               motor_sticky = FALSE, bandit_sticky = FALSE),
  ThetaOmega = list(theta = TRUE,  omega = TRUE,
                    motor_sticky = FALSE, bandit_sticky = FALSE),
  MotorSticky = list(theta = TRUE,  omega = TRUE,
                     motor_sticky = TRUE,  bandit_sticky = FALSE),
  BanditSticky = list(theta = TRUE,  omega = TRUE,
                      motor_sticky = FALSE, bandit_sticky = TRUE)
)

has_theta        <- function(m) model_features[[m]]$theta
has_omega        <- function(m) model_features[[m]]$omega
has_motor_sticky <- function(m) model_features[[m]]$motor_sticky
has_bandit_sticky<- function(m) model_features[[m]]$bandit_sticky


# Hyperparameters & subject-level params 
# Here We just sim one dataset
sample_hypers <- function(model) {
  feats <- model_features[[model]]
  
  h <- list(
    mu_log_tau    = 0, #rnorm(1, 0, 1),
    sigma_log_tau = 1 #abs(rnorm(1, 0, 1)) + 1e-3
  )
  
  if (feats$theta) {
    h$mu_theta    <- 0 #rnorm(1, 0, 1)
    h$sigma_theta <-1 #abs(rnorm(1, 0, 1)) + 1e-3
  }
  if (feats$omega) {
    h$mu_omega    <- 0 #rnorm(1, 0, 1)
    h$sigma_omega <- 1 #abs(rnorm(1, 0, 1)) + 1e-3
  }
  if (feats$motor_sticky || feats$bandit_sticky) {
    h$mu_kappa    <- 0 #rnorm(1, 0, 1)
    h$sigma_kappa <- 1 #abs(rnorm(1, 0, 1)) + 1e-3
  }
  
  h
}

sample_subject_params <- function(model, hypers, nSub) {
  feats <- model_features[[model]]
  
  log_tau <- rnorm(nSub,
                   mean = 0,
                   sd   = 0.5)
  
  tau <- exp(-log_tau) # Here this is inverse temperature!
  
  theta <- omega <- kappa <- rep(0, nSub)
  
  if (feats$theta) {
    #theta <- rnorm(nSub, hypers$mu_theta, hypers$sigma_theta)
    theta<- runif(nSub, -1.5, 1.5)
  }
  if (feats$omega) {
    #omega <- rnorm(nSub, hypers$mu_omega, hypers$sigma_omega)
    omega <- runif(nSub, -1.5, 1.5)
  }
  if (feats$motor_sticky || feats$bandit_sticky) {
    #kappa <- rnorm(nSub, hypers$mu_kappa, hypers$sigma_kappa)
    kappa <- runif(nSub, -1.5, 1.5)
  }
  
  tibble(
    id    = 1:nSub,
    tau   = tau,
    theta = theta,
    omega = omega,
    kappa = kappa
  )
}

# BLBB specification
beta_features <- function(a, b, eps = 1e-8) {
  sum_ab <- a + b
  p      <- a / sum_ab
  v      <- (a * b) / (sum_ab^2 * (sum_ab + 1))
  sd     <- sqrt(v)
  
  p_clamp  <- pmin(pmax(p, eps), 1 - eps)
  sd2      <- pmin(pmax(2 * sd, eps), 1 - eps)
  
  smu <- qlogis(p_clamp)
  ss  <- qlogis(sd2)
  
  list(smu = smu, ss = ss)
}

softmax_2 <- function(logit1, logit2) {
  m  <- max(logit1, logit2)
  e1 <- exp(logit1 - m)
  e2 <- exp(logit2 - m)
  p1 <- e1 / (e1 + e2)
  if (runif(1) < p1) 1L else 2L
}


# Simulate one dataset for one generating model
simulate_dataset <- function(model) {
  
  feats    <- model_features[[model]]
  hypers   <- sample_hypers(model)
  sub_pars <- sample_subject_params(model, hypers, nSub)
  
  # Arrays shared across subjects (design and true bandit properties)
  opt1 <- array(NA_integer_, c(nSub, nRd, nT))
  opt2 <- array(NA_integer_, c(nSub, nRd, nT))
  stim <- array(0L,          c(nSub, nRd, nT))   # will fill Phase 2
  choice <- array(0L,        c(nSub, nRd, nT))   
  
  # Reward outcomes (for Kalman, full feedback)
  rew1 <- array(NA_real_, c(nSub, nRd, nT))
  rew2 <- array(NA_real_, c(nSub, nRd, nT))
  
  # Stim probabilities per (s, rd, bandit)
  p_stim_arr <- array(NA_real_, c(nSub, nRd, nB))
  
  # 1) Build design + true parameters per subject & round
  for (s in 1:nSub) {
    for (rd in 1:nRd) {
      
      des <- build_round_design()
      opt1[s, rd, ] <- des$opt1
      opt2[s, rd, ] <- des$opt2
      
      props <- assign_bandit_properties()
      mu_reward <- props$mu_reward
      p_stim    <- props$p_stim
      p_stim_arr[s, rd, ] <- p_stim
      
      # Generate reward outcomes for ALL trials, both options shown
      for (t in 1:nT) {
        o1 <- opt1[s, rd, t]
        o2 <- opt2[s, rd, t]
        rew1[s, rd, t] <- rnorm(1, mu_reward[o1], sigma_reward)
        rew2[s, rd, t] <- rnorm(1, mu_reward[o2], sigma_reward)
      }
    }
  }
  
  # 2) Kalman filter for rewards (full feedback)
  kf <- kalman_full_feedback(nSub, nRd, nT, nB,
                             opt1, opt2,
                             rew1, rew2,
                             initR0 = 0,
                             noise = sigma_reward)
  
  Rmu_raw <- kf$Rmu
  # Scale reward signals to [0,1] as in your Stan pipeline
  Rmu <- scale_01(Rmu_raw)
  
  # 3) Simulate Phase-2 choices & stim using Stan-like generative logic
  eps <- 1e-8
  
  for (s in 1:nSub) {
    pars_s <- sub_pars[sub_pars$id == s, ]
    
    for (rd in 1:nRd) {
      # Beta counts for stim probabilities (Phase 2 only)
      a <- rep(1.0, nB)
      b <- rep(1.0, nB)
      
      last_side   <- 0L
      last_bandit <- 0L
      
      for (t in 1:nT) {
        o1 <- opt1[s, rd, t]
        o2 <- opt2[s, rd, t]
        
        ## -------------------------
        ## Phase 1: reward-only
        ## -------------------------
        if (t < phase2_start) {
          base1 <- Rmu[s, rd, t, o1] - 0.5
          base2 <- Rmu[s, rd, t, o2] - 0.5
          
          V1 <- base1
          V2 <- base2
          
          logit1 <- pars_s$tau * V1
          logit2 <- pars_s$tau * V2
          
          # (optional) stickiness in Phase 1:
          if (feats$motor_sticky && last_side != 0L) {
            if (last_side == 1L) logit1 <- logit1 + pars_s$kappa
            if (last_side == 2L) logit2 <- logit2 + pars_s$kappa
          }
          if (feats$bandit_sticky && last_bandit != 0L) {
            if (last_bandit == o1) logit1 <- logit1 + pars_s$kappa
            if (last_bandit == o2) logit2 <- logit2 + pars_s$kappa
          }
          
          ch <- softmax_2(logit1, logit2)
          choice[s, rd, t] <- ch
          
          # No stim in Phase 1
          stim[s, rd, t] <- 0L  # or NA_integer_ if Stan ignores
          
          chosen_bandit <- if (ch == 1L) o1 else o2
          last_side   <- ch
          last_bandit <- chosen_bandit
          
          next  # skip stim/Beta logic
        }
        
        ## -------------------------
        ## Phase 2: reward + stim features
        ## -------------------------
        f1 <- beta_features(a[o1], b[o1], eps)
        f2 <- beta_features(a[o2], b[o2], eps)
        
        base1 <- Rmu[s, rd, t, o1] - 0.5
        base2 <- Rmu[s, rd, t, o2] - 0.5
        
        V1 <- base1
        V2 <- base2
        
        if (feats$theta) {
          V1 <- V1 + pars_s$theta * f1$smu
          V2 <- V2 + pars_s$theta * f2$smu
        }
        if (feats$omega) {
          V1 <- V1 + pars_s$omega * f1$ss
          V2 <- V2 + pars_s$omega * f2$ss
        }
        
        logit1 <- pars_s$tau * V1
        logit2 <- pars_s$tau * V2
        
        # stickiness as you had it
        if (feats$motor_sticky && last_side != 0L) {
          if (last_side == 1L) logit1 <- logit1 + pars_s$kappa
          if (last_side == 2L) logit2 <- logit2 + pars_s$kappa
        }
        if (feats$bandit_sticky && last_bandit != 0L) {
          if (last_bandit == o1) logit1 <- logit1 + pars_s$kappa
          if (last_bandit == o2) logit2 <- logit2 + pars_s$kappa
        }
        
        ch <- softmax_2(logit1, logit2)
        choice[s, rd, t] <- ch
        
        # stim only in Phase 2
        chosen_bandit <- if (ch == 1L) o1 else o2
        p_stim <- p_stim_arr[s, rd, chosen_bandit]
        z <- rbinom(1, 1, p_stim)
        stim[s, rd, t] <- z
        
        # Beta updates only in Phase 2
        if (z == 1L) {
          a[chosen_bandit] <- a[chosen_bandit] + 1
        } else {
          b[chosen_bandit] <- b[chosen_bandit] + 1
        }
        
        last_side   <- ch
        last_bandit <- chosen_bandit
      }
    }
  }
  
  # Assemble stan_data (Phase 1 is in arrays but ignored by Stan model)
  stan_data <- list(
    nSub = nSub,
    nRd  = nRd,
    nT   = nT,
    nB   = nB,
    opt1 = opt1,
    opt2 = opt2,
    choice = choice,
    stim = stim,
    phase2_start = phase2_start,
    Rmu = Rmu
  )
  
  list(
    generating_model   = model,
    hyperparameters    = hypers,
    subject_parameters = sub_pars,
    stan_data          = stan_data
  )
}

#Generate and save datasets per model
models <- names(model_features)
n_datasets_per_model <- 1  # adjust as needed

for (m in models) {
  cat("Simulating for model:", m, "\n")
  
  sims <- map(1:n_datasets_per_model, ~ simulate_dataset(m))
  
  saveRDS(sims, file = paste0("STANmodels/simulations/","sim_", m, ".rds"))
}

cat("All simulation files written.\n")
