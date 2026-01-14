functions {
}

data {
  int<lower=1> nSub;                 // subjects
  int<lower=1> nRd;                  // rounds
  int<lower=1> nT;                   // trials per round (full horizon)
  int<lower=1> nB;                   // number of bandits
  int<lower=1, upper=nB> opt1[nSub,nRd,nT]; // shown option (left)
  int<lower=1, upper=nB> opt2[nSub,nRd,nT]; // shown option (right)
  int<lower=0, upper=2> choice[nSub,nRd,nT]; // 0=miss, 1=left, 2=right
  int stim[nSub,nRd,nT];             // 1=stim, 0=no stim, other values (e.g., 99) ignored
  int<lower=1, upper=nT> phase2_start; // e.g., 31 if phase 2 is 31..60

  // Reward signals (scaled in R). Center around ~0 on the fly.
  real Rmu[nSub,nRd,nT,nB];          // expected reward ~[0,1]
}

parameters {
  // ---- Hierarchical stimulation weights (subject-level) ----
  real mu_theta;               // mean weight for stimulation probability
  real<lower=0> sigma_theta;   // sd of subject deviations
  vector[nSub] theta_raw;      // subject deviations

  real mu_omega;               // mean weight for stimulation uncertainty
  real<lower=0> sigma_omega;
  vector[nSub] omega_raw;

  // ---- Inverse temperature (subject-level) ----
  real mu_log_tau;             // mean log tau
  real<lower=0> sigma_log_tau; // sd of log tau
  vector[nSub] log_tau_raw;

  // ---- Beta count increments (asymmetric; subject-level, positive) ----
  real mu_log_dA;              // log increment for alpha (stim trials)
  real<lower=0> sigma_log_dA;
  vector[nSub] log_dA_raw;

  real mu_log_dB;              // log increment for beta (no-stim trials)
  real<lower=0> sigma_log_dB;
  vector[nSub] log_dB_raw;

  // ---- Motor stickiness (subject-level) ----
  real mu_kappa;               // mean perseveration weight
  real<lower=0> sigma_kappa;   // sd of subject deviations
  vector[nSub] kappa_raw;
}

transformed parameters {
  // Non-centered parameterization
  vector[nSub] theta = mu_theta + sigma_theta * theta_raw; // weight for Smu
  vector[nSub] omega = mu_omega + sigma_omega * omega_raw; // weight for Ssig
  vector[nSub] tau   = exp( mu_log_tau + sigma_log_tau * log_tau_raw );
  vector[nSub] dA    = exp( mu_log_dA  + sigma_log_dA  * log_dA_raw  ); // alpha increment > 0
  vector[nSub] dB    = exp( mu_log_dB  + sigma_log_dB  * log_dB_raw  ); // beta  increment > 0
  vector[nSub] kappa = mu_kappa + sigma_kappa * kappa_raw;             // motor stickiness
}

model {
  // Hyperpriors (weakly-informative)
  mu_theta     ~ normal(0, 1);
  sigma_theta  ~ normal(0, 1);
  theta_raw    ~ normal(0, 1);

  mu_omega     ~ normal(0, 1);
  sigma_omega  ~ normal(0, 1);
  omega_raw    ~ normal(0, 1);

  mu_log_tau    ~ normal(0, 1);    
  sigma_log_tau ~ normal(0, 1);
  log_tau_raw   ~ normal(0, 1);

  mu_log_dA     ~ normal(0, 1);     // increments around exp(0)=1 as baseline
  sigma_log_dA  ~ normal(0, 1);
  log_dA_raw    ~ normal(0, 1);

  mu_log_dB     ~ normal(0, 1);
  sigma_log_dB  ~ normal(0, 1);
  log_dB_raw    ~ normal(0, 1);

  mu_kappa     ~ normal(0, 1);
  sigma_kappa  ~ normal(0, 1);
  kappa_raw    ~ normal(0, 1);

  // Likelihood (reward weight anchored to 1.0)
  {
    real eps; eps = 1e-8; // for safe logit
    for (s in 1:nSub) {
      for (rd in 1:nRd) {
        // Running Beta counts per bandit (Phase-2 only updates)
        real a[nB];  // alpha
        real b[nB];  // beta
        int last;    // last motor response in this round: 0 none, 1 left, 2 right
        for (bb in 1:nB) { a[bb] = 1; b[bb] = 1; }
        last = 0;

        for (t in 1:nT) {
          if (t >= phase2_start && choice[s,rd,t] != 0) {
            int o1; int o2;
            real sum1; real p1; real v1; real sd1; real smu1; real ss1;
            real sum2; real p2; real v2; real sd2; real smu2; real ss2;
            vector[2] V;
            vector[2] logits;
            int bch;

            o1 = opt1[s,rd,t];
            o2 = opt2[s,rd,t];

            // features for o1
            sum1 = a[o1] + b[o1];
            p1   = a[o1] / sum1;
            v1   = (a[o1] * b[o1]) / (sum1 * sum1 * (sum1 + 1));
            sd1  = sqrt(v1);
            smu1 = logit(fmin(fmax(p1,        eps), 1 - eps));
            ss1  = logit(fmin(fmax(2.0 * sd1,  eps), 1 - eps));

            // features for o2
            sum2 = a[o2] + b[o2];
            p2   = a[o2] / sum2;
            v2   = (a[o2] * b[o2]) / (sum2 * sum2 * (sum2 + 1));
            sd2  = sqrt(v2);
            smu2 = logit(fmin(fmax(p2,        eps), 1 - eps));
            ss2  = logit(fmin(fmax(2.0 * sd2,  eps), 1 - eps));

            // base value signal (to be scaled by tau)
            V[1] = (Rmu[s,rd,t,o1] - 0.5) + theta[s] * smu1 + omega[s] * ss1;
            V[2] = (Rmu[s,rd,t,o2] - 0.5) + theta[s] * smu2 + omega[s] * ss2;

            // scale only the value part by tau
            logits = tau[s] * V;

            // add unscaled motor stickiness
            if (last == 1)      logits[1] = logits[1] + kappa[s];
            else if (last == 2) logits[2] = logits[2] + kappa[s];

            target += categorical_logit_lpmf(choice[s,rd,t] | logits);

            // Update counts for next trial (only valid choices in Phase 2)
            if (choice[s,rd,t] == 1) bch = o1;
            else                     bch = o2;
            if (stim[s,rd,t] == 1)      a[bch] += dA[s];
            else if (stim[s,rd,t] == 0) b[bch] += dB[s];

            // update last motor response for stickiness
            last = choice[s,rd,t];
          }
        }
      }
    }
  }
}

generated quantities {
  // Group-level summaries
  real gTheta = mu_theta;
  real gOmega = mu_omega;
  real gTau   = exp(mu_log_tau);
  real g_dA   = exp(mu_log_dA);
  real g_dB   = exp(mu_log_dB);
  real gKappa = mu_kappa;

  // Subject log-likelihoods (for LOO/WAIC); recompute using the same online features
  real log_lik[nSub];
  {
    real eps; eps = 1e-8;
    for (s in 1:nSub) {
      real ll; ll = 0;
      for (rd in 1:nRd) {
        real a[nB];  // alpha
        real b[nB];  // beta
        int last;    // last motor response
        for (bb in 1:nB) { a[bb] = 1; b[bb] = 1; }
        last = 0;

        for (t in 1:nT) {
          if (t >= phase2_start && choice[s,rd,t] != 0) {
            int o1; int o2;
            real sum1; real p1; real v1; real sd1; real smu1; real ss1;
            real sum2; real p2; real v2; real sd2; real smu2; real ss2;
            vector[2] V;
            vector[2] logits;
            int bch;

            o1 = opt1[s,rd,t];
            o2 = opt2[s,rd,t];

            sum1 = a[o1] + b[o1];
            p1   = a[o1] / sum1;
            v1   = (a[o1] * b[o1]) / (sum1 * sum1 * (sum1 + 1));
            sd1  = sqrt(v1);
            smu1 = logit(fmin(fmax(p1,        eps), 1 - eps));
            ss1  = logit(fmin(fmax(2.0 * sd1,  eps), 1 - eps));

            sum2 = a[o2] + b[o2];
            p2   = a[o2] / sum2;
            v2   = (a[o2] * b[o2]) / (sum2 * sum2 * (sum2 + 1));
            sd2  = sqrt(v2);
            smu2 = logit(fmin(fmax(p2,        eps), 1 - eps));
            ss2  = logit(fmin(fmax(2.0 * sd2,  eps), 1 - eps));

            V[1] = (Rmu[s,rd,t,o1] - 0.5) + theta[s] * smu1 + omega[s] * ss1;
            V[2] = (Rmu[s,rd,t,o2] - 0.5) + theta[s] * smu2 + omega[s] * ss2;

            logits = tau[s] * V;

            if (last == 1)      logits[1] = logits[1] + kappa[s];
            else if (last == 2) logits[2] = logits[2] + kappa[s];

            ll  += categorical_logit_lpmf(choice[s,rd,t] | logits);

            if (choice[s,rd,t] == 1) bch = o1;
            else                     bch = o2;
            if (stim[s,rd,t] == 1)      a[bch] += dA[s];
            else if (stim[s,rd,t] == 0) b[bch] += dB[s];

            last = choice[s,rd,t];
          }
        }
      }
      log_lik[s] = ll;
    }
  }
}



