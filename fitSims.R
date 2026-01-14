## 01_fit_all_models_rstan_safe.R

library(rstan)
library(loo)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

sim_dir   <- "STANmodels/simulations"
fits_dir  <- "STANmodels/recovery"
model_dir <- "STANmodels"

if (!dir.exists(fits_dir)) dir.create(fits_dir, recursive = TRUE)

model_files <- c(
  Reward       = file.path(model_dir, "HBM_Reward.stan"),
  Theta        = file.path(model_dir, "HBM_Theta.stan"),
  Omega        = file.path(model_dir, "HBM_Omega.stan"),
  ThetaOmega   = file.path(model_dir, "HBM_ThetaOmega.stan"),
  MotorSticky  = file.path(model_dir, "HBM_MotorSticky.stan"),
  BanditSticky = file.path(model_dir, "HBM_BanditSticky.stan")
)

stan_models <- lapply(model_files, rstan::stan_model)
names(stan_models) <- names(model_files)

generating_models <- names(stan_models)
candidate_models  <- names(stan_models)

iter_warmup    <- 1000
iter_sampling  <- 2000
chains         <- 4
total_iter     <- iter_warmup + iter_sampling

for (g in generating_models) {
  cat("=== Generating model:", g, "===\n")
  sim_path <- file.path(sim_dir, paste0("sim_", g, ".rds"))
  if (!file.exists(sim_path)) {
    warning("Missing ", sim_path, " — skipping.")
    next
  }
  
  sims <- readRDS(sim_path)
  
  for (i in seq_along(sims)) {
    cat("  Dataset", i, "of", length(sims), "\n")
    stan_data <- sims[[i]]$stan_data
    
    for (m in candidate_models) {
      out_file <- file.path(
        fits_dir,
        paste0("fit_gen-", g, "_data-", i, "_cand-", m, ".rds")
      )
      
      # If file exists, inspect and maybe only do LOO
      if (file.exists(out_file)) {
        cat("    Found existing file for", m, "\n")
        obj <- readRDS(out_file)
        
        if (!is.null(obj$loo) && !is.null(obj$elpd_loo)) {
          cat("      -> loo + elpd present, skipping.\n")
          next
        }
        
        if (!is.null(obj$fit)) {
          cat("      -> fit present, computing loo only.\n")
          fit <- obj$fit
          ext <- rstan::extract(fit)
          if (!"log_lik" %in% names(ext)) {
            stop(paste("Model", m, "missing log_lik in generated quantities."))
          }
          log_lik_mat <- as.matrix(fit, pars = "log_lik")
          loo_obj     <- loo::loo(log_lik_mat)
          elpd        <- loo_obj$estimates["elpd_loo","Estimate"]
          obj$loo      <- loo_obj
          obj$elpd_loo <- elpd
          saveRDS(obj, out_file)
          next
        }
        
        cat("      -> file incomplete, refitting.\n")
      }
      
      cat("    Fitting candidate model:", m, "\n")
      
      fit <- rstan::sampling(
        object = stan_models[[m]],
        data   = stan_data,
        chains = chains,
        iter   = total_iter,
        warmup = iter_warmup,
        seed   = 1000 + i,
        refresh = 0
      )
      
      # Immediately save fit
      obj <- list(
        generating_model = g,
        dataset          = i,
        candidate_model  = m,
        fit              = fit,
        loo              = NULL,
        elpd_loo         = NULL
      )
      saveRDS(obj, out_file)
      
      # Now compute LOO
      ext <- rstan::extract(fit)
      if (!"log_lik" %in% names(ext)) {
        stop(paste("Model", m, "missing log_lik in generated quantities."))
      }
      log_lik_mat <- as.matrix(fit, pars = "log_lik")
      loo_obj     <- loo::loo(log_lik_mat)
      elpd        <- loo_obj$estimates["elpd_loo","Estimate"]
      
      obj$loo      <- loo_obj
      obj$elpd_loo <- elpd
      saveRDS(obj, out_file)
      
      cat("      -> saved fit + loo for", m, "\n")
    }
  }
}

cat("All requested fits complete or already present.\n")

