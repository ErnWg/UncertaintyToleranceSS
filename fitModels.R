rm(list = ls()); gc()
options(stringsAsFactors = FALSE, encoding = "UTF-8")

pkgs <- c("rstan", "parallel", "furrr")
for (p in pkgs) if (!requireNamespace(p, quietly = TRUE)) install.packages(p, dependencies = TRUE)
lapply(pkgs, library, character.only = TRUE)

# Progress bar for furrr, chatgpt inspired
SHOW_OVERALL_PROGRESS <- TRUE
if (SHOW_OVERALL_PROGRESS) {
  if (!requireNamespace("progressr", quietly = TRUE)) install.packages("progressr")
  library(progressr)
  handlers(global = TRUE)
  # pick one:
  handlers("txtprogressbar")   
  # handlers("cli")            
}


rstan_options(auto_write = TRUE)
rstan_options(threads_per_chain = 1)
Sys.setenv(OMP_NUM_THREADS = "1")            


load("data/stanData.RData")                   

# Find models to fit in stanModel folder
models <- list.files("STANmodels", pattern = "\\.stan$", full.names = TRUE)
if (length(models) == 0L) stop("No .stan files found in STANmodels/")

res_dir <- "results"
if (!dir.exists(res_dir)) dir.create(res_dir, recursive = TRUE, showWarnings = FALSE)

avail_cores <- max(1L, parallel::detectCores())
MAX_CORES   <- max(2L, floor(avail_cores * 0.5))   # cap at 50% of CPU (tune this)
CHAINS_DEFAULT <- min(4L, MAX_CORES)
workers <- max(1L, min(length(models), floor(MAX_CORES / CHAINS_DEFAULT)))

message(sprintf("Scheduling: %d cores total | cap=%d | chains/fit=%d | concurrent fits=%d",
                avail_cores, MAX_CORES, CHAINS_DEFAULT, workers))

if (workers > 1L) plan(multisession, workers = workers) else plan(sequential)
set.seed(29061996)

# Progress configs
SHOW_CHAIN_PROGRESS <- TRUE
REFRESH_EVERY <- 200        
LOG_TO_FILE   <- TRUE       

# Fit function
fitModel <- function(model, stanData,
                     chains = CHAINS_DEFAULT,
                     iter = 4000,
                     warmup = 2000,
                     seed = NULL,
                     adapt_delta = 0.95,
                     max_treedepth = 12) {
  t0 <- proc.time()[["elapsed"]]
  mdl_name <- basename(model)
  cat("\n=== Fitting:", mdl_name, "===\n")
  
  if (is.null(seed)) {
    seed <- (abs(sum(utf8ToInt(mdl_name))) %% .Machine$integer.max)
  }
  
  # optionally log chain progress to file (keeps console quieter)
  if (LOG_TO_FILE) {
    log_path <- file.path(res_dir, sub("\\.stan$", ".log", mdl_name))
    zz_out <- file(log_path, open = "wt")
    zz_err <- file(log_path, open = "at")
    sink(zz_out, type = "output"); on.exit(sink(type = "output"), add = TRUE)
    sink(zz_err, type = "message"); on.exit(sink(type = "message"), add = TRUE)
    cat(sprintf("Log for %s\nStarted: %s\n\n", mdl_name, Sys.time()))
  }
  
  fit <- rstan::stan(
    file = model,
    data = stanData,
    chains = chains,
    iter = iter,
    warmup = warmup,
    seed = seed,
    refresh = if (SHOW_CHAIN_PROGRESS) REFRESH_EVERY else 0,  # <-- progress per chain
    cores = chains,
    save_warmup = FALSE,
    verbose = FALSE,
    control = list(adapt_delta = adapt_delta, max_treedepth = max_treedepth)
  )
  
  out_path <- file.path(res_dir, paste0(mdl_name, ".fit.rds"))
  saveRDS(fit, out_path)
  
  elapsed <- round(proc.time()[["elapsed"]] - t0, 1)
  cat(sprintf("Saved: %s | elapsed: %0.1fs\n", out_path, elapsed))
  
  list(model = mdl_name, path = out_path, time = elapsed)
}


safe_fitModel <- function(model, ...) {
  tryCatch(fitModel(model, ...),
           error = function(e) {
             msg <- paste("ERROR in", basename(model), ":", conditionMessage(e))
             cat(msg, "\n")
             list(model = basename(model), error = msg)
           })
}

# Run (with progress bar, inspired by chatgpt)
if (SHOW_OVERALL_PROGRESS) {
  results <- progressr::with_progress({
    p <- progressr::progressor(along = models)
    furrr::future_map(models, function(m) { res <- safe_fitModel(m, stanData = stanData); p(basename(m)); res },
                      .options = furrr::furrr_options(seed = TRUE))
  })
} else {
  results <- furrr::future_map(models, safe_fitModel, stanData = stanData,
                               .options = furrr::furrr_options(seed = TRUE))
}


errors <- vapply(results, function(x) is.list(x) && !is.null(x$error), logical(1))
if (any(errors)) {
  cat("\nModels with errors:\n")
  print(vapply(results[errors], `[[`, "", "model"))
}

ok <- which(!errors)
if (length(ok)) {
  first_fit <- readRDS(results[[ok[1]]]$path)
  print(first_fit, pars = c("mu_theta", "mu_omega", "mu_log_tau", "mu_kappa"))
}

invisible(results)




