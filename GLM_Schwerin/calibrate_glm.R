# =============================================================================
# Scientifically rigorous GLM calibration for Lake Schwerin
# -----------------------------------------------------------------------------
# Approach:
#   * Split-sample design - calibrate on an early period, validate on a later,
#     independent period (no observation overlap between the two).
#   * Joint parameter search via Latin Hypercube Sampling (LHS) so that
#     parameter interactions are accounted for (instead of staged, one-at-a-time
#     grids that can settle on local optima or compensate errors).
#   * Physically bounded parameters - the bulk transfer coefficients are kept
#     within their physically plausible range rather than pushed to absorb bias.
#   * Multiple skill metrics (RMSE, bias, Nash-Sutcliffe efficiency, R2, MAE)
#     reported per depth layer for both calibration and validation periods.
#   * Reproducible - a fixed random seed defines the LHS design.
#
# The best parameter set (lowest calibration-period RMSE) is written back to
# glm4.nml and the full sample table is saved to Data/Processed/.
# =============================================================================

source("config.R")

suppressMessages(library(ncdf4))

set.seed(42)

# Number of LHS samples (each is one GLM run; runs are fast, ~seconds).
N_SAMPLES <- 200

# -----------------------------------------------------------------------------
# Parameter definition (name, GLM block, lower bound, upper bound)
# Bounds reflect physically plausible ranges for a sheltered dimictic lake.
# -----------------------------------------------------------------------------
params <- list(
  list(name = "wind_factor",   block = "meteorology", lo = 0.70,   hi = 1.00),
  list(name = "Kw",            block = "light",       lo = 0.30,   hi = 0.80),
  list(name = "ce",            block = "meteorology", lo = 0.0011, hi = 0.0016),
  list(name = "coef_mix_conv", block = "mixing",      lo = 0.10,   hi = 0.30)
)
param_names <- vapply(params, function(p) p$name, character(1))

# -----------------------------------------------------------------------------
# NML helpers (block-aware, so duplicated parameter names are set in the
# correct namelist block)
# -----------------------------------------------------------------------------
read_nml_text  <- function(file) readLines(file)
write_nml_text <- function(lines, file) writeLines(lines, file)

# Set `param` inside namelist `block` (e.g. "meteorology"). Returns the lines.
set_param_in_block <- function(lines, block, param, value) {
  block_start <- grep(paste0("^\\s*&", block, "\\b"), lines)
  if (length(block_start) == 0) return(lines)
  block_start <- block_start[1]
  # block ends at the next line that is a single "/"
  rest <- lines[(block_start + 1):length(lines)]
  block_end <- block_start + which(grepl("^\\s*/\\s*$", rest))[1]

  pattern <- paste0("^\\s*", param, "\\s*=")
  idx <- grep(pattern, lines[block_start:block_end])
  if (length(idx) > 0) {
    target <- block_start - 1 + idx[1]
    indent <- sub("\\S.*$", "", lines[target])
    lines[target] <- sprintf("%s%s = %g", indent, param, value)
  }
  lines
}

apply_params <- function(lines, values) {
  for (i in seq_along(params)) {
    p <- params[[i]]
    lines <- set_param_in_block(lines, p$block, p$name, values[i])
  }
  # Keep the sensible-heat coefficient equal to the latent-heat coefficient.
  if ("ce" %in% param_names) {
    ce_val <- values[match("ce", param_names)]
    lines <- set_param_in_block(lines, "meteorology", "ch", ce_val)
  }
  lines
}

run_glm <- function() {
  system2(GLM_EXEC, args = c("--nml", NML_FILE),
          stdout = FALSE, stderr = FALSE)
}

# -----------------------------------------------------------------------------
# Latin Hypercube Sampling (base R, no external dependency)
# -----------------------------------------------------------------------------
latin_hypercube <- function(n, lo, hi) {
  k <- length(lo)
  out <- matrix(0, n, k)
  for (j in seq_len(k)) {
    strata <- (sample(seq_len(n)) - 1 + runif(n)) / n  # one point per stratum
    out[, j] <- lo[j] + strata * (hi[j] - lo[j])
  }
  colnames(out) <- param_names
  out
}

# -----------------------------------------------------------------------------
# Extract modeled temperatures at observation depths/times
# -----------------------------------------------------------------------------
get_modeled_temps <- function(obs, nc_file) {
  nc <- nc_open(nc_file)
  time_var <- ncvar_get(nc, "time")
  time_units <- ncatt_get(nc, "time", "units")$value
  origin <- sub(".*since ", "", time_units)
  mod_time <- as.POSIXct(time_var * 3600, origin = origin, tz = "UTC")
  temp <- ncvar_get(nc, "temp")  # [z, time]
  z <- ncvar_get(nc, "z")        # [z, time]
  nc_close(nc)

  modeled <- rep(NA_real_, nrow(obs))
  for (i in seq_len(nrow(obs))) {
    time_diff <- abs(difftime(mod_time, obs$date[i], units = "hours"))
    t_idx <- which.min(time_diff)
    if (min(time_diff) > 12) next                 # same-day match (daily output)

    prof_temp <- temp[, t_idx]
    prof_z <- z[, t_idx]
    valid <- !is.na(prof_temp) & !is.na(prof_z) & prof_z > 0
    if (sum(valid) < 2) next

    prof_temp <- prof_temp[valid]
    prof_z <- prof_z[valid]
    prof_depth <- max(prof_z) - prof_z            # height above bottom -> depth
    ord <- order(prof_depth)
    prof_depth <- prof_depth[ord]
    prof_temp <- prof_temp[ord]

    if (obs$depth[i] >= min(prof_depth) && obs$depth[i] <= max(prof_depth)) {
      modeled[i] <- approx(prof_depth, prof_temp, obs$depth[i])$y
    }
  }
  modeled
}

# -----------------------------------------------------------------------------
# Skill metrics
# -----------------------------------------------------------------------------
metrics <- function(observed, modeled) {
  ok <- is.finite(observed) & is.finite(modeled)
  o <- observed[ok]; m <- modeled[ok]
  if (length(o) < 10) {
    return(list(rmse = Inf, bias = NA, nse = NA, r2 = NA, mae = NA, n = length(o)))
  }
  resid <- m - o
  list(
    rmse = sqrt(mean(resid^2)),
    bias = mean(resid),
    nse  = 1 - sum(resid^2) / sum((o - mean(o))^2),
    r2   = cor(o, m)^2,
    mae  = mean(abs(resid)),
    n    = length(o)
  )
}

# Metrics by depth layer
layered_metrics <- function(obs) {
  layers <- list(
    "Epilimnion (0-5m)"   = c(0, 5),
    "Metalimnion (5-15m)" = c(5, 15),
    "Hypolimnion (>15m)"  = c(15, 1000)
  )
  for (lab in names(layers)) {
    r <- layers[[lab]]
    sel <- obs$depth >= r[1] & obs$depth < r[2]
    mm <- metrics(obs$observed[sel], obs$modeled[sel])
    cat(sprintf("    %-22s RMSE=%.2f  Bias=%+.2f  NSE=%.2f  n=%d\n",
                lab, mm$rmse, mm$bias, mm$nse, mm$n))
  }
}

# =============================================================================
# Load observations and split into calibration / validation sets
# =============================================================================
cat("Loading observations...\n")
obs <- read.csv(OBS_FILE)
obs$date <- as.POSIXct(obs$date, format = "%m/%d/%Y", tz = "UTC")
obs <- obs[, c("date", "depth", "Water_temperature")]
names(obs)[3] <- "observed"

cal_obs <- obs[obs$date >= CAL_START & obs$date <= CAL_END, ]
val_obs <- obs[obs$date >= VAL_START & obs$date <= VAL_END, ]
cat(sprintf("  Calibration: %s to %s  (%d obs, %d profiles)\n",
            as.Date(CAL_START), as.Date(CAL_END),
            nrow(cal_obs), length(unique(cal_obs$date))))
cat(sprintf("  Validation:  %s to %s  (%d obs, %d profiles)\n",
            as.Date(VAL_START), as.Date(VAL_END),
            nrow(val_obs), length(unique(val_obs$date))))

# =============================================================================
# Latin Hypercube search over the calibration period
# =============================================================================
lo <- vapply(params, function(p) p$lo, numeric(1))
hi <- vapply(params, function(p) p$hi, numeric(1))
design <- latin_hypercube(N_SAMPLES, lo, hi)

nml_orig <- read_nml_text(NML_FILE)

cat(sprintf("\n=== LHS calibration: %d samples, %d parameters ===\n",
            N_SAMPLES, length(params)))

results <- data.frame()
for (s in seq_len(N_SAMPLES)) {
  vals <- design[s, ]
  write_nml_text(apply_params(nml_orig, vals), NML_FILE)
  run_glm()
  if (!file.exists(NC_FILE)) next

  cal_eval <- cal_obs
  cal_eval$modeled <- get_modeled_temps(cal_eval, NC_FILE)
  mm <- metrics(cal_eval$observed, cal_eval$modeled)

  row <- data.frame(sample = s, t(vals),
                    rmse = mm$rmse, bias = mm$bias,
                    nse = mm$nse, r2 = mm$r2, mae = mm$mae, n = mm$n)
  results <- rbind(results, row)

  if (s %% 10 == 0 || s == 1) {
    cat(sprintf("  [%3d/%d] best RMSE so far = %.3f C\n",
                s, N_SAMPLES, min(results$rmse, na.rm = TRUE)))
  }
}

results <- results[order(results$rmse), ]
write.csv(results, file.path(PROCESSED_DIR, "calibration_lhs_results.csv"),
          row.names = FALSE)

best <- results[1, ]
cat("\n=== Best parameter set (lowest calibration RMSE) ===\n")
for (nm in param_names) cat(sprintf("  %-14s = %g\n", nm, best[[nm]]))
cat(sprintf("  Calibration: RMSE=%.3f C  Bias=%+.3f C  NSE=%.3f  R2=%.3f  n=%d\n",
            best$rmse, best$bias, best$nse, best$r2, best$n))

# =============================================================================
# Apply best parameters, run once, evaluate on BOTH periods
# =============================================================================
write_nml_text(apply_params(nml_orig, unlist(best[param_names])), NML_FILE)
run_glm()

cal_obs$modeled <- get_modeled_temps(cal_obs, NC_FILE)
val_obs$modeled <- get_modeled_temps(val_obs, NC_FILE)
cal_m <- metrics(cal_obs$observed, cal_obs$modeled)
val_m <- metrics(val_obs$observed, val_obs$modeled)

cat("\n=== Final skill (calibrated parameters) ===\n")
cat("\n  CALIBRATION period:\n")
cat(sprintf("    Overall RMSE=%.3f  Bias=%+.3f  NSE=%.3f  R2=%.3f  n=%d\n",
            cal_m$rmse, cal_m$bias, cal_m$nse, cal_m$r2, cal_m$n))
layered_metrics(cal_obs)

cat("\n  VALIDATION period (independent):\n")
cat(sprintf("    Overall RMSE=%.3f  Bias=%+.3f  NSE=%.3f  R2=%.3f  n=%d\n",
            val_m$rmse, val_m$bias, val_m$nse, val_m$r2, val_m$n))
layered_metrics(val_obs)

# Save the paired observed/modeled comparison and a compact summary
write.csv(rbind(
  cbind(period = "calibration", cal_obs),
  cbind(period = "validation",  val_obs)
), file.path(PROCESSED_DIR, "calibration_comparison.csv"), row.names = FALSE)

summary_df <- data.frame(
  period = c("calibration", "validation"),
  rmse = c(cal_m$rmse, val_m$rmse),
  bias = c(cal_m$bias, val_m$bias),
  nse  = c(cal_m$nse,  val_m$nse),
  r2   = c(cal_m$r2,   val_m$r2),
  mae  = c(cal_m$mae,  val_m$mae),
  n    = c(cal_m$n,    val_m$n)
)
write.csv(summary_df, file.path(PROCESSED_DIR, "calibration_summary.csv"),
          row.names = FALSE)

cat("\nBest parameters written to", NML_FILE, "\n")
cat("Results saved to", PROCESSED_DIR, "\n")
