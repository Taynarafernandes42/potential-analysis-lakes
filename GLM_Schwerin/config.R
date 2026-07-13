# Central configuration for the Lake Schwerin GLM R scripts.
#
# Paths are relative to the project root (where this file lives), so the
# project is portable. The GLM executable is read from the GLM_EXEC
# environment variable and falls back to "glm" on the system PATH.

# --- Model / executable ------------------------------------------------------
GLM_EXEC <- Sys.getenv("GLM_EXEC", unset = "glm")
NML_FILE <- "glm4.nml"
NC_FILE  <- file.path("output", "output.nc")

# --- Input data --------------------------------------------------------------
OBS_FILE <- file.path("Data", "Raw", "Temp.csv")

# --- Split-sample calibration / validation periods ---------------------------
# The model is run continuously over the full simulation period; observations
# are partitioned by date so that calibration and validation use independent
# data (no information leakage between the two).
CAL_START <- as.POSIXct("2016-04-18", tz = "UTC")
CAL_END   <- as.POSIXct("2019-12-31", tz = "UTC")
VAL_START <- as.POSIXct("2020-01-01", tz = "UTC")
VAL_END   <- as.POSIXct("2021-03-31", tz = "UTC")

# --- Output directories ------------------------------------------------------
RESULTS_DIR   <- file.path("Data", "Results")
PROCESSED_DIR <- file.path("Data", "Processed")

dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(PROCESSED_DIR, recursive = TRUE, showWarnings = FALSE)
