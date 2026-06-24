"""
Central configuration for the Lake Schwerin GLM Python scripts.

All paths are resolved relative to this file, so the project is portable.
The GLM executable is read from the GLM_EXEC environment variable
(falls back to "glm" on PATH).
"""

import os
from pathlib import Path

# --- Directories -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
RAW_DIR = DATA_DIR / "Raw"
PROCESSED_DIR = DATA_DIR / "Processed"
RESULTS_DIR = DATA_DIR / "Results"
OUTPUT_DIR = BASE_DIR / "output"

for _d in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Model / executable ------------------------------------------------------
GLM_EXEC = os.environ.get("GLM_EXEC", "glm")
NML_FILE = BASE_DIR / "glm4.nml"

# --- Input data --------------------------------------------------------------
GLM_OUTPUT = OUTPUT_DIR / "output.nc"      # GLM NetCDF output
OBS_FILE = RAW_DIR / "Temp.csv"            # Observed temperature profiles
METEO_FILE = RAW_DIR / "meteo_glm.csv"     # Meteorological forcing (sim period)
METEO_LONGTERM_FILE = RAW_DIR / "meteo_glm_1994_2023.csv"  # Long-term forcing

# --- Simulation period -------------------------------------------------------
SIM_START = "2016-04-18"
SIM_STOP = "2021-03-31"
