# Lake Schwerin – GLM Model Scripts

Scripts to set up, calibrate, validate and visualise a
[General Lake Model (GLM)](https://aquaticecodynamics.github.io/glm-workbook/)
one-dimensional hydrodynamic simulation of **Lake Schwerin (Schweriner See)**,
Germany.

The workflow:

1. **Run** GLM with the configuration in `glm4.nml`.
2. **Calibrate** key parameters (wind factor, light extinction, mixing and
   heat-exchange coefficients) against observed temperature profiles.
3. **Validate** the calibrated run and compute error metrics (RMSE, bias, R²).
4. **Plot** the meteorological forcing and validation figures.

---

## Project structure

```
.
├── glm4.nml                      # GLM configuration (model setup, morphometry, forcing)
├── config.py                     # Central paths/parameters for the Python scripts
├── config.R                      # Central paths/parameters for the R scripts
├── calibrate_glm.R               # Split-sample, Latin-Hypercube calibration
├── validate_glm.py               # Validation metrics + scatter/profile plots
├── final_validation.py           # 4-panel validation figure for the calibrated run
├── comprehensive_validation.py   # Contour + scatter + error-by-depth + time series
├── plot_meteorology.py           # Meteorological forcing over the simulation period
├── plot_meteo_forcing.py         # Publication-ready long-term forcing figure
├── requirements.txt
└── Data/
    ├── Raw/                      # Input data (place here, not committed)
    ├── Processed/                # Intermediate outputs (CSV)
    └── Results/                  # Figures
```

---

## Requirements

**GLM** – a compiled GLM executable (v4). Point the scripts to it with the
`GLM_EXEC` environment variable; otherwise `glm` on your `PATH` is used:

```bash
export GLM_EXEC=/path/to/glm
```

**Python** (3.9+):

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS  (.venv\Scripts\activate on Windows)
pip install -r requirements.txt
```

**R** (4.x) with packages: `ncdf4`, `dplyr`, `tidyr`. The calibration script
(`calibrate_glm.R`) uses only base R plus `ncdf4`.

---

## Input data

Place the following files in `Data/Raw/` before running. They are **not**
included in this repository.

| File | Description |
|------|-------------|
| `Temp.csv` | Observed temperature profiles (`date`, `depth`, `Water_temperature`) |
| `meteo_glm.csv` | Meteorological forcing for the simulation period |
| `meteo_glm_1994_2023.csv` | Long-term meteorological forcing (for `plot_meteo_forcing.py`) |
| `inflow_heatpump.csv`, `inflow1.csv` | Inflow time series (referenced by `glm4.nml`) |
| `outflow_0.csv`, `outflow1.csv` | Outflow time series (referenced by `glm4.nml`) |

Observation CSV format expected by the calibration/validation scripts:

```
date,depth,Water_temperature
04/18/2016,0.5,7.5
...
```

---

## Usage

Run from the project root so the relative paths in `config.py` / `config.R`
and `glm4.nml` resolve correctly.

```bash
# 1. Run GLM
$GLM_EXEC --nml glm4.nml

# 2. Calibrate (split-sample, Latin Hypercube)
Rscript calibrate_glm.R

# 3. Validate
python validate_glm.py
python final_validation.py
python comprehensive_validation.py

# 4. Plot forcing
python plot_meteorology.py
python plot_meteo_forcing.py
```

The calibration script modifies `glm4.nml` in place and writes result tables to
`Data/Processed/`. Validation and plotting scripts write figures to
`Data/Results/`.

---

## Calibration approach

The model is calibrated with a **split-sample** design and a **joint**
parameter search:

* **Split-sample.** Parameters are fitted against observations from the
  calibration period and the resulting parameter set is then evaluated against
  an **independent** validation period that is never used during fitting. The
  periods are defined in `config.R` (`CAL_START`/`CAL_END`,
  `VAL_START`/`VAL_END`; default: calibrate 2016-04-18 to 2019-12-31, validate
  2020-01-01 to 2021-03-31). A single continuous GLM run covers both periods;
  only the *observations* are partitioned, so there is no information leakage.
* **Joint search via Latin Hypercube Sampling (LHS).** All parameters are
  varied simultaneously over physically plausible ranges, so parameter
  interactions are captured. This avoids the local optima and compensating
  errors that staged, one-parameter-at-a-time grid searches can produce. The
  LHS design is reproducible (fixed random seed).
* **Physically bounded parameters.** Calibrated parameters and ranges:
  `wind_factor` (0.70-1.00), `Kw` (0.30-0.80 m^-1), `ce` = `ch`
  (0.0011-0.0016), `coef_mix_conv` (0.10-0.30). The bulk transfer coefficients
  are kept within their physically plausible range rather than pushed to absorb
  bias.
* **Multiple skill metrics.** RMSE (the selection criterion), bias,
  Nash-Sutcliffe efficiency (NSE), R^2 and MAE are reported for the whole water
  column and per depth layer (epilimnion 0-5 m, metalimnion 5-15 m,
  hypolimnion >15 m), for both periods.

The best parameter set (lowest calibration RMSE) is written back to `glm4.nml`
and the full results are saved to `Data/Processed/`
(`calibration_lhs_results.csv`, `calibration_summary.csv`,
`calibration_comparison.csv`).

### Known limitations

* **Persistent surface warm bias.** During the stratified season the model is
  systematically warmer than observed in the epilimnion (~+4 C on average).
  This bias remains within the physically plausible parameter ranges and is
  reported rather than removed by adopting non-physical coefficient values.
* **Objective weighting.** RMSE is pooled over all observation points, so the
  more frequently sampled depths (the hypolimnion) carry more weight in the
  selection criterion. Per-layer metrics are reported alongside so that
  layer-specific performance is transparent.
* **Observation coverage.** Profiles are concentrated in spring-autumn; winter
  (Dec-Mar) is sparsely sampled within the simulation window.

---

## Configuration

All paths and shared settings live in `config.py` (Python) and `config.R` (R).
Model parameters are set in `glm4.nml`. Key entries:

| Parameter | Location | Description |
|-----------|----------|-------------|
| `wind_factor` | `&meteorology` | Wind speed scaling (main calibration target) |
| `Kw` | `&glm_setup`, `&light` | Light extinction coefficient |
| `coef_mix_conv`, `coef_wind_stir` | `&mixing` | Surface mixing coefficients |
| `ce`, `ch` | `&meteorology` | Bulk transfer coefficients |
| `lw_factor` | `&meteorology` | Longwave radiation factor |
| simulation period | `&time` | `start` / `stop` |

---

## License

Add your license here.

## Citation

If you use these scripts in academic work, please cite accordingly.
