#!/usr/bin/env python3
"""
Validation plots for the calibrated Lake Schwerin GLM run.

Produces a 4-panel figure (scatter, error by depth, surface and deep time
series) and saves the paired observed/modeled comparison table.
"""

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from netCDF4 import Dataset

from config import GLM_OUTPUT, OBS_FILE, RESULTS_DIR, PROCESSED_DIR

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Load model output
# -----------------------------------------------------------------------------
nc = Dataset(GLM_OUTPUT, "r")
time_hours = nc.variables["time"][:]
time_units = nc.variables["time"].units
origin = datetime.strptime(time_units.split("since ")[1], "%Y-%m-%d %H:%M:%S")
model_times = [origin + timedelta(hours=float(t)) for t in time_hours]

temp = np.ma.filled(nc.variables["temp"][:, :, 0, 0], np.nan)  # [time, z]
z = np.ma.filled(nc.variables["z"][:, :, 0, 0], np.nan)        # [time, z]
nc.close()

# -----------------------------------------------------------------------------
# Load observations
# -----------------------------------------------------------------------------
obs = pd.read_csv(OBS_FILE)
obs["date"] = pd.to_datetime(obs["date"], format="%m/%d/%Y")

# -----------------------------------------------------------------------------
# Match observations to model profiles
# -----------------------------------------------------------------------------
comparisons = []
for _, row in obs.iterrows():
    obs_time = row["date"]
    obs_depth = row["depth"]

    time_diffs = [abs((t - obs_time.to_pydatetime()).total_seconds() / 3600)
                  for t in model_times]
    t_idx = int(np.argmin(time_diffs))
    if time_diffs[t_idx] > 12:
        continue

    prof_z = z[t_idx, :]
    prof_temp = temp[t_idx, :]
    valid = ~np.isnan(prof_temp) & ~np.isnan(prof_z) & (prof_z > 0)
    if valid.sum() < 2:
        continue

    prof_z = prof_z[valid]
    prof_temp = prof_temp[valid]
    prof_depth = np.max(prof_z) - prof_z  # height above bottom -> depth

    if obs_depth >= prof_depth.min() and obs_depth <= prof_depth.max():
        order = np.argsort(prof_depth)
        mod_temp = np.interp(obs_depth, prof_depth[order], prof_temp[order])
        comparisons.append({
            "date": obs_time,
            "depth": obs_depth,
            "observed": row["Water_temperature"],
            "modeled": mod_temp,
        })

comp_df = pd.DataFrame(comparisons)
comp_df["residual"] = comp_df["modeled"] - comp_df["observed"]

rmse = np.sqrt(np.mean(comp_df["residual"] ** 2))
bias = np.mean(comp_df["residual"])
r2 = np.corrcoef(comp_df["observed"], comp_df["modeled"])[0, 1] ** 2
n = len(comp_df)

print("Calibrated model validation:")
print(f"  N comparisons: {n}")
print(f"  RMSE: {rmse:.3f} C")
print(f"  Bias: {bias:+.3f} C")
print(f"  R2: {r2:.4f}")

# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Observed vs modeled scatter
ax1 = axes[0, 0]
sc = ax1.scatter(comp_df["observed"], comp_df["modeled"],
                 c=comp_df["depth"], cmap="viridis", alpha=0.6, s=20,
                 edgecolor="none")
ax1.plot([0, 30], [0, 30], "r--", lw=2, label="1:1 line")
ax1.set_xlabel("Observed temperature (C)")
ax1.set_ylabel("Modeled temperature (C)")
ax1.set_title("Observed vs modeled temperature")
ax1.set_xlim(0, 30)
ax1.set_ylim(0, 30)
ax1.set_aspect("equal")
plt.colorbar(sc, ax=ax1, label="Depth (m)")
ax1.legend(loc="upper left")
ax1.text(0.98, 0.02,
         f"RMSE = {rmse:.2f} C\nBias = {bias:+.2f} C\nR2 = {r2:.3f}\nn = {n}",
         transform=ax1.transAxes, ha="right", va="bottom",
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# 2. Error by depth
ax2 = axes[0, 1]
depths = np.arange(0, 55, 5)
depth_rmse, depth_bias = [], []
for d in depths:
    mask = (comp_df["depth"] >= d) & (comp_df["depth"] < d + 5)
    if mask.sum() > 10:
        depth_rmse.append(np.sqrt(np.mean(comp_df.loc[mask, "residual"] ** 2)))
        depth_bias.append(np.mean(comp_df.loc[mask, "residual"]))
    else:
        depth_rmse.append(np.nan)
        depth_bias.append(np.nan)
ax2.barh(depths + 2.5, depth_rmse, height=4, alpha=0.7, label="RMSE",
         color="steelblue")
ax2.barh(depths + 2.5, depth_bias, height=2, alpha=0.9, label="Bias",
         color="coral")
ax2.axvline(0, color="black", linestyle="-", linewidth=0.5)
ax2.set_xlabel("Error (C)")
ax2.set_ylabel("Depth (m)")
ax2.set_title("Error by depth")
ax2.invert_yaxis()
ax2.legend(loc="lower right")
ax2.set_xlim(-3, 6)

# 3. Surface time series (0-2m)
ax3 = axes[1, 0]
surface_obs = comp_df[comp_df["depth"] <= 2].groupby("date").mean().reset_index()
ax3.scatter(surface_obs["date"], surface_obs["observed"], label="Observed",
            alpha=0.7, s=30, color="blue")
ax3.scatter(surface_obs["date"], surface_obs["modeled"], label="Modeled",
            alpha=0.7, s=30, color="red")
ax3.set_xlabel("Date")
ax3.set_ylabel("Temperature (C)")
ax3.set_title("Surface temperature (0-2m)")
ax3.legend(loc="upper left")
ax3.xaxis.set_major_locator(mdates.YearLocator())
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax3.grid(True, alpha=0.3)

# 4. Deep time series (30-40m)
ax4 = axes[1, 1]
deep_obs = comp_df[(comp_df["depth"] >= 30) & (comp_df["depth"] <= 40)]
deep_obs = deep_obs.groupby("date").mean().reset_index()
ax4.scatter(deep_obs["date"], deep_obs["observed"], label="Observed",
            alpha=0.7, s=30, color="blue")
ax4.scatter(deep_obs["date"], deep_obs["modeled"], label="Modeled",
            alpha=0.7, s=30, color="red")
ax4.set_xlabel("Date")
ax4.set_ylabel("Temperature (C)")
ax4.set_title("Deep temperature (30-40m)")
ax4.legend(loc="upper left")
ax4.xaxis.set_major_locator(mdates.YearLocator())
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax4.grid(True, alpha=0.3)

plt.suptitle("Lake Schwerin GLM calibration validation",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()

out_plot = RESULTS_DIR / "calibration_validation.png"
plt.savefig(out_plot, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nPlot saved to: {out_plot}")

comp_df.to_csv(PROCESSED_DIR / "validation_comparison.csv", index=False)
print(f"Comparison data saved to: {PROCESSED_DIR / 'validation_comparison.csv'}")
