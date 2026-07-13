#!/usr/bin/env python3
"""
Comprehensive GLM validation figure for Lake Schwerin.

Builds a 4-panel figure:
  1. Modeled temperature contour with observed points overlaid
  2. Modeled vs observed scatter
  3. Model error by depth
  4. Surface and deep temperature time series
and prints metrics per depth layer.
"""

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from netCDF4 import Dataset

from config import GLM_OUTPUT, OBS_FILE, RESULTS_DIR

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Load GLM output
# -----------------------------------------------------------------------------
print("Loading GLM output...")
nc = Dataset(GLM_OUTPUT, "r")
time_hours = nc.variables["time"][:]
time_units = nc.variables["time"].units
origin = datetime.strptime(time_units.split("since ")[1], "%Y-%m-%d %H:%M:%S")
model_times = np.array([origin + timedelta(hours=float(t)) for t in time_hours])

temp_raw = nc.variables["temp"][:]
z_raw = nc.variables["z"][:]

if temp_raw.ndim == 4:
    temp = np.ma.filled(temp_raw[:, :, 0, 0], np.nan)  # [time, z]
    z = np.ma.filled(z_raw[:, :, 0, 0], np.nan)
else:  # 2-D
    temp = np.ma.filled(temp_raw, np.nan)
    z = np.ma.filled(z_raw, np.nan)
    if temp.shape[0] < temp.shape[1]:  # [z, time] -> transpose
        temp = temp.T
        z = z.T
nc.close()

print(f"  Model period: {model_times[0]} to {model_times[-1]}")
print(f"  Timesteps: {len(model_times)}, Layers: {temp.shape[1]}")

# -----------------------------------------------------------------------------
# Load observations
# -----------------------------------------------------------------------------
print("Loading observations...")
obs = pd.read_csv(OBS_FILE)
obs["date"] = pd.to_datetime(obs["date"], format="%m/%d/%Y")
print(f"  Profiles: {obs['date'].nunique()}")

# -----------------------------------------------------------------------------
# Build temperature matrix for the contour plot
# -----------------------------------------------------------------------------
max_depth = 50
depth_grid = np.linspace(0, max_depth, 100)
temp_matrix = np.full((len(depth_grid), len(model_times)), np.nan)

print("Building temperature matrix...")
for t_idx in range(len(model_times)):
    prof_z = z[t_idx, :]
    prof_temp = temp[t_idx, :]
    valid = ~np.isnan(prof_temp) & ~np.isnan(prof_z) & (prof_z > 0)
    if valid.sum() < 2:
        continue
    prof_z = prof_z[valid]
    prof_temp = prof_temp[valid]
    prof_depth = np.max(prof_z) - prof_z
    order = np.argsort(prof_depth)
    prof_depth = prof_depth[order]
    prof_temp = prof_temp[order]
    for d_idx, d in enumerate(depth_grid):
        if prof_depth.min() <= d <= prof_depth.max():
            temp_matrix[d_idx, t_idx] = np.interp(d, prof_depth, prof_temp)

# -----------------------------------------------------------------------------
# Match observations to model
# -----------------------------------------------------------------------------
print("Matching observations...")
comparisons = []
for _, row in obs.iterrows():
    obs_time = row["date"]
    obs_depth = row["depth"]
    time_diffs = np.array([abs((t - obs_time.to_pydatetime()).total_seconds() / 3600)
                           for t in model_times])
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
    prof_depth = np.max(prof_z) - prof_z
    if prof_depth.min() <= obs_depth <= prof_depth.max():
        order = np.argsort(prof_depth)
        mod_temp = np.interp(obs_depth, prof_depth[order], prof_temp[order])
        comparisons.append({
            "date": obs_time,
            "depth": obs_depth,
            "observed": row["Water_temperature"],
            "modeled": mod_temp,
        })

comp_df = pd.DataFrame(comparisons)
comp_df["error"] = comp_df["modeled"] - comp_df["observed"]

rmse = np.sqrt(np.mean(comp_df["error"] ** 2))
bias = np.mean(comp_df["error"])
r2 = np.corrcoef(comp_df["observed"], comp_df["modeled"])[0, 1] ** 2
n = len(comp_df)
print(f"\nMetrics: RMSE={rmse:.2f} C, Bias={bias:+.2f} C, R2={r2:.3f}, n={n}")


def get_model_temp_at_depth(target_depth):
    """Time series of modeled temperature interpolated to a fixed depth."""
    temps = []
    for t_idx in range(len(model_times)):
        prof_z = z[t_idx, :]
        prof_temp = temp[t_idx, :]
        valid = ~np.isnan(prof_temp) & ~np.isnan(prof_z) & (prof_z > 0)
        if valid.sum() < 2:
            temps.append(np.nan)
            continue
        prof_z, prof_temp = prof_z[valid], prof_temp[valid]
        prof_depth = np.max(prof_z) - prof_z
        order = np.argsort(prof_depth)
        if prof_depth.min() <= target_depth <= prof_depth.max():
            temps.append(np.interp(target_depth, prof_depth[order], prof_temp[order]))
        else:
            temps.append(np.nan)
    return np.array(temps)


model_1m = get_model_temp_at_depth(1)
model_30m = get_model_temp_at_depth(30)
obs_surface = obs[(obs["depth"] >= 0.5) & (obs["depth"] <= 1.5)]
obs_deep = obs[(obs["depth"] >= 29) & (obs["depth"] <= 31)]

# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
print("Creating figure...")
fig = plt.figure(figsize=(16, 14))

# 1. Temperature contour with observations
ax1 = fig.add_subplot(3, 1, 1)
plot_times = mdates.date2num(model_times)
levels = np.linspace(0, 32, 33)
cf = ax1.contourf(plot_times, depth_grid, temp_matrix, levels=levels,
                  cmap="RdYlBu_r", extend="both")
for date in obs["date"].unique():
    obs_day = obs[obs["date"] == date]
    ax1.scatter([mdates.date2num(date)] * len(obs_day), obs_day["depth"],
                c="white", s=10, edgecolor="black", linewidth=0.3, alpha=0.7)
ax1.set_ylabel("Depth (m)")
ax1.set_ylim(max_depth, 0)
ax1.set_title("Lake Schwerin: modeled temperature with observed points overlaid")
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.colorbar(cf, ax=ax1, label="Temperature (C)", shrink=0.8)

# 2. Model vs observed scatter
ax2 = fig.add_subplot(3, 2, 3)
sc = ax2.scatter(comp_df["observed"], comp_df["modeled"], c=comp_df["depth"],
                 cmap="viridis", alpha=0.5, s=15, edgecolor="none")
ax2.plot([0, 30], [0, 30], "k--", lw=1.5, label="1:1 line")
ax2.set_xlabel("Observed temperature (C)")
ax2.set_ylabel("Modeled temperature (C)")
ax2.set_title("Model vs observed temperature")
ax2.set_xlim(0, 30)
ax2.set_ylim(0, 30)
ax2.set_aspect("equal")
plt.colorbar(sc, ax=ax2, label="Depth (m)")
ax2.text(0.05, 0.95,
         f"RMSE = {rmse:.2f} C\nBias = {bias:+.2f} C\nR2 = {r2:.3f}\nn = {n}",
         transform=ax2.transAxes, va="top",
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# 3. Error by depth
ax3 = fig.add_subplot(3, 2, 4)
depth_bins = np.arange(0, 55, 1)
error_bins = np.linspace(-15, 15, 61)
H, _, _ = np.histogram2d(comp_df["error"], comp_df["depth"],
                         bins=[error_bins, depth_bins])
H = H.T
extent = [error_bins[0], error_bins[-1], depth_bins[-1], depth_bins[0]]
ax3.imshow(H, extent=extent, aspect="auto", cmap="Blues",
           norm=Normalize(vmin=0, vmax=np.percentile(H, 95)))
depth_centers = (depth_bins[:-1] + depth_bins[1:]) / 2
mean_errors = []
for i in range(len(depth_bins) - 1):
    mask = (comp_df["depth"] >= depth_bins[i]) & (comp_df["depth"] < depth_bins[i + 1])
    mean_errors.append(comp_df.loc[mask, "error"].mean() if mask.sum() else np.nan)
ax3.scatter(mean_errors, depth_centers, c="black", s=30, zorder=5,
            label="Mean error")
ax3.axvline(0, color="black", linestyle="--", linewidth=1)
ax3.set_xlabel("Error (model - observed) (C)")
ax3.set_ylabel("Depth (m)")
ax3.set_title("Model error by depth")
ax3.set_xlim(-15, 15)
ax3.legend(loc="lower right")

# 4. Time series
ax4 = fig.add_subplot(3, 1, 3)
ax4.plot(model_times, model_1m, "b-", linewidth=0.8, alpha=0.8, label="Model (1m)")
ax4.plot(model_times, model_30m, "g-", linewidth=0.8, alpha=0.8, label="Model (30m)")
ax4.scatter(obs_surface["date"], obs_surface["Water_temperature"], c="red", s=25,
            marker="o", label="Observed (0.5-1.5m)", zorder=5)
ax4.scatter(obs_deep["date"], obs_deep["Water_temperature"], c="lime", s=25,
            marker="o", edgecolor="green", label="Observed (29-31m)", zorder=5)
ax4.set_xlabel("Date")
ax4.set_ylabel("Temperature (C)")
ax4.set_title("Time series: surface (1m) and deep (30m) temperatures")
ax4.legend(loc="upper right", fontsize=9)
ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax4.set_ylim(0, 35)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
out_plot = RESULTS_DIR / "comprehensive_validation.png"
plt.savefig(out_plot, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nPlot saved to: {out_plot}")

print("\n=== Metrics by depth layer ===")
for label, (d_min, d_max) in [("Epilimnion (0-5m)", (0, 5)),
                              ("Metalimnion (5-15m)", (5, 15)),
                              ("Hypolimnion (>15m)", (15, 100))]:
    mask = (comp_df["depth"] >= d_min) & (comp_df["depth"] < d_max)
    if mask.sum() > 0:
        layer_rmse = np.sqrt(np.mean(comp_df.loc[mask, "error"] ** 2))
        layer_bias = np.mean(comp_df.loc[mask, "error"])
        print(f"  {label}: RMSE={layer_rmse:.2f} C, Bias={layer_bias:+.2f} C, "
              f"n={mask.sum()}")
