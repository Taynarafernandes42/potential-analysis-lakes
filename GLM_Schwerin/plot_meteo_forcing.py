#!/usr/bin/env python3
"""
Publication-ready plot of the long-term meteorological forcing
(DWD station 4625 "Schwerin", lat 53.64N, lon 11.39E, 59 m a.s.l.).

Five panels with daily means, monthly means and annual means:
air temperature, relative humidity, wind speed, air pressure, cloud cover.
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from config import METEO_LONGTERM_FILE, RESULTS_DIR

warnings.filterwarnings("ignore")

# --- Publication style --------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.5,
    "grid.color": "#BBBBBB",
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# --- Load data ----------------------------------------------------------------
print("Loading meteo data ...")
df = pd.read_csv(METEO_LONGTERM_FILE, parse_dates=["time"], index_col="time")
df = df.loc[:"2023-12-31 23:00"]
print(f"  Rows: {len(df):,}  ({df.index[0].date()} -> {df.index[-1].date()})")

# --- Aggregates ---------------------------------------------------------------
daily = pd.DataFrame({
    "AirTemp":   df["AirTemp"].resample("D").mean(),
    "RelHum":    df["RelHum"].resample("D").mean(),
    "WindSpeed": df["WindSpeed"].resample("D").mean(),
    "AirPres":   df["AirPres"].resample("D").mean(),
    "Cloud":     df["Cloud"].resample("D").mean(),
    "Rain":      df["Rain"].resample("D").sum(),
})

monthly = pd.DataFrame({
    "AirTemp":   df["AirTemp"].resample("M").mean(),
    "RelHum":    df["RelHum"].resample("M").mean(),
    "WindSpeed": df["WindSpeed"].resample("M").mean(),
    "AirPres":   df["AirPres"].resample("M").mean(),
    "Cloud":     df["Cloud"].resample("M").mean(),
})

annual_temp = df["AirTemp"].resample("A").mean()
annual_rad = df["ShortWave"].resample("A").mean()
annual_rain = df["Rain"].resample("A").sum()
print(f"  Mean air temperature: {annual_temp.mean():.1f} C")
print(f"  Mean shortwave radiation: {annual_rad.mean():.1f} W/m2")
print(f"  Mean annual precipitation: {annual_rain.mean():.0f} mm/yr")

# --- Figure -------------------------------------------------------------------
fig, axes = plt.subplots(
    5, 1, figsize=(7.2, 8.0), sharex=True,
    gridspec_kw={"hspace": 0.08, "height_ratios": [1.15, 1, 1, 1, 0.85]},
)

DAILY_COLOR = "#9ECAE1"
MONTHLY_COLOR = "#08519C"
PANEL_LABELS = "abcde"


def shade_summer(ax, alpha=0.07):
    for y in range(int(df.index[0].year), int(df.index[-1].year) + 1):
        ax.axvspan(pd.Timestamp(f"{y}-06-01"), pd.Timestamp(f"{y}-08-31"),
                   color="#FFF0AA", alpha=alpha, linewidth=0, zorder=0)


# (a) Air temperature
ax = axes[0]
ax.fill_between(daily.index, daily["AirTemp"], alpha=0.35, color=DAILY_COLOR,
                linewidth=0, label="Daily mean")
ax.plot(monthly.index, monthly["AirTemp"], color=MONTHLY_COLOR, linewidth=1.2,
        label="Monthly mean")
shade_summer(ax)
ann_t = annual_temp.copy()
ann_t.index = ann_t.index - pd.DateOffset(months=6)
ax.plot(ann_t.index, ann_t.values, color="#CC0000", linewidth=0.7,
        linestyle="--", alpha=0.7, label="Annual mean")
ax.set_ylabel("Air temp.\n(C)")
ax.axhline(0, color="k", linewidth=0.5, linestyle=":", alpha=0.6)

# (b) Relative humidity
ax = axes[1]
ax.fill_between(daily.index, daily["RelHum"], alpha=0.35, color=DAILY_COLOR, linewidth=0)
ax.plot(monthly.index, monthly["RelHum"], color=MONTHLY_COLOR, linewidth=1.2)
ax.set_ylabel("Rel. humidity\n(%)")
ax.set_ylim(40, 100)

# (c) Wind speed
ax = axes[2]
ax.fill_between(daily.index, daily["WindSpeed"], alpha=0.35, color=DAILY_COLOR, linewidth=0)
ax.plot(monthly.index, monthly["WindSpeed"], color=MONTHLY_COLOR, linewidth=1.2)
ax.set_ylabel("Wind speed\n(m/s)")
ax.set_ylim(bottom=0)

# (d) Air pressure
ax = axes[3]
ax.fill_between(daily.index, daily["AirPres"], alpha=0.35, color=DAILY_COLOR, linewidth=0)
ax.plot(monthly.index, monthly["AirPres"], color=MONTHLY_COLOR, linewidth=1.2)
ax.set_ylabel("Air pressure\n(hPa)")
p_lo = np.floor(daily["AirPres"].min() / 5) * 5
p_hi = np.ceil(daily["AirPres"].max() / 5) * 5
ax.set_ylim(p_lo, p_hi)

# (e) Cloud cover
ax = axes[4]
ax.fill_between(daily.index, daily["Cloud"], alpha=0.35, color=DAILY_COLOR, linewidth=0)
ax.plot(monthly.index, monthly["Cloud"], color=MONTHLY_COLOR, linewidth=1.2)
ax.set_ylabel("Cloud cover\n(fraction)")
ax.set_ylim(0, 1)

# Shared x-axis
for ax in axes:
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.margins(x=0.005)
    ax.yaxis.set_label_coords(-0.075, 0.5)
axes[-1].set_xlabel("Year")

for i, ax in enumerate(axes):
    ax.text(0.012, 0.90, f"({PANEL_LABELS[i]})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top")

custom_lines = [
    Line2D([0], [0], color=DAILY_COLOR, linewidth=6, alpha=0.6),
    Line2D([0], [0], color=MONTHLY_COLOR, linewidth=1.8),
    Line2D([0], [0], color="#CC0000", linewidth=1, linestyle="--"),
]
axes[0].legend(custom_lines, ["Daily mean", "Monthly mean", "Annual mean"],
               loc="upper right", ncol=3, framealpha=0.9, borderpad=0.4,
               handlelength=1.4, handletextpad=0.5, columnspacing=0.8)

fig.suptitle(
    "Meteorological forcing - Lake Schwerin\n"
    "DWD station 4625 'Schwerin' (53.64N, 11.39E, 59 m a.s.l.)",
    fontsize=9, y=0.995, va="top",
)

out = RESULTS_DIR / "schwerin_meteo_forcing.png"
fig.savefig(out, dpi=300)
print(f"\nSaved -> {out}")
plt.close()
