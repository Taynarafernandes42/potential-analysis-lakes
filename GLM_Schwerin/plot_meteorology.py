#!/usr/bin/env python3
"""
Plot the meteorological forcing data used for the Lake Schwerin GLM run
over the simulation period (six panels, daily aggregates).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import METEO_FILE, RESULTS_DIR, SIM_START, SIM_STOP

# Load meteorology data
meteo = pd.read_csv(METEO_FILE)
meteo["time"] = pd.to_datetime(meteo["time"], format="mixed")
meteo.set_index("time", inplace=True)

# Resample to daily for cleaner plots
meteo_daily = meteo.resample("D").agg({
    "ShortWave": "mean",
    "Cloud": "mean",
    "AirTemp": "mean",
    "RelHum": "mean",
    "WindSpeed": "mean",
    "Rain": "sum",
    "AirPres": "mean",
})
meteo_daily = meteo_daily[SIM_START:SIM_STOP]

fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)
fig.suptitle("Lake Schwerin - meteorological forcing data",
             fontsize=14, fontweight="bold")

# 1. Shortwave radiation
ax = axes[0]
ax.fill_between(meteo_daily.index, meteo_daily["ShortWave"], alpha=0.7, color="orange")
ax.set_ylabel("Shortwave\n(W/m2)")
ax.set_ylim(0, None)
ax.grid(True, alpha=0.3)
mean_sw = meteo_daily["ShortWave"].mean()
ax.axhline(mean_sw, color="red", linestyle="--", alpha=0.7,
           label=f"Mean: {mean_sw:.0f} W/m2")
ax.legend(loc="upper right")

# 2. Air temperature
ax = axes[1]
ax.plot(meteo_daily.index, meteo_daily["AirTemp"], color="red", linewidth=0.8)
ax.fill_between(meteo_daily.index, meteo_daily["AirTemp"], alpha=0.3, color="red")
ax.set_ylabel("Air temp\n(C)")
ax.axhline(0, color="blue", linestyle=":", alpha=0.5)
ax.grid(True, alpha=0.3)
mean_at = meteo_daily["AirTemp"].mean()
ax.axhline(mean_at, color="darkred", linestyle="--", alpha=0.7,
           label=f"Mean: {mean_at:.1f} C")
ax.legend(loc="upper right")

# 3. Wind speed
ax = axes[2]
ax.fill_between(meteo_daily.index, meteo_daily["WindSpeed"], alpha=0.7, color="steelblue")
ax.set_ylabel("Wind speed\n(m/s)")
ax.set_ylim(0, None)
ax.grid(True, alpha=0.3)
mean_ws = meteo_daily["WindSpeed"].mean()
ax.axhline(mean_ws, color="navy", linestyle="--", alpha=0.7,
           label=f"Mean: {mean_ws:.1f} m/s")
ax.legend(loc="upper right")

# 4. Relative humidity
ax = axes[3]
ax.fill_between(meteo_daily.index, meteo_daily["RelHum"], alpha=0.7, color="green")
ax.set_ylabel("Rel. humidity\n(%)")
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)
mean_rh = meteo_daily["RelHum"].mean()
ax.axhline(mean_rh, color="darkgreen", linestyle="--", alpha=0.7,
           label=f"Mean: {mean_rh:.0f}%")
ax.legend(loc="upper right")

# 5. Cloud cover
ax = axes[4]
ax.fill_between(meteo_daily.index, meteo_daily["Cloud"], alpha=0.7, color="gray")
ax.set_ylabel("Cloud cover\n(fraction)")
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
mean_cc = meteo_daily["Cloud"].mean()
ax.axhline(mean_cc, color="black", linestyle="--", alpha=0.7,
           label=f"Mean: {mean_cc:.2f}")
ax.legend(loc="upper right")

# 6. Precipitation
ax = axes[5]
ax.bar(meteo_daily.index, meteo_daily["Rain"], width=1, alpha=0.7, color="blue")
ax.set_ylabel("Precipitation\n(mm/day)")
ax.set_ylim(0, None)
ax.grid(True, alpha=0.3)
total_rain = meteo_daily["Rain"].sum()
annual_rain = total_rain / (len(meteo_daily) / 365.25)
ax.text(0.02, 0.95,
        f"Total: {total_rain:.0f} mm\nAnnual avg: {annual_rain:.0f} mm/yr",
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

axes[-1].set_xlabel("Date")
axes[-1].xaxis.set_major_locator(mdates.YearLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[-1].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))

plt.tight_layout()
out_plot = RESULTS_DIR / "meteorology_plots.png"
plt.savefig(out_plot, dpi=150, bbox_inches="tight")
plt.close()
print(f"Meteorology plot saved to: {out_plot}")

# Summary statistics
print("\n=== Meteorological summary ===")
print(f"Shortwave radiation: {meteo_daily['ShortWave'].mean():.1f} +/- "
      f"{meteo_daily['ShortWave'].std():.1f} W/m2")
print(f"Air temperature:     {meteo_daily['AirTemp'].mean():.1f} +/- "
      f"{meteo_daily['AirTemp'].std():.1f} C")
print(f"Wind speed:          {meteo_daily['WindSpeed'].mean():.1f} +/- "
      f"{meteo_daily['WindSpeed'].std():.1f} m/s")
print(f"Relative humidity:   {meteo_daily['RelHum'].mean():.0f} +/- "
      f"{meteo_daily['RelHum'].std():.0f} %")
print(f"Cloud cover:         {meteo_daily['Cloud'].mean():.2f} +/- "
      f"{meteo_daily['Cloud'].std():.2f}")
print(f"Total precipitation: {meteo_daily['Rain'].sum():.0f} mm over "
      f"{len(meteo_daily) / 365.25:.1f} years")
