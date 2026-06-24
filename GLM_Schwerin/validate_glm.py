#!/usr/bin/env python3
"""
GLM model validation: compare simulated vs observed temperature profiles.

Steps:
1. Load GLM output (NetCDF).
2. Load observed temperature profiles.
3. Calculate validation metrics (RMSE, bias, R-squared).
4. Generate comparison plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date

from config import GLM_OUTPUT, OBS_FILE, RESULTS_DIR, PROCESSED_DIR


def load_glm_output(nc_path):
    """Load GLM NetCDF output and return (dates, temperature, height)."""
    ds = Dataset(nc_path, "r")
    time_var = ds.variables["time"][:]
    temp = ds.variables["temp"][:]  # (time, layers)
    z = ds.variables["z"][:]        # (time, layers)
    time_units = ds.variables["time"].units
    dates = num2date(time_var, time_units)
    dates = pd.to_datetime([d.strftime("%Y-%m-%d %H:%M:%S") for d in dates])
    ds.close()
    return dates, temp, z


def load_observations(obs_path):
    """Load observed temperature data."""
    df = pd.read_csv(obs_path)
    df["datetime"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    return df


def interpolate_model_to_obs_depth(model_z, model_temp, obs_depth):
    """Interpolate model temperature (height above bottom) to an observation depth."""
    surface_height = np.max(model_z[~np.isnan(model_z)])
    height_from_bottom = surface_height - obs_depth

    valid = ~np.isnan(model_z) & ~np.isnan(model_temp)
    z_valid = model_z[valid]
    t_valid = model_temp[valid]

    if len(z_valid) < 2 or height_from_bottom < 0:
        return np.nan
    return np.interp(height_from_bottom, z_valid, t_valid)


def find_nearest_time_idx(model_dates, obs_date):
    """Return nearest model timestep within 12 hours, else None."""
    time_diff = np.abs(model_dates - obs_date)
    idx = time_diff.argmin()
    if time_diff[idx] < pd.Timedelta(hours=12):
        return idx
    return None


def calculate_metrics(obs, mod):
    """Return n, RMSE, bias and R-squared for paired observations/model values."""
    valid = ~np.isnan(obs) & ~np.isnan(mod)
    obs_v, mod_v = obs[valid], mod[valid]
    if len(obs_v) < 2:
        return {"n": 0, "rmse": np.nan, "bias": np.nan, "r2": np.nan}
    return {
        "n": len(obs_v),
        "rmse": np.sqrt(np.mean((mod_v - obs_v) ** 2)),
        "bias": np.mean(mod_v - obs_v),
        "r2": np.corrcoef(obs_v, mod_v)[0, 1] ** 2,
    }


def main():
    print("=" * 60)
    print("GLM model validation")
    print("=" * 60)

    print("\nLoading GLM output...")
    model_dates, model_temp, model_z = load_glm_output(GLM_OUTPUT)
    print(f"  Model period: {model_dates[0]} to {model_dates[-1]}")
    print(f"  Timesteps: {len(model_dates)}")

    print("\nLoading observations...")
    obs_df = load_observations(OBS_FILE)
    print(f"  Observation period: {obs_df['datetime'].min()} to {obs_df['datetime'].max()}")
    print(f"  Profiles: {obs_df['datetime'].nunique()}")

    print("\nMatching observations to model timesteps...")
    results = []
    for _, row in obs_df.iterrows():
        idx = find_nearest_time_idx(model_dates, row["datetime"])
        if idx is None:
            continue
        mod_temp = interpolate_model_to_obs_depth(
            model_z[idx], model_temp[idx], row["depth"]
        )
        results.append({
            "date": row["datetime"],
            "depth": row["depth"],
            "obs_temp": row["Water_temperature"],
            "mod_temp": mod_temp,
        })

    results_df = pd.DataFrame(results)
    print(f"  Matched observations: {len(results_df)}")

    print("\n" + "=" * 60)
    print("VALIDATION METRICS")
    print("=" * 60)
    metrics = calculate_metrics(
        results_df["obs_temp"].values, results_df["mod_temp"].values
    )
    print(f"\n  Overall: N={metrics['n']}, RMSE={metrics['rmse']:.2f} C, "
          f"Bias={metrics['bias']:.2f} C, R2={metrics['r2']:.3f}")

    print("\n  By depth layer:")
    for d_min, d_max, label in [(0, 5, "Epilimnion (0-5m)"),
                                (5, 15, "Metalimnion (5-15m)"),
                                (15, 100, "Hypolimnion (>15m)")]:
        layer = results_df[(results_df["depth"] >= d_min) &
                           (results_df["depth"] < d_max)]
        m = calculate_metrics(layer["obs_temp"].values, layer["mod_temp"].values)
        print(f"    {label}: RMSE={m['rmse']:.2f} C, Bias={m['bias']:.2f} C, "
              f"R2={m['r2']:.3f} (n={m['n']})")

    results_df.to_csv(PROCESSED_DIR / "validation_results.csv", index=False)
    print(f"\nResults saved to {PROCESSED_DIR / 'validation_results.csv'}")

    # Plot 1: scatter + error by depth
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    scatter = ax.scatter(results_df["obs_temp"], results_df["mod_temp"],
                         c=results_df["depth"], cmap="viridis", alpha=0.6, s=20)
    ax.plot([0, 30], [0, 30], "k--", label="1:1 line")
    ax.set_xlabel("Observed temperature (C)")
    ax.set_ylabel("Modeled temperature (C)")
    ax.set_title(f"GLM validation: Lake Schwerin\n"
                 f"RMSE={metrics['rmse']:.2f} C, R2={metrics['r2']:.3f}")
    ax.legend()
    plt.colorbar(scatter, ax=ax, label="Depth (m)")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)

    ax = axes[1]
    results_df["error"] = results_df["mod_temp"] - results_df["obs_temp"]
    ax.scatter(results_df["error"], results_df["depth"], alpha=0.5, s=20)
    ax.axvline(0, color="k", linestyle="--")
    ax.set_xlabel("Model error (C)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Model error vs depth")
    ax.invert_yaxis()
    ax.set_xlim(-10, 10)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "validation_scatter.png", dpi=150)
    print(f"Plot saved to {RESULTS_DIR / 'validation_scatter.png'}")

    # Plot 2: sample profile comparison
    unique_dates = results_df["date"].unique()
    n_profiles = min(6, len(unique_dates))
    fig, axes = plt.subplots(2, 3, figsize=(12, 10))
    axes = axes.flatten()
    for i, date in enumerate(unique_dates[:n_profiles]):
        ax = axes[i]
        profile = results_df[results_df["date"] == date].sort_values("depth")
        ax.plot(profile["obs_temp"], profile["depth"], "ko-",
                label="Observed", markersize=3)
        ax.plot(profile["mod_temp"], profile["depth"], "r-",
                label="Modeled", linewidth=2)
        ax.set_xlabel("Temperature (C)")
        ax.set_ylabel("Depth (m)")
        ax.set_title(pd.Timestamp(date).strftime("%Y-%m-%d"))
        ax.invert_yaxis()
        ax.legend()
        ax.set_xlim(0, 25)
    plt.suptitle("Temperature profile comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "profile_comparison.png", dpi=150)
    print(f"Plot saved to {RESULTS_DIR / 'profile_comparison.png'}")


if __name__ == "__main__":
    main()
