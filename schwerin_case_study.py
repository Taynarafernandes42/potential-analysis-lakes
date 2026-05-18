"""
Combined Analysis: Lake Thermal Potential vs. District Heating Demand
=====================================================================
1. Technical potential (water side) – static & dynamic
2. Heating demand profile (BDEW) scaled to static potential
3. Overlay plot for a year with outages
4. Monthly coverage heatmap (all years)
5. Hourly deficit heatmap (all years)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LinearSegmentedColormap
import warnings
import holidays as hol
from demandlib import bdew

from config import Config

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

cfg = Config()

LAKE_ID = 13463
OUTPUT_DIR = Path(cfg.results_dir) / "CaseStudy_Schwerin" / "CombinedAnalysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAKE_TEMP_CSV = Path("Data/Raw/schwerin_temperature_13.5m_hourly.csv")
AIR_TEMP_CSV = Path(
    r"D:\Users\pau07368\PycharmProjects\potential-analysis-lakes-1"
    r"\Data\Raw\schwerin_airtemp_1994_2020.csv"
)

# Physical / technical parameters
RHO = cfg.density_water
CP = cfg.cp_water
DELTA_T_TECH = cfg.deltaT_technical_extractable
DELTA_T_HEX_MAX = 5.0
T_MIN_EXTRACTION = 3.0

# COP parameters
T_SUPPLY_C = 90.0
COP_C1 = 6.7
COP_C2 = 0.03697
COP_C3 = -0.03697
COP_C4 = 1.605e-7

# Demand profile parameters
BUILDING_TYPE = "MFH"
BUILDING_CLASS = 2
WIND_CLASS = 0
STATE = "MV"

# Plot style
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calc_delta_t_hex(lake_temp_C: np.ndarray) -> np.ndarray:
    """Calculate effective temperature difference at heat exchanger."""
    lake_temp_C = np.asarray(lake_temp_C, dtype="float64")
    return np.where(
        lake_temp_C <= T_MIN_EXTRACTION, 0.0,
        np.where(
            lake_temp_C < T_MIN_EXTRACTION + DELTA_T_HEX_MAX,
            lake_temp_C - T_MIN_EXTRACTION,
            DELTA_T_HEX_MAX,
        ),
    )


def calc_cop(lake_temp_C: np.ndarray) -> np.ndarray:
    """Calculate COP based on lake temperature and supply temperature."""
    t = np.asarray(lake_temp_C, dtype="float64")
    return COP_C1 + COP_C2 * t + COP_C3 * T_SUPPLY_C + COP_C4 * t * T_SUPPLY_C


def set_month_xaxis(axes):
    """Set x-axis ticks to month boundaries for hourly data."""
    month_starts = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_labels)


# ============================================================
# STEP 1: STATIC POTENTIAL
# ============================================================

def compute_static_potential(lake_vol_km3):
    """Compute volume-based static thermal potential."""
    print("\n" + "=" * 60)
    print("  STEP 1: STATIC POTENTIAL (volume-based)")
    print("=" * 60)

    seconds_per_year = 365 * 24 * 3600
    vol_m3 = lake_vol_km3 * 1e6

    flow_rate_m3s = vol_m3 / seconds_per_year
    withdrawal_m3s = flow_rate_m3s * (DELTA_T_TECH / DELTA_T_HEX_MAX)
    capacity_W = withdrawal_m3s * RHO * CP * DELTA_T_HEX_MAX

    hours_per_year = cfg.hours_per_year
    potential_Wh = capacity_W * hours_per_year
    potential_GWh = potential_Wh / 1e9
    potential_kWh = potential_Wh / 1e3

    max_annual_GWh = vol_m3 * RHO * CP * DELTA_T_TECH / 3.6e12

    print(f"  Lake volume:        {lake_vol_km3:.4f} km3 ({vol_m3/1e6:.2f} Mio m3)")
    print(f"  Flow rate:          {flow_rate_m3s:.4f} m3/s")
    print(f"  Withdrawal flow:    {withdrawal_m3s:.4f} m3/s")
    print(f"  Thermal capacity:   {capacity_W/1e6:.2f} MW")
    print(f"  Full-load hours:    {hours_per_year} h")
    print(f"  Annual potential:   {potential_GWh:.2f} GWh/yr")
    print(f"  Annual potential:   {potential_kWh:,.0f} kWh/yr")
    print(f"  Max annual (dT constraint): {max_annual_GWh:.2f} GWh/yr")

    return {
        "vol_m3": vol_m3,
        "flow_rate_m3s": flow_rate_m3s,
        "withdrawal_m3s": withdrawal_m3s,
        "capacity_MW": capacity_W / 1e6,
        "potential_GWh": potential_GWh,
        "potential_kWh": potential_kWh,
        "max_annual_GWh": max_annual_GWh,
    }


# ============================================================
# STEP 2: HOURLY TEMPERATURE-DEPENDENT POTENTIAL
# ============================================================

def compute_hourly_potential(lake_vol_km3, df_lake, complete_years):
    """Compute hourly thermal potential constrained by lake temperature."""
    print("\n" + "=" * 60)
    print("  STEP 2: HOURLY TEMPERATURE-DEPENDENT POTENTIAL")
    print("=" * 60)

    seconds_per_year = 365.25 * 24 * 3600
    vol_m3 = lake_vol_km3 * 1e6
    flow_m3s = vol_m3 / seconds_per_year
    max_annual_kWh = vol_m3 * RHO * CP * DELTA_T_TECH / 3.6e6

    df = df_lake.copy()
    df["deltaT_hex"] = calc_delta_t_hex(df["temp_C"].values)
    df["extractable"] = df["deltaT_hex"] > 0
    df["COP"] = calc_cop(df["temp_C"].values)

    df["capacity_kW"] = np.where(
        df["extractable"],
        flow_m3s * RHO * CP * df["deltaT_hex"] / 1000.0,
        0.0,
    )
    df["capacity_MW"] = df["capacity_kW"] / 1000.0
    df["year"] = df.index.year

    annual_results = []
    for year in complete_years:
        mask_yr = df["year"] == year
        raw_kWh = df.loc[mask_yr, "capacity_kW"].sum()

        if raw_kWh > max_annual_kWh:
            scale = max_annual_kWh / raw_kWh
            df.loc[mask_yr, "capacity_kW"] *= scale
            df.loc[mask_yr, "capacity_MW"] = df.loc[mask_yr, "capacity_kW"] / 1000.0
            actual_kWh = max_annual_kWh
        else:
            scale = 1.0
            actual_kWh = raw_kWh

        yr_data = df.loc[mask_yr]
        extractable_mask = yr_data["extractable"]

        annual_results.append({
            "year": year,
            "raw_potential_GWh": raw_kWh / 1e6,
            "constrained_potential_GWh": actual_kWh / 1e6,
            "scale_factor": scale,
            "limited": raw_kWh > max_annual_kWh,
            "hours_extractable": int(extractable_mask.sum()),
            "hours_outage": 8760 - int(extractable_mask.sum()),
            "outage_pct": (8760 - int(extractable_mask.sum())) / 8760 * 100,
            "mean_temp_C": yr_data["temp_C"].mean(),
            "min_temp_C": yr_data["temp_C"].min(),
            "max_temp_C": yr_data["temp_C"].max(),
            "mean_deltaT_hex": float(yr_data.loc[extractable_mask, "deltaT_hex"].mean())
                if extractable_mask.any() else 0,
            "mean_COP": float(yr_data.loc[extractable_mask, "COP"].mean())
                if extractable_mask.any() else np.nan,
            "peak_capacity_MW": yr_data["capacity_MW"].max(),
        })

    df_annual = pd.DataFrame(annual_results)

    print(f"\n  {'Year':<6}{'Potential':<12}{'Peak MW':<10}{'Outage h':<10}"
          f"{'Out %':<8}{'Mean T':<8}{'COP'}")
    print(f"  {'-' * 62}")
    for _, r in df_annual.iterrows():
        print(f"  {int(r['year']):<6}{r['constrained_potential_GWh']:<12.2f}"
              f"{r['peak_capacity_MW']:<10.2f}{int(r['hours_outage']):<10}"
              f"{r['outage_pct']:<8.1f}{r['mean_temp_C']:<8.1f}{r['mean_COP']:.3f}")

    print(f"\n  Mean annual potential: {df_annual['constrained_potential_GWh'].mean():.2f} GWh/yr")
    print(f"  Mean peak capacity:   {df_annual['peak_capacity_MW'].mean():.2f} MW")
    print(f"  Mean outage:          {df_annual['hours_outage'].mean():.0f} h/yr "
          f"({df_annual['outage_pct'].mean():.1f}%)")
    print(f"  Mean COP:             {df_annual['mean_COP'].mean():.3f}")
    print(f"  Max annual (dT lim.): {max_annual_kWh/1e6:.2f} GWh")

    return df, df_annual, max_annual_kWh


# ============================================================
# STEP 3: OUTAGE ANALYSIS
# ============================================================

def analyse_outages(df_hourly, complete_years):
    """Analyse outage events by month and consecutive duration."""
    print("\n" + "=" * 60)
    print("  STEP 3: OUTAGE ANALYSIS")
    print("=" * 60)

    df = df_hourly.copy()
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["is_outage"] = ~df["extractable"]

    monthly_outage = df.groupby("month")["is_outage"].mean() * 100
    print("\n  Monthly outage probability [%]:")
    for m in range(1, 13):
        pct = monthly_outage.get(m, 0)
        bar = "█" * int(pct / 2)
        month_name = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m - 1]
        print(f"  {month_name:<8}{pct:<10.1f}{bar}")

    # Identify consecutive outage runs
    outage_runs = []
    current_run = 0
    for val in df["is_outage"].values:
        if val:
            current_run += 1
        else:
            if current_run > 0:
                outage_runs.append(current_run)
            current_run = 0
    if current_run > 0:
        outage_runs.append(current_run)

    if outage_runs:
        outage_runs = np.array(outage_runs)
        print(f"\n  Consecutive outage events:")
        print(f"    Total events:      {len(outage_runs)}")
        print(f"    Mean duration:     {outage_runs.mean():.1f} h")
        print(f"    Max duration:      {outage_runs.max()} h ({outage_runs.max()/24:.1f} days)")
    else:
        outage_runs = np.array([0])
        print("\n  No outage events detected.")

    return {
        "monthly_outage_pct": monthly_outage,
        "outage_runs": outage_runs,
    }


# ============================================================
# STEP 4: DEMAND PROFILE GENERATION
# ============================================================

def generate_demand_profile(annual_demand_kWh, air_temp_df, target_years):
    """Generate BDEW heating demand profile scaled to annual_demand_kWh."""
    print("\n" + "=" * 60)
    print("  STEP 4: DEMAND PROFILE GENERATION")
    print("=" * 60)
    print(f"  Annual demand (from static potential): {annual_demand_kWh:,.0f} kWh")
    print(f"  = {annual_demand_kWh/1e6:.2f} GWh")

    heat_demand = annual_demand_kWh * 0.85
    hw_demand = annual_demand_kWh * 0.15

    profiles = []
    for year in target_years:
        year_data = air_temp_df.loc[air_temp_df.index.year == year]
        if len(year_data) < 8760:
            continue

        holidays_dict = dict(hol.Germany(state=STATE, years=year))

        heating = bdew.HeatBuilding(
            year_data.index,
            holidays=holidays_dict,
            temperature=year_data["airtemp_C"],
            shlp_type=BUILDING_TYPE,
            building_class=BUILDING_CLASS,
            wind_class=WIND_CLASS,
            annual_heat_demand=heat_demand,
            name="heating",
            ww_incl=False,
        ).get_bdew_profile()

        hotwater = bdew.HeatBuilding(
            year_data.index,
            holidays=holidays_dict,
            temperature=year_data["airtemp_C"],
            shlp_type=BUILDING_TYPE,
            building_class=BUILDING_CLASS,
            wind_class=WIND_CLASS,
            annual_heat_demand=hw_demand,
            name="hotwater",
            ww_incl=True,
        ).get_bdew_profile()

        combined = heating + hotwater
        combined.name = "demand_kW"
        profiles.append(combined)
        print(f"    {year} done - peak: {combined.max():.1f} kW, "
              f"sum: {combined.sum()/1e6:.2f} GWh")

    demand_all = pd.concat(profiles)
    return demand_all


# ============================================================
# STEP 5: COMBINED PLOT – POTENTIAL vs. DEMAND (year with outages)
# ============================================================

def plot_potential_vs_demand(df_hourly, df_annual, demand_series, selected_year):
    """Plot thermal potential and demand profile for a year with outages."""
    print("\n" + "=" * 60)
    print(f"  STEP 5: COMBINED PLOT - Year {selected_year}")
    print("=" * 60)

    # Extract data for selected year
    pot_yr = df_hourly[df_hourly.index.year == selected_year].copy()
    dem_yr = demand_series[demand_series.index.year == selected_year].copy()

    if len(pot_yr) == 0:
        raise ValueError(f"No potential data for year {selected_year}")
    if len(dem_yr) == 0:
        raise ValueError(f"No demand data for year {selected_year}")

    # Align to same length (8760 hours)
    pot_yr = pot_yr.iloc[:8760]
    dem_yr = dem_yr.iloc[:8760]

    # Get annual stats
    yr_stats = df_annual[df_annual["year"] == selected_year].iloc[0]

    # Create hour-of-year axis
    hoy = np.arange(len(pot_yr))

    # Convert potential to kW for comparison
    potential_kW = pot_yr["capacity_kW"].values
    demand_kW = dem_yr.values

    # --- PLOT 1: Time series overlay ---
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True,
                             gridspec_kw={"height_ratios": [1.8, 2.5, 1.5, 1.2]})

    # Panel 1: Lake temperature and delta T at heat exchanger
    ax = axes[0]
    ax2 = ax.twinx()

    # Lake temperature on left axis
    ln1 = ax.plot(hoy, pot_yr["temp_C"].values, lw=0.8, color="steelblue",
                  label="Lake temperature (13.5 m)")
    ax.axhline(T_MIN_EXTRACTION, color="red", ls="--", lw=1.5,
               label=f"T_min extraction = {T_MIN_EXTRACTION} °C")
    ax.axhline(T_MIN_EXTRACTION + DELTA_T_HEX_MAX, color="orange", ls="--", lw=1,
               label=f"Full dT threshold = {T_MIN_EXTRACTION + DELTA_T_HEX_MAX} °C")
    ax.fill_between(hoy, 0, pot_yr["temp_C"].max() * 1.1,
                    where=~pot_yr["extractable"].values,
                    color="red", alpha=0.08, step="mid")
    ax.set_ylabel("Lake Temperature [°C]")
    ax.set_ylim(bottom=0)

    # Delta T at heat exchanger on right axis
    ln2 = ax2.plot(hoy, pot_yr["deltaT_hex"].values, lw=0.6, color="darkorange",
                   alpha=0.8, label=r"$\Delta T_{HEX}$ (effective)")
    ax2.axhline(DELTA_T_HEX_MAX, color="darkorange", ls=":", lw=1.2,
                label=f"Max dT_HEX = {DELTA_T_HEX_MAX} K")
    ax2.set_ylabel(r"$\Delta T_{HEX}$ [K]", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2.set_ylim(0, DELTA_T_HEX_MAX * 1.3)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    ax.set_title(
        f"Lake Schwerin - Thermal Potential vs. Heating Demand ({selected_year})\n"
        f"Outage: {int(yr_stats['hours_outage'])} h ({yr_stats['outage_pct']:.1f}%) | "
        f"Potential: {yr_stats['constrained_potential_GWh']:.2f} GWh | "
        f"Peak capacity: {yr_stats['peak_capacity_MW']:.2f} MW",
        fontsize=12,
    )

    # Panel 2: Potential vs Demand
    ax = axes[1]
    ax.fill_between(hoy, potential_kW / 1000, color="teal", alpha=0.4, lw=0,
                    label=f"Lake thermal potential (peak: {potential_kW.max()/1000:.2f} MW)")
    ax.plot(hoy, demand_kW / 1000, lw=0.4, color="firebrick",
            label=f"Heating demand (peak: {demand_kW.max()/1000:.2f} MW)")
    ax.fill_between(hoy, 0, potential_kW.max() / 1000 * 1.2,
                    where=~pot_yr["extractable"].values,
                    color="red", alpha=0.08, step="mid", label="Outage zone")

    # Mark deficit zones (demand > potential)
    deficit_mask = demand_kW > potential_kW
    if deficit_mask.any():
        ax.fill_between(hoy, potential_kW / 1000, demand_kW / 1000,
                        where=deficit_mask,
                        color="red", alpha=0.3, interpolate=True,
                        label=f"Deficit ({deficit_mask.sum()} h)")

    ax.set_ylabel("Power [MW]")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(bottom=0)

    # Panel 3: Surplus / Deficit
    ax = axes[2]
    balance_kW = potential_kW - demand_kW
    surplus = np.where(balance_kW > 0, balance_kW, 0) / 1000
    deficit = np.where(balance_kW < 0, balance_kW, 0) / 1000

    ax.fill_between(hoy, surplus, color="green", alpha=0.4, lw=0, label="Surplus")
    ax.fill_between(hoy, deficit, color="red", alpha=0.4, lw=0, label="Deficit")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Balance [MW]\n(Potential - Demand)")
    ax.legend(loc="upper right", fontsize=9)

    # Panel 4: Coverage ratio
    ax = axes[3]
    with np.errstate(divide="ignore", invalid="ignore"):
        coverage = np.where(demand_kW > 0, potential_kW / demand_kW * 100, 100)
    coverage = np.clip(coverage, 0, 300)
    ax.plot(hoy, coverage, lw=0.4, color="purple")
    ax.axhline(100, color="black", ls="--", lw=1.5, label="100% coverage")
    ax.fill_between(hoy, 0, coverage, where=coverage < 100,
                    color="red", alpha=0.15, interpolate=True)
    ax.set_ylabel("Coverage [%]")
    ax.set_xlabel("Hour of Year")
    ax.set_ylim(0, 300)
    ax.legend(loc="upper right", fontsize=9)

    set_month_xaxis(axes)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"potential_vs_demand_{selected_year}.svg",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  > Plot saved: potential_vs_demand_{selected_year}.svg")

    # --- PLOT 2: Monthly energy balance ---
    fig, ax = plt.subplots(figsize=(12, 6))

    pot_yr_indexed = pot_yr.copy()
    pot_yr_indexed["month"] = pot_yr_indexed.index.month
    pot_yr_indexed["potential_kWh"] = pot_yr_indexed["capacity_kW"]

    dem_yr_indexed = pd.DataFrame({"demand_kWh": demand_kW}, index=pot_yr.index)
    dem_yr_indexed["month"] = dem_yr_indexed.index.month

    monthly_pot = pot_yr_indexed.groupby("month")["potential_kWh"].sum() / 1e6  # GWh
    monthly_dem = dem_yr_indexed.groupby("month")["demand_kWh"].sum() / 1e6  # GWh

    months = np.arange(1, 13)
    width = 0.35
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    ax.bar(months - width / 2, monthly_pot.values, width,
           color="teal", edgecolor="k", lw=0.3, alpha=0.8,
           label="Lake potential")
    ax.bar(months + width / 2, monthly_dem.values, width,
           color="firebrick", edgecolor="k", lw=0.3, alpha=0.8,
           label="Heating demand")

    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.set_ylabel("Energy [GWh/month]")
    ax.set_xlabel("Month")
    ax.set_title(f"Monthly Energy Balance - Potential vs. Demand ({selected_year})")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"monthly_balance_{selected_year}.svg",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  > Plot saved: monthly_balance_{selected_year}.svg")

    # --- Summary statistics ---
    total_potential_GWh = potential_kW.sum() / 1e6
    total_demand_GWh = demand_kW.sum() / 1e6
    deficit_hours = int(deficit_mask.sum())
    total_deficit_kWh = np.where(balance_kW < 0, -balance_kW, 0).sum()
    total_surplus_kWh = np.where(balance_kW > 0, balance_kW, 0).sum()
    coverage_overall = min(total_potential_GWh / total_demand_GWh * 100, 100)

    print(f"\n  --- Year {selected_year} Summary ---")
    print(f"  Total potential:     {total_potential_GWh:.2f} GWh")
    print(f"  Total demand:        {total_demand_GWh:.2f} GWh")
    print(f"  Overall coverage:    {coverage_overall:.1f}%")
    print(f"  Hours with deficit:  {deficit_hours} h ({deficit_hours/87.6:.1f}%)")
    print(f"  Total deficit:       {total_deficit_kWh/1e6:.2f} GWh")
    print(f"  Total surplus:       {total_surplus_kWh/1e6:.2f} GWh")
    print(f"  Peak demand:         {demand_kW.max()/1000:.2f} MW")
    print(f"  Peak potential:      {potential_kW.max()/1000:.2f} MW")
    print(f"  Outage hours:        {int(yr_stats['hours_outage'])}")

    return {
        "year": selected_year,
        "potential_GWh": total_potential_GWh,
        "demand_GWh": total_demand_GWh,
        "coverage_pct": coverage_overall,
        "deficit_hours": deficit_hours,
        "deficit_GWh": total_deficit_kWh / 1e6,
        "surplus_GWh": total_surplus_kWh / 1e6,
    }


# ============================================================
# STEP 6: HEATMAP – Monthly Energy Coverage [%] (all years, capped at 100%)
# ============================================================

def plot_monthly_coverage_heatmap(df_hourly, demand_series, common_years):
    """Heatmap: monthly energy coverage (potential/demand) over all years, capped at 100%."""
    print("\n" + "=" * 60)
    print("  STEP 6: HEATMAP - Monthly Energy Coverage (capped at 100%)")
    print("=" * 60)

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Build matrix: rows = years, cols = months
    coverage_matrix_raw = np.full((len(common_years), 12), np.nan)

    for i, year in enumerate(common_years):
        pot_yr = df_hourly[df_hourly.index.year == year]
        dem_yr = demand_series[demand_series.index.year == year]

        if len(pot_yr) == 0 or len(dem_yr) == 0:
            continue

        n = min(len(pot_yr), len(dem_yr), 8760)
        pot_kW = pot_yr["capacity_kW"].values[:n]
        dem_kW = dem_yr.values[:n]
        months = pot_yr.index[:n].month

        for m in range(1, 13):
            mask_m = months == m
            pot_month_kWh = pot_kW[mask_m].sum()
            dem_month_kWh = dem_kW[mask_m].sum()

            if dem_month_kWh > 0:
                coverage_matrix_raw[i, m - 1] = pot_month_kWh / dem_month_kWh * 100
            else:
                coverage_matrix_raw[i, m - 1] = 100.0

    # Cap at 100%
    coverage_matrix = np.clip(coverage_matrix_raw, 0, 100)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, max(6, len(common_years) * 0.4)))

    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=0, vmax=100)

    im = ax.imshow(coverage_matrix, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest")

    # Annotate cells with values
    for i in range(len(common_years)):
        for j in range(12):
            val = coverage_matrix[i, j]
            raw_val = coverage_matrix_raw[i, j]
            if not np.isnan(val):
                display_str = f"{raw_val:.0f}" if raw_val <= 100 else "100"
                text_color = "white" if val < 40 else "black"
                ax.text(j, i, display_str, ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels)
    ax.set_yticks(range(len(common_years)))
    ax.set_yticklabels([str(y) for y in common_years])
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_title(
        "Monthly Energy Coverage: Lake Potential / Heating Demand [%]\n"
        "(Capped at 100% - Red = low coverage, Green = fully covered)",
        fontsize=12,
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Coverage [%] (max 100%)")
    cbar.set_ticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap_monthly_coverage_all_years.svg",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  > Plot saved: heatmap_monthly_coverage_all_years.svg")

    # Print summary table
    print(f"\n  {'Year':<6}", end="")
    for ml in month_labels:
        print(f"{ml:<6}", end="")
    print(f"{'Mean':<6}")
    print(f"  {'-' * 84}")
    for i, year in enumerate(common_years):
        print(f"  {year:<6}", end="")
        for j in range(12):
            val = coverage_matrix_raw[i, j]
            print(f"{val:<6.0f}" if not np.isnan(val) else f"{'--':<6}", end="")
        row_mean = np.nanmean(coverage_matrix_raw[i, :])
        print(f"{row_mean:<6.0f}")

    return coverage_matrix_raw


# ============================================================
# STEP 7: HEATMAP – Hourly Deficit [MW] (all years, green-to-red)
# ============================================================

def plot_hourly_deficit_heatmap(df_hourly, demand_series, common_years):
    """
    Heatmap: hourly unmet demand (deficit) over all years.
    Deficit = max(0, demand - potential).
    If potential >= demand -> deficit = 0 (fully supplied, shown green).
    Colormap: green (no deficit) -> red (high deficit).
    """
    print("\n" + "=" * 60)
    print("  STEP 7: HEATMAP - Hourly Deficit (green -> red)")
    print("=" * 60)

    n_hours = 8760
    deficit_matrix = np.zeros((len(common_years), n_hours))

    for i, year in enumerate(common_years):
        pot_yr = df_hourly[df_hourly.index.year == year]
        dem_yr = demand_series[demand_series.index.year == year]

        if len(pot_yr) == 0 or len(dem_yr) == 0:
            deficit_matrix[i, :] = np.nan
            continue

        n = min(len(pot_yr), len(dem_yr), n_hours)
        pot_kW = pot_yr["capacity_kW"].values[:n]
        dem_kW = dem_yr.values[:n]

        # Deficit: demand that cannot be met by lake potential
        deficit = np.maximum(0, dem_kW - pot_kW)

        # Pad if shorter than 8760
        if n < n_hours:
            deficit = np.pad(deficit, (0, n_hours - n),
                             mode="constant", constant_values=np.nan)

        deficit_matrix[i, :] = deficit

    # Convert to MW for readability
    deficit_matrix_MW = deficit_matrix / 1000.0

    # Custom green-to-red colormap
    colors_list = [
        (0.0, "#1a9641"),    # dark green (no deficit)
        (0.001, "#1a9641"),  # keep green for zero
        (0.01, "#a6d96a"),   # light green
        (0.15, "#d9ef8b"),   # yellow-green
        (0.3, "#fee08b"),    # light yellow
        (0.5, "#fdae61"),    # orange
        (0.7, "#f46d43"),    # red-orange
        (0.85, "#d73027"),   # red
        (1.0, "#67001f"),    # dark red (high deficit)
    ]
    cmap_gr = LinearSegmentedColormap.from_list(
        "green_to_red",
        [(pos, col) for pos, col in colors_list],
        N=256,
    )
    cmap_gr.set_bad(color="lightgray")

    # vmax: 99th percentile of non-zero values
    nonzero_vals = deficit_matrix_MW[deficit_matrix_MW > 0]
    vmax = np.nanpercentile(nonzero_vals, 99) if len(nonzero_vals) > 0 else 1.0

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(18, max(6, len(common_years) * 0.45)))

    im = ax.imshow(deficit_matrix_MW, aspect="auto", cmap=cmap_gr,
                   vmin=0, vmax=vmax, interpolation="nearest",
                   origin="upper")

    # X-axis: month boundaries
    month_starts = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels)

    # Y-axis: years
    ax.set_yticks(range(len(common_years)))
    ax.set_yticklabels([str(y) for y in common_years])

    ax.set_xlabel("Hour of Year (by month)")
    ax.set_ylabel("Year")
    ax.set_title(
        "Hourly Power Deficit: Unmet Heating Demand [MW]\n"
        "(Green = fully covered by lake potential, Red = high deficit)",
        fontsize=12,
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, extend="max")
    cbar.set_label("Deficit [MW]")

    # Vertical lines at month boundaries
    for ms in month_starts[1:]:
        ax.axvline(ms, color="white", lw=0.3, alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap_hourly_deficit_all_years.svg",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  > Plot saved: heatmap_hourly_deficit_all_years.svg")

    # --- Summary statistics ---
    print(f"\n  {'Year':<6}{'Deficit h':<14}{'Deficit %':<12}"
          f"{'Max Deficit':<14}{'Total Deficit':<14}")
    print(f"  {'':6}{'':14}{'':12}{'[MW]':<14}{'[GWh]':<14}")
    print(f"  {'-' * 60}")
    for i, year in enumerate(common_years):
        row = deficit_matrix[i, :]
        valid = ~np.isnan(row)
        deficit_hours = int(np.sum(row[valid] > 0))
        deficit_pct = deficit_hours / np.sum(valid) * 100 if np.sum(valid) > 0 else 0
        max_deficit_MW = np.nanmax(row) / 1000
        total_deficit_GWh = np.nansum(row) / 1e6
        print(f"  {year:<6}{deficit_hours:<14}{deficit_pct:<12.1f}"
              f"{max_deficit_MW:<14.2f}{total_deficit_GWh:<14.3f}")

    mean_deficit_hours = np.mean([
        np.sum(deficit_matrix[i, ~np.isnan(deficit_matrix[i, :])] > 0)
        for i in range(len(common_years))
    ])
    mean_deficit_GWh = np.nanmean([
        np.nansum(deficit_matrix[i, :]) for i in range(len(common_years))
    ]) / 1e6
    print(f"\n  Mean deficit hours:  {mean_deficit_hours:.0f} h/yr")
    print(f"  Mean total deficit:  {mean_deficit_GWh:.3f} GWh/yr")

    return deficit_matrix


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  COMBINED ANALYSIS: THERMAL POTENTIAL vs. HEATING DEMAND")
    print("  Lake Schwerin (Hylak_id=13463)")
    print("=" * 60)

    # --- Load lake geometry ---
    lakes_gdf = gpd.read_file(cfg.lakes_path).to_crs(cfg.target_crs)
    lakes_gdf["Hylak_id"] = lakes_gdf["Hylak_id"].astype(int)
    lake_row = lakes_gdf[lakes_gdf["Hylak_id"] == LAKE_ID]
    if lake_row.empty:
        raise ValueError(f"Lake {LAKE_ID} not found!")

    lake_vol_km3 = float(lake_row.iloc[0]["Vol_total"])
    lake_area_km2 = lake_row.iloc[0].geometry.area / 1e6
    print(f"\n  Lake ID: {LAKE_ID}")
    print(f"  Area: {lake_area_km2:.2f} km2")
    print(f"  Volume: {lake_vol_km3:.4f} km3")

    # --- Load lake temperature data ---
    df_lake = pd.read_csv(LAKE_TEMP_CSV, parse_dates=["datetime"], index_col="datetime")
    hours_per_year_lake = df_lake.groupby(df_lake.index.year).size()
    complete_years_lake = sorted(
        hours_per_year_lake[hours_per_year_lake >= 8760].index.tolist()
    )
    df_lake = df_lake[df_lake.index.year.isin(complete_years_lake)]
    print(f"\n  Lake temp data: {complete_years_lake[0]}-{complete_years_lake[-1]}")

    # --- Load air temperature data ---
    df_air = pd.read_csv(AIR_TEMP_CSV, parse_dates=["datetime"], index_col="datetime")
    hours_per_year_air = df_air.groupby(df_air.index.year).size()
    complete_years_air = sorted(
        hours_per_year_air[hours_per_year_air >= 8760].index.tolist()
    )
    df_air = df_air[df_air.index.year.isin(complete_years_air)]
    print(f"  Air temp data:  {complete_years_air[0]}-{complete_years_air[-1]}")

    # Common complete years
    common_years = sorted(set(complete_years_lake) & set(complete_years_air))
    print(f"  Common years:   {common_years[0]}-{common_years[-1]} ({len(common_years)} yrs)")

    # =========================================================
    # STEP 1: Static potential
    # =========================================================
    static = compute_static_potential(lake_vol_km3)

    # =========================================================
    # STEP 2: Hourly dynamic potential
    # =========================================================
    df_hourly, df_annual, max_annual_kWh = compute_hourly_potential(
        lake_vol_km3, df_lake, complete_years_lake
    )

    # =========================================================
    # STEP 3: Outage analysis
    # =========================================================
    outage_info = analyse_outages(df_hourly, complete_years_lake)

    # =========================================================
    # STEP 4: Generate demand profile using fixed economic potential
    # =========================================================
    ANNUAL_DEMAND_TOTAL_KWH = 97.5711875 * 1e6
    print(f"\n  >>> Setting ANNUAL_DEMAND_TOTAL_KWH = Fixed Economic Potential = "
          f"{ANNUAL_DEMAND_TOTAL_KWH:,.0f} kWh ({ANNUAL_DEMAND_TOTAL_KWH/1e6:.3f} GWh)")

    demand_series = generate_demand_profile(
        ANNUAL_DEMAND_TOTAL_KWH, df_air, common_years
    )

    # =========================================================
    # STEP 5: Select year with outages and plot
    # =========================================================
    df_annual_common = df_annual[df_annual["year"].isin(common_years)]
    years_with_outages = df_annual_common[df_annual_common["hours_outage"] > 0]

    if len(years_with_outages) > 0:
        mean_outage_hours = years_with_outages["hours_outage"].mean()
        closest_idx = (years_with_outages["hours_outage"] - mean_outage_hours).abs().idxmin()
        selected_year = 2015
        closest_hours = int(years_with_outages.loc[closest_idx, "hours_outage"])
        print(f"\n  >>> Selected year: {selected_year} "
              f"({closest_hours} h outage, mean: {mean_outage_hours:.0f} h)")
    else:
        median_idx = (df_annual_common["constrained_potential_GWh"] -
                      df_annual_common["constrained_potential_GWh"].median()).abs().idxmin()
        selected_year = int(df_annual_common.loc[median_idx, "year"])
        print(f"\n  >>> No outages found; using median year: {selected_year}")

    # Plot combined analysis for selected year
    results = plot_potential_vs_demand(df_hourly, df_annual, demand_series, selected_year)

    # =========================================================
    # STEP 6: Heatmap - Monthly coverage all years
    # =========================================================
    coverage_matrix = plot_monthly_coverage_heatmap(
        df_hourly, demand_series, common_years
    )

    # =========================================================
    # STEP 7: Heatmap - Hourly deficit all years
    # =========================================================
    deficit_matrix = plot_hourly_deficit_heatmap(
        df_hourly, demand_series, common_years
    )

    # =========================================================
    # Save combined results
    # =========================================================
    combined_summary = {
        "lake_id": LAKE_ID,
        "static_potential_GWh": static["potential_GWh"],
        "annual_demand_GWh": ANNUAL_DEMAND_TOTAL_KWH / 1e6,
        "selected_year": selected_year,
        **results,
    }
    pd.DataFrame([combined_summary]).to_csv(
        OUTPUT_DIR / "combined_summary.csv", index=False
    )

    print("\n" + "=" * 60)
    print("  > COMBINED ANALYSIS COMPLETE")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()