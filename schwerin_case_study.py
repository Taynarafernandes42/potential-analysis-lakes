"""
Combined Analysis: Lake Thermal Potential vs. District Heating Demand
=====================================================================
1. Technical potential (water side) – static & dynamic
2. Heating demand profile (BDEW) scaled to static potential
3. Overlay plot for a year with outages
4. Daily coverage heatmap (all years)
5. Hourly deficit heatmap (all years)

Scenarios: every step from the hourly potential onwards is run for
  - two lake temperature series (without / with heat-pump feedback to the lake)
  - two minimum extraction temperatures (2 °C / 3 °C)
Each scenario writes to its own subfolder; scenario_comparison.csv
in the base folder collects the key results of all scenarios.
"""

from pathlib import Path
import inspect
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LinearSegmentedColormap
import warnings
import holidays as hol
from demandlib import bdew

from config import Config, lake_to_useful, useful_to_lake
from lake_io import read_lakes

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

cfg = Config()

# Lakes: one or more HydroLAKES IDs. Several lakes are treated as ONE combined
# lake: volumes, areas and economic heat (Heat_EI_ge1_MWh) are summed, and the
# single temperature series LAKE_TEMP_CSV is applied to the combined lake.
LAKE_IDS = [13463]            # e.g. [13463, <second Hylak_id>]
LAKE_NAME = "Lake Schwerin"   # used in titles and console output

# Optional hard-coded volume [Mio m3] per lake ID (None / not listed = HydroLAKES Vol_total)
LAKE_VOLUME_OVERRIDE_MCM = {
    13463: 356,
}
OUTPUT_BASE_DIR = Path(cfg.results_dir) / "CaseStudy_Schwerin"
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = OUTPUT_BASE_DIR  # set per scenario in run_scenario()

AIR_TEMP_CSV = Path("Data/Raw/schwerin_airtemp_1994_2020.csv")

# ------------------------------------------------------------
# SCENARIOS: lake temperature series x minimum extraction temperature
# Every combination is computed; results go to
#   OUTPUT_BASE_DIR / "<temp_key>_Tmin<T>C"
# ------------------------------------------------------------
T_MIN_SCENARIOS = [2.0, 3.0]  # minimum lake temperature for extraction [°C]
LAKE_DEPTH_LABEL = "16.5 m"   # intake depth (plot titles only)

# Lake temperature file per series and T_min [°C].
# The HP-feedback run was coupled with the same T_min as the analysis.
_T_REF_CSV = Path("Data/Raw/schwerin_temperature_16.5m_hourly.csv")
LAKE_TEMP_SCENARIOS = {
    # analysis without heat-pump feedback to the lake (same file for all T_min)
    "no_hp_feedback": {t: _T_REF_CSV for t in T_MIN_SCENARIOS},
    # analysis with heat-pump feedback to the lake (coupled run per T_min)
    "hp_feedback": {
        t: Path(f"Data/Raw/schwerin_temperature_16.5m_hourly_heat_pump_coupled_Tmin{t:g}C.csv")
        for t in T_MIN_SCENARIOS
    },
}

# Physical / technical parameters
RHO = cfg.density_water
CP = cfg.cp_water
DELTA_T_TECH = cfg.deltaT_technical_extractable
DELTA_T_HEX_MAX = 5.0
T_MIN_EXTRACTION = 3.0  # overwritten per scenario in run_scenario()

# COP parameters: Lorenz-efficiency approach (Pieper et al. 2019/2020)
#   COP(t) = eta_L * T_H,lm / (T_H,lm - T_C,lm(t))
#   T_H,lm : log-mean sink temperature (DH supply / return)
#   T_C,lm : log-mean source temperature (lake inlet / heat-exchanger outlet)
T_SUPPLY_C = 75.0     # DH supply temperature [°C] (manuscript: 75-80 °C)
T_RETURN_C = 40.0     # DH return temperature [°C]
LORENZ_EFF = 0.5      # Lorenz efficiency eta_L [-] (Pieper et al.: 0.5 literature value)

# ASHP reference (climate-year comparison, appendix):
#   hourly COP following When2Heat (Ruhnau et al. 2019; Ruhnau & Muessel 2022,
#   parameters from github.com/oruhnau/when2heat, input/cop/cop_parameters.csv):
#   COP = f_corr * (c0 + c1*dT + c2*dT^2),  dT = T_sink - T_air  (dT >= 15 K)
#   Sink: radiator heating curve T_sink = 40 - T_air (space heating),
#         50 °C for domestic hot water
ASHP_C0 = 6.08
ASHP_C1 = -0.0941
ASHP_C2 = 0.000464
ASHP_DT_MIN = 15.0         # lower bound of the temperature lift [K]
ASHP_CORRECTION = 0.85     # correction based on field measurements
ASHP_SINK_SH_BASE = 40.0   # radiator heating curve: T_sink = 40 - 1.0 * T_air
ASHP_SINK_SH_SLOPE = -1.0
ASHP_SINK_DHW = 50.0       # domestic hot water sink temperature [°C]

# Backup for hours without lake-source supply (appendix comparison only)
BACKUP_EFF = 1.0         # electric boiler, same tariff as LSWSHP

# Year shown in the detailed plots (None = most representative outage year)
SELECTED_YEAR_OVERRIDE = 2017

# Demand profile parameters
BUILDING_TYPE = "MFH"
BUILDING_CLASS = 2
WIND_CLASS = 0
STATE = "MV"

# Plot style
plt.rcParams.update({
    "font.size": 18,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 20,
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


def _log_mean_K(t_in_C, t_out_C):
    """Logarithmic mean temperature [K] of a stream from t_in to t_out [°C]."""
    a = np.asarray(t_in_C, dtype="float64") + 273.15
    b = np.asarray(t_out_C, dtype="float64") + 273.15
    with np.errstate(divide="ignore", invalid="ignore"):
        lm = (a - b) / np.log(a / b)
    return np.where(np.abs(a - b) < 1e-6, a, lm)


def calc_cop(lake_temp_C: np.ndarray, delta_t_hex: np.ndarray) -> np.ndarray:
    """Hourly COP from the Lorenz cycle with a constant Lorenz efficiency.

    Source: lake water cooled from T_lake to T_lake - deltaT_hex.
    Sink:   district heating water heated from T_RETURN_C to T_SUPPLY_C.
    """
    t = np.asarray(lake_temp_C, dtype="float64")
    dt = np.asarray(delta_t_hex, dtype="float64")
    t_h = _log_mean_K(T_SUPPLY_C, T_RETURN_C)
    t_c = _log_mean_K(t, t - dt)
    return LORENZ_EFF * t_h / (t_h - t_c)


def _align_year(df_hourly, demand_series, year):
    """Lake data and demand of one year, aligned by timestamp (all hours,
    8784 h in leap years). Replaces the former positional cut at 8760 h,
    which dropped 31 Dec in leap years."""
    pot = df_hourly[df_hourly.index.year == year]
    dem = demand_series[demand_series.index.year == year]
    idx = pot.index.intersection(dem.index)
    if not (len(idx) == len(pot) == len(dem)):
        raise ValueError(
            f"{year}: index mismatch (lake {len(pot)} h, demand {len(dem)} h, "
            f"common {len(idx)} h) - check time zone / time stamps"
        )
    return pot.loc[idx], dem.loc[idx].values


MAX_GAP_FILL_H = 48  # max. length of a data gap that is filled [h]


def load_hourly_complete_years(csv_path, label, max_gap_h=MAX_GAP_FILL_H):
    """Read an hourly CSV, reindex to a gap-free hourly time axis and fill
    short gaps (time interpolation, at series ends: nearest value).
    A year is complete only if it has all hours (8784 in leap years)
    and no gap longer than max_gap_h remains."""
    df = pd.read_csv(csv_path, parse_dates=["datetime"], index_col="datetime")
    df = df[~df.index.duplicated(keep="first")].sort_index()

    full_idx = pd.date_range(f"{df.index.year.min()}-01-01 00:00",
                             f"{df.index.year.max()}-12-31 23:00",
                             freq="h", tz=df.index.tz)
    df = df.reindex(full_idx)
    df.index.name = "datetime"
    missing = df.isna().any(axis=1)

    # length of each gap (consecutive missing hours); only short gaps are filled
    run_id = (missing != missing.shift()).cumsum()
    gap_len = missing.groupby(run_id).transform("sum")
    fillable = missing & (gap_len <= max_gap_h)

    filled_df = df.interpolate(method="time", limit_area="inside").ffill().bfill()
    df = df.where(~fillable, filled_df)
    still_missing = df.isna().any(axis=1)

    miss_per_year = missing.groupby(df.index.year).sum()
    left_per_year = still_missing.groupby(df.index.year).sum()
    complete_years = sorted(left_per_year[left_per_year == 0].index.tolist())

    filled = miss_per_year[(miss_per_year > 0) & (left_per_year == 0)]
    dropped = left_per_year[left_per_year > 0]
    for y, n in filled.items():
        print(f"  {label}: {y} - {int(n)} missing h filled")
    for y, n in dropped.items():
        print(f"  {label}: {y} excluded ({int(n)} h missing, gap > {max_gap_h} h)")

    df = df[df.index.year.isin(complete_years)]
    return df, complete_years


def _drop_feb29(index, values):
    """Remove 29 Feb for plots with a fixed 365-day / 8760-h axis."""
    keep = ~((index.month == 2) & (index.day == 29))
    return values[keep]


def set_month_xaxis(axes):
    """Set x-axis ticks to month boundaries for hourly data."""
    month_starts = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_labels)


def save_case_study_plot(fig, stem):
    """Save a case-study figure as PDF and SVG in the results folder."""
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  > Plot saved: {stem}.pdf and {stem}.svg")


# ============================================================
# STEP 1: STATIC POTENTIAL
# ============================================================

def compute_static_potential(lake_vol_mcm):
    """Compute volume-based static thermal potential."""
    print("\n" + "=" * 60)
    print("  STEP 1: STATIC POTENTIAL (volume-based)")
    print("=" * 60)

    seconds_per_year = 365 * 24 * 3600
    vol_m3 = lake_vol_mcm * 1e6

    flow_rate_m3s = vol_m3 / seconds_per_year
    withdrawal_m3s = flow_rate_m3s * (DELTA_T_TECH / DELTA_T_HEX_MAX)
    capacity_W = withdrawal_m3s * RHO * CP * DELTA_T_HEX_MAX

    hours_per_year = cfg.hours_per_year
    potential_Wh = capacity_W * hours_per_year
    potential_GWh = potential_Wh / 1e9
    potential_kWh = potential_Wh / 1e3

    max_annual_GWh = vol_m3 * RHO * CP * DELTA_T_TECH / 3.6e12

    print(f"  Lake volume:        {lake_vol_mcm:.2f} Mio m3 ({vol_m3/1e9:.4f} km3)")
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

def compute_hourly_potential(lake_vol_mcm, df_lake, complete_years):
    """Compute hourly thermal potential constrained by lake temperature."""
    print("\n" + "=" * 60)
    print("  STEP 2: HOURLY TEMPERATURE-DEPENDENT POTENTIAL")
    print("=" * 60)

    seconds_per_year = 365.25 * 24 * 3600
    vol_m3 = lake_vol_mcm * 1e6
    flow_m3s = vol_m3 / seconds_per_year * (DELTA_T_TECH / DELTA_T_HEX_MAX)
    max_annual_kWh = vol_m3 * RHO * CP * DELTA_T_TECH / 3.6e6

    df = df_lake.copy()
    df["deltaT_hex"] = calc_delta_t_hex(df["temp_C"].values)
    df["extractable"] = df["deltaT_hex"] > 0
    df["COP"] = calc_cop(df["temp_C"].values, df["deltaT_hex"].values)

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
        n_hours_yr = len(yr_data)
        n_extractable = int(extractable_mask.sum())

        annual_results.append({
            "year": year,
            "raw_potential_GWh": raw_kWh / 1e6,
            "constrained_potential_GWh": actual_kWh / 1e6,
            "scale_factor": scale,
            "limited": raw_kWh > max_annual_kWh,
            "hours_extractable": n_extractable,
            "hours_outage": n_hours_yr - n_extractable,
            "outage_pct": (n_hours_yr - n_extractable) / n_hours_yr * 100,
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

_DEMANDLIB_HAS_WW_ONLY = "ww_only" in inspect.getsource(bdew.HeatBuilding)


def generate_demand_profile(annual_demand_kWh, air_temp_df, target_years):
    """Generate BDEW demand profile: space heating (85 %) + hot water (15 %)."""
    print("\n" + "=" * 60)
    print("  STEP 4: DEMAND PROFILE GENERATION")
    print("=" * 60)
    if _DEMANDLIB_HAS_WW_ONLY:
        print("  Hot water: BDEW profile (ww_only=True)")
    else:
        print("  Hot water: demandlib without ww_only -> constant profile used")
    print(f"  Annual demand (from static potential): {annual_demand_kWh:,.0f} kWh")
    print(f"  = {annual_demand_kWh/1e6:.2f} GWh")

    heat_demand = annual_demand_kWh * 0.85
    hw_demand = annual_demand_kWh * 0.15

    profiles = []
    profiles_sh = []
    profiles_dhw = []
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

        # ww_incl=True would return heating INCLUDING hot water (seasonal shape).
        # Pure hot water needs ww_only=True (newer demandlib versions only).
        if _DEMANDLIB_HAS_WW_ONLY:
            hotwater = bdew.HeatBuilding(
                year_data.index,
                holidays=holidays_dict,
                temperature=year_data["airtemp_C"],
                shlp_type=BUILDING_TYPE,
                building_class=BUILDING_CLASS,
                wind_class=WIND_CLASS,
                annual_heat_demand=hw_demand,
                name="hotwater",
                ww_only=True,
            ).get_bdew_profile()
        else:
            # Fallback: constant hot-water load over the year
            hotwater = pd.Series(hw_demand / len(year_data), index=year_data.index)

        combined = heating + hotwater
        combined.name = "demand_kW"
        profiles.append(combined)
        profiles_sh.append(heating)
        profiles_dhw.append(hotwater)
        print(f"    {year} done - peak: {combined.max():.1f} kW, "
              f"sum: {combined.sum()/1e6:.2f} GWh")

    demand_all = pd.concat(profiles)
    demand_sh = pd.concat(profiles_sh)
    demand_dhw = pd.concat(profiles_dhw)
    return demand_all, demand_sh, demand_dhw


# ============================================================
# STEP 5: COMBINED PLOT – POTENTIAL vs. DEMAND (year with outages)
# ============================================================

def plot_potential_vs_demand(df_hourly, df_annual, demand_series, selected_year):
    """Plot thermal potential and demand profile for a year with outages."""
    print("\n" + "=" * 60)
    print(f"  STEP 5: COMBINED PLOT - Year {selected_year}")
    print("=" * 60)

    # Extract data for selected year (aligned by timestamp, full year)
    pot_yr, demand_kW = _align_year(df_hourly, demand_series, selected_year)
    pot_yr = pot_yr.copy()

    if len(pot_yr) == 0:
        raise ValueError(f"No data for year {selected_year}")

    # Get annual stats
    yr_stats = df_annual[df_annual["year"] == selected_year].iloc[0]

    # Create hour-of-year axis
    hoy = np.arange(len(pot_yr))

    # Convert potential to kW for comparison
    potential_kW = pot_yr["useful_kW"].values

    # --- Distinct color palette ---
    COLOR_LAKE_TEMP = "#1f4e79"       # dark navy blue
    COLOR_TMIN_LINE = "#c00000"       # strong red
    COLOR_DELTA_T = "#ff8c00"         # amber/orange
    COLOR_POTENTIAL = "#008080"       # teal
    COLOR_DEMAND = "#8b0000"          # dark red
    COLOR_DEFICIT = "#ff4444"         # bright red
    COLOR_COVERAGE = "#6a0dad"        # purple
    COLOR_BALANCE_POS = "#228b22"     # forest green
    COLOR_BALANCE_NEG = "#dc143c"     # crimson

    # --- PLOT 1: Time series overlay ---
    fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True,
                             gridspec_kw={"height_ratios": [1.8, 2.5, 1.5, 1.2]})

    # ----------------------------------------------------------------
    # Panel A: Lake temperature and delta T at heat exchanger
    # ----------------------------------------------------------------
    ax = axes[0]
    ax2 = ax.twinx()

    # Lake temperature on left axis
    ax.plot(hoy, pot_yr["temp_C"].values, lw=1.8, color=COLOR_LAKE_TEMP,
            label="Lake temperature", zorder=3)
    ax.axhline(T_MIN_EXTRACTION, color=COLOR_TMIN_LINE, ls="--", lw=2.0,
               label=f"$T_{{min}}$ = {T_MIN_EXTRACTION} °C", zorder=2)
    ax.fill_between(hoy, 0, pot_yr["temp_C"].max() * 1.1,
                    where=~pot_yr["extractable"].values,
                    color="red", alpha=0.06, step="mid")

    # Left axis limits
    y_max_left = pot_yr["temp_C"].max() * 1.15
    y_min_left = 0.0
    ax.set_ylim(y_min_left, y_max_left)
    ax.set_ylabel("Lake Temperature [°C]")

    # Delta T at heat exchanger on right axis
    ax2.plot(hoy, pot_yr["deltaT_hex"].values, lw=1.4, color=COLOR_DELTA_T,
             alpha=0.9, label=r"$\Delta T_{HEX}$", zorder=2)

    # --- Align right axis 0 with left axis T_MIN_EXTRACTION (3°C) ---
    frac_3C = (T_MIN_EXTRACTION - y_min_left) / (y_max_left - y_min_left)
    y_max_right = DELTA_T_HEX_MAX * 1.4
    y_min_right = -frac_3C * y_max_right / (1.0 - frac_3C)
    ax2.set_ylim(y_min_right, y_max_right)

    ax2.set_ylabel(r"$\Delta T_{HEX}$ [K]", color=COLOR_DELTA_T)
    ax2.tick_params(axis="y", labelcolor=COLOR_DELTA_T)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax.set_title(
        f"{LAKE_NAME} \u2013 Thermal Potential vs. Heating Demand ({selected_year})",
        fontweight="bold",
    )

    # Subplot label (A)
    ax.text(-0.02, 1.05, "A", transform=ax.transAxes,
            fontsize=22, fontweight="bold", va="bottom", ha="right")

    # ----------------------------------------------------------------
    # Panel B: Potential vs Demand
    # ----------------------------------------------------------------
    ax = axes[1]
    ax.fill_between(hoy, potential_kW / 1000, color=COLOR_POTENTIAL, alpha=0.45, lw=0,
                    label=f"Capped at installed capacity (peak: {potential_kW.max()/1000:.2f} MW)")
    ax.plot(hoy, demand_kW / 1000, lw=0.6, color=COLOR_DEMAND,
            label=f"Heating demand (peak: {demand_kW.max()/1000:.2f} MW)")

    # Mark deficit zones (demand > potential)
    deficit_mask = demand_kW > potential_kW
    if deficit_mask.any():
        ax.fill_between(hoy, potential_kW / 1000, demand_kW / 1000,
                        where=deficit_mask,
                        color=COLOR_DEFICIT, alpha=0.45, interpolate=True,
                        label=f"Deficit ({deficit_mask.sum()} h)")

    ax.set_ylabel("Power [MW]")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)

    # Subplot label (B)
    ax.text(-0.02, 1.05, "B", transform=ax.transAxes,
            fontsize=22, fontweight="bold", va="bottom", ha="right")

    # ----------------------------------------------------------------
    # Panel C: Surplus / Deficit
    # ----------------------------------------------------------------
    ax = axes[2]
    balance_kW = potential_kW - demand_kW
    surplus = np.where(balance_kW > 0, balance_kW, 0) / 1000
    deficit = np.where(balance_kW < 0, balance_kW, 0) / 1000

    ax.fill_between(hoy, surplus, color=COLOR_BALANCE_POS, alpha=0.5, lw=0,
                    label="Surplus")
    ax.fill_between(hoy, deficit, color=COLOR_BALANCE_NEG, alpha=0.5, lw=0,
                    label="Deficit")
    ax.axhline(0, color="black", lw=1.0)
    ax.set_ylabel("Balance [MW]\n(Useful heat \u2212 Demand)")
    ax.legend(loc="upper right")

    # Subplot label (C)
    ax.text(-0.02, 1.05, "C", transform=ax.transAxes,
            fontsize=22, fontweight="bold", va="bottom", ha="right")

    # ----------------------------------------------------------------
    # Panel D: Coverage ratio
    # ----------------------------------------------------------------
    ax = axes[3]
    with np.errstate(divide="ignore", invalid="ignore"):
        coverage = np.where(demand_kW > 0, potential_kW / demand_kW * 100, 100)
    coverage = np.clip(coverage, 0, 300)
    ax.plot(hoy, coverage, lw=0.5, color=COLOR_COVERAGE)
    ax.axhline(100, color="black", ls="--", lw=2.0, label="100% coverage")
    ax.fill_between(hoy, 0, coverage, where=coverage < 100,
                    color=COLOR_DEFICIT, alpha=0.15, interpolate=True)
    ax.set_ylabel("Coverage [%]")
    ax.set_xlabel("Hour of Year")
    ax.set_ylim(0, 300)
    ax.legend(loc="upper right")

    # Subplot label (D)
    ax.text(-0.02, 1.05, "D", transform=ax.transAxes,
            fontsize=22, fontweight="bold", va="bottom", ha="right")

    set_month_xaxis(axes)
    plt.tight_layout()
    save_case_study_plot(fig, f"potential_vs_demand_{selected_year}")

    # --- PLOT 2: Monthly energy balance ---
    fig, ax = plt.subplots(figsize=(12, 6))

    pot_yr_indexed = pot_yr.copy()
    pot_yr_indexed["month"] = pot_yr_indexed.index.month
    pot_yr_indexed["potential_kWh"] = pot_yr_indexed["useful_kW"]

    dem_yr_indexed = pd.DataFrame({"demand_kWh": demand_kW}, index=pot_yr.index)
    dem_yr_indexed["month"] = dem_yr_indexed.index.month

    monthly_pot = pot_yr_indexed.groupby("month")["potential_kWh"].sum() / 1e6  # GWh
    monthly_dem = dem_yr_indexed.groupby("month")["demand_kWh"].sum() / 1e6  # GWh

    months = np.arange(1, 13)
    width = 0.35
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    ax.bar(months - width / 2, monthly_pot.values, width,
           color=COLOR_POTENTIAL, edgecolor="k", lw=0.3, alpha=0.85,
           label="Lake-based useful heat")
    ax.bar(months + width / 2, monthly_dem.values, width,
           color=COLOR_DEMAND, edgecolor="k", lw=0.3, alpha=0.85,
           label="Heating demand")

    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.set_ylabel("Energy [GWh/month]")
    ax.set_xlabel("Month")
    ax.set_title(f"Monthly Energy Balance \u2013 Lake-based Useful Heat vs. Demand ({selected_year})")
    ax.legend(loc="upper right")
    plt.tight_layout()
    save_case_study_plot(fig, f"monthly_balance_{selected_year}")

    # --- Summary statistics ---
    total_potential_GWh = potential_kW.sum() / 1e6
    total_demand_GWh = demand_kW.sum() / 1e6
    deficit_hours = int(deficit_mask.sum())
    total_deficit_kWh = np.where(balance_kW < 0, -balance_kW, 0).sum()
    total_surplus_kWh = np.where(balance_kW > 0, balance_kW, 0).sum()
    # Time-resolved coverage: only demand met in the same hour counts
    covered_GWh = np.minimum(potential_kW, demand_kW).sum() / 1e6
    coverage_overall = covered_GWh / total_demand_GWh * 100

    print(f"\n  --- Year {selected_year} Summary ---")
    print(f"  Total useful heat:   {total_potential_GWh:.2f} GWh")
    print(f"  Total demand:        {total_demand_GWh:.2f} GWh")
    print(f"  Covered demand:      {covered_GWh:.2f} GWh")
    print(f"  Overall coverage:    {coverage_overall:.1f}% (hourly)")
    print(f"  Hours with deficit:  {deficit_hours} h ({deficit_hours/len(pot_yr)*100:.1f}%)")
    print(f"  Total deficit:       {total_deficit_kWh/1e6:.2f} GWh")
    print(f"  Total surplus:       {total_surplus_kWh/1e6:.2f} GWh")
    print(f"  Peak demand:         {demand_kW.max()/1000:.2f} MW")
    print(f"  Peak useful heat:    {potential_kW.max()/1000:.2f} MW")
    print(f"  Outage hours:        {int(yr_stats['hours_outage'])}")

    return {
        "year": selected_year,
        "useful_heat_GWh": total_potential_GWh,
        "demand_GWh": total_demand_GWh,
        "covered_GWh": covered_GWh,
        "coverage_pct": coverage_overall,
        "deficit_hours": deficit_hours,
        "deficit_GWh": total_deficit_kWh / 1e6,
        "surplus_GWh": total_surplus_kWh / 1e6,
    }


# ============================================================
# STEP 6: HEATMAP – Daily Energy Coverage [%] (all years)
# ============================================================

def plot_daily_coverage_heatmap(df_hourly, demand_series, common_years):
    """Heatmap: daily share of demand covered hour by hour (0-100 %), all years."""
    print("\n" + "=" * 60)
    print("  STEP 6: HEATMAP - Daily Energy Coverage (all years)")
    print("=" * 60)

    n_days = 366  # leap years incl. 31 Dec; day 366 = NaN in common years
    coverage_matrix_raw = np.full((len(common_years), n_days), np.nan)

    for i, year in enumerate(common_years):
        pot_yr, dem_kW = _align_year(df_hourly, demand_series, year)
        if len(pot_yr) == 0:
            continue
        pot_kW = pot_yr["useful_kW"].values

        # Day of year (0-indexed)
        doy = pot_yr.index.dayofyear - 1  # 0-based

        for d in range(n_days):
            mask_d = doy == d
            if not mask_d.any():
                continue
            # covered = demand met hour by hour (no shifting within the day)
            covered_day_kWh = np.minimum(pot_kW[mask_d], dem_kW[mask_d]).sum()
            dem_day_kWh = dem_kW[mask_d].sum()

            if dem_day_kWh > 0:
                coverage_matrix_raw[i, d] = covered_day_kWh / dem_day_kWh * 100
            else:
                coverage_matrix_raw[i, d] = 100.0

    # Cap at 100%
    coverage_matrix = np.clip(coverage_matrix_raw, 0, 100)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(18, max(6, len(common_years) * 0.45)))

    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=0, vmax=100)

    im = ax.imshow(coverage_matrix, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest", origin="upper")

    # X-axis: month boundaries (day-of-year)
    month_day_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(month_day_starts)
    ax.set_xticklabels(month_labels)

    # Vertical lines at month boundaries
    for ms in month_day_starts[1:]:
        ax.axvline(ms - 0.5, color="white", lw=0.5, alpha=0.7)

    # Y-axis: years
    ax.set_yticks(range(len(common_years)))
    ax.set_yticklabels([str(y) for y in common_years])

    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Year")
    ax.set_title("Daily Energy Coverage: Lake-based Useful Heat / Heating Demand [%]")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Coverage [%]")
    cbar.set_ticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    plt.tight_layout()
    save_case_study_plot(fig, "heatmap_daily_coverage_all_years")

    # --- Summary: days below thresholds ---
    below_50 = np.nansum(coverage_matrix_raw < 50, axis=1)
    below_80 = np.nansum(coverage_matrix_raw < 80, axis=1)
    below_100 = np.nansum(coverage_matrix_raw < 100, axis=1)

    print(f"\n  {'Year':<6}{'Days <50%':<12}{'Days <80%':<12}{'Days <100%':<12}"
          f"{'Mean Cov %':<12}")
    print(f"  {'-' * 54}")
    for i, year in enumerate(common_years):
        mean_cov = np.nanmean(coverage_matrix_raw[i, :])
        print(f"  {year:<6}{int(below_50[i]):<12}{int(below_80[i]):<12}"
              f"{int(below_100[i]):<12}{mean_cov:<12.1f}")

    print(f"  {'-' * 54}")
    print(f"  {'Mean':<6}{below_50.mean():<12.0f}{below_80.mean():<12.0f}"
          f"{below_100.mean():<12.0f}"
          f"{np.nanmean(coverage_matrix_raw):<12.1f}")

    return coverage_matrix_raw


# ============================================================
# STEP 7: HEATMAP – Hourly Deficit [MW] (all years, green-to-red)
# ============================================================

def plot_hourly_deficit_heatmap(df_hourly, demand_series, common_years):
    """
    Heatmap: hourly unmet demand (deficit) over all years.
    Deficit = max(0, demand - potential).
    Colormap: green (no deficit) -> red (high deficit).
    """
    print("\n" + "=" * 60)
    print("  STEP 7: HEATMAP - Hourly Deficit (green -> red)")
    print("=" * 60)

    n_hours = 8760  # plot axis; 29 Feb removed from leap years for display only
    deficit_matrix = np.full((len(common_years), n_hours), np.nan)
    deficit_full = {}  # full-year deficit per year (incl. 29 Feb) for statistics

    for i, year in enumerate(common_years):
        pot_yr, dem_kW = _align_year(df_hourly, demand_series, year)
        if len(pot_yr) == 0:
            continue
        pot_kW = pot_yr["useful_kW"].values

        # Deficit: demand that cannot be met by lake potential
        deficit = np.maximum(0, dem_kW - pot_kW)
        deficit_full[year] = deficit

        deficit_plot = _drop_feb29(pot_yr.index, deficit)
        m = min(len(deficit_plot), n_hours)
        deficit_matrix[i, :m] = deficit_plot[:m]

    # Convert to MW for readability
    deficit_matrix_MW = deficit_matrix / 1000.0

    # Custom green-to-red colormap
    colors_list = [
        (0.0, "#1a9641"),
        (0.001, "#1a9641"),
        (0.01, "#a6d96a"),
        (0.15, "#d9ef8b"),
        (0.3, "#fee08b"),
        (0.5, "#fdae61"),
        (0.7, "#f46d43"),
        (0.85, "#d73027"),
        (1.0, "#67001f"),
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

    ax.set_xlabel("Hour of Year")
    ax.set_ylabel("Year")
    ax.set_title("Hourly Power Deficit: Unmet Heating Demand [MW]")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, extend="max")
    cbar.set_label("Deficit [MW]")

    # Vertical lines at month boundaries
    for ms in month_starts[1:]:
        ax.axvline(ms, color="white", lw=0.3, alpha=0.5)

    plt.tight_layout()
    save_case_study_plot(fig, "heatmap_hourly_deficit_all_years")

    # --- Summary statistics ---
    print(f"\n  {'Year':<6}{'Deficit h':<14}{'Deficit %':<12}"
          f"{'Max Deficit':<14}{'Total Deficit':<14}")
    print(f"  {'':6}{'':14}{'':12}{'[MW]':<14}{'[GWh]':<14}")
    print(f"  {'-' * 60}")
    for year in common_years:
        if year not in deficit_full:
            continue
        row = deficit_full[year]
        deficit_hours = int(np.sum(row > 0))
        deficit_pct = deficit_hours / len(row) * 100
        max_deficit_MW = np.max(row) / 1000
        total_deficit_GWh = np.sum(row) / 1e6
        print(f"  {year:<6}{deficit_hours:<14}{deficit_pct:<12.1f}"
              f"{max_deficit_MW:<14.2f}{total_deficit_GWh:<14.3f}")

    mean_deficit_hours = np.mean([np.sum(r > 0) for r in deficit_full.values()])
    mean_deficit_GWh = np.mean([np.sum(r) for r in deficit_full.values()]) / 1e6
    print(f"\n  Mean deficit hours:  {mean_deficit_hours:.0f} h/yr")
    print(f"  Mean total deficit:  {mean_deficit_GWh:.3f} GWh/yr")

    return deficit_matrix


# ============================================================
# STEP 8b: HOURLY EXPORT FOR LAKE FEEDBACK MODELLING
# ============================================================

def export_hourly_lake_interaction(df_hourly, demand_series, common_years,
                                   lake_area_m2):
    """Write an hourly CSV with potentials, heat-pump source temperatures and
    lake heat flux. Pure post-processing: reuses the columns computed in
    STEP 2/2b/CAP and the same formulas as STEP 8 – no result is changed.

    Sign convention: heat flux > 0 = heat withdrawn from the lake.
    Heat flux is referred to the total lake surface area [W/m2].
    """
    print("\n" + "=" * 60)
    print("  STEP 8b: HOURLY EXPORT FOR LAKE FEEDBACK")
    print("=" * 60)

    frames = []
    for year in common_years:
        # same timestamp alignment as STEP 5-9 (full year incl. 29 Feb / 31 Dec)
        pot, dem_kW = _align_year(df_hourly, demand_series, year)
        if len(pot) == 0:
            continue

        t_in = pot["temp_C"].values
        dT = pot["deltaT_hex"].values
        cop = pot["COP"].values
        ext = pot["extractable"].values
        lake_pot_kW = pot["lake_kW"].values                # after annual dT limit
        useful_unc_kW = pot["useful_uncapped_kW"].values
        useful_cap_kW = pot["useful_kW"].values            # capped at installed capacity

        # Actual operation (identical to STEP 8)
        covered_kW = np.minimum(dem_kW, useful_cap_kW)
        lake_extr_kW = np.where(
            ext, useful_to_lake(covered_kW, cop, cfg.q_loss), 0.0
        )

        # Water flow through the heat exchanger: Q = V * rho * cp * dT
        with np.errstate(divide="ignore", invalid="ignore"):
            flow_pot_m3s = np.where(dT > 0, lake_pot_kW * 1e3 / (RHO * CP * dT), 0.0)
            flow_op_m3s = np.where(dT > 0, lake_extr_kW * 1e3 / (RHO * CP * dT), 0.0)

        frames.append(pd.DataFrame({
            "year": year,
            "extractable": ext,
            # temperatures at the heat pump (source side)
            "T_lake_intake_C": t_in,
            "deltaT_hex_K": dT,
            "T_return_to_lake_C": t_in - dT,
            "COP": np.where(ext, cop, np.nan),
            # potentials
            "lake_heat_potential_kW": lake_pot_kW,
            "useful_heat_potential_uncapped_kW": useful_unc_kW,
            "useful_heat_potential_capped_kW": useful_cap_kW,
            # actual operation
            "demand_kW": dem_kW,
            "useful_heat_covered_kW": covered_kW,
            "lake_heat_extracted_kW": lake_extr_kW,
            "water_flow_potential_m3s": flow_pot_m3s,
            "water_flow_operation_m3s": flow_op_m3s,
            # heat flux per lake surface area
            "heat_flux_potential_W_m2": lake_pot_kW * 1e3 / lake_area_m2,
            "heat_flux_extracted_W_m2": lake_extr_kW * 1e3 / lake_area_m2,
        }, index=pot.index))

    df_out = pd.concat(frames)
    df_out.index.name = "datetime"

    csv_path = OUTPUT_DIR / "hourly_lake_interaction.csv"
    df_out.to_csv(csv_path, float_format="%.6g")

    # Consistency check against STEP 8 (Lake_Heat_Extracted_GWh)
    yearly = df_out.groupby("year")["lake_heat_extracted_kW"].sum() / 1e6
    print(f"  Lake area used for flux: {lake_area_m2/1e6:.2f} km2")
    print(f"  Mean lake heat extracted: {yearly.mean():.3f} GWh/yr (should match STEP 8)")
    print(f"  Mean heat flux (operation): "
          f"{df_out['heat_flux_extracted_W_m2'].mean():.3f} W/m2, "
          f"max {df_out['heat_flux_extracted_W_m2'].max():.3f} W/m2")
    print(f"  > Hourly table saved: {csv_path} ({len(df_out)} rows)")
    return df_out


# ============================================================
# STEP 9: CLIMATE-YEAR COMPARISON LSWSHP vs. ASHP (appendix)
# ============================================================

def _annuity_factor(wacc, lifetime):
    if wacc == 0:
        return 1.0 / lifetime
    return wacc * (1 + wacc) ** lifetime / ((1 + wacc) ** lifetime - 1)


def calc_cop_ashp(t_air_C, t_sink_C):
    """Hourly ASHP COP following When2Heat (incl. field correction factor)."""
    dt = np.maximum(np.asarray(t_sink_C, dtype="float64")
                    - np.asarray(t_air_C, dtype="float64"), ASHP_DT_MIN)
    return ASHP_CORRECTION * (ASHP_C0 + ASHP_C1 * dt + ASHP_C2 * dt ** 2)


def climate_year_comparison(df_hourly, demand_series, demand_sh, demand_dhw,
                            df_air, common_years, Installed_Capacity_kW):
    """Compare LSWSHP and ASHP for each simulated year at the case-study lake(s).

    Does NOT change any potential. It puts the constant COPs of the base case
    (COP_Air, COP_Water) into the context of climate variability.

    LSWSHP: hourly Lorenz COP; demand not covered by the lake is supplied by an
            electric boiler (BACKUP_EFF), so both systems deliver the same heat.
    ASHP:   hourly COP from air temperature (When2Heat), radiator heating curve
            for space heating and 50 °C for domestic hot water.
    Annual costs follow the annuity method of the EI (CAPEX sized as in the EI),
    pipeline costs are identical for all years and therefore omitted.
    """
    print("\n" + "=" * 60)
    print("  STEP 9: CLIMATE-YEAR COMPARISON LSWSHP vs. ASHP")
    print("=" * 60)
    print(f"  ASHP: When2Heat COP, radiator curve T_sink = {ASHP_SINK_SH_BASE:.0f} "
          f"{ASHP_SINK_SH_SLOPE:+.1f}*T_air, DHW {ASHP_SINK_DHW:.0f} °C, "
          f"correction {ASHP_CORRECTION}")

    af_dec = _annuity_factor(cfg.wacc_decentral, cfg.lifetime)
    af_cen = _annuity_factor(cfg.wacc_central, cfg.lifetime)

    t_air = df_air["airtemp_C"]

    rows = []
    for year in common_years:
        pot_yr, dem = _align_year(df_hourly, demand_series, year)  # dem: consumer side [kW]
        if len(pot_yr) == 0:
            continue
        idx = pot_yr.index
        dlv = pot_yr["useful_kW"].values                       # deliverable [kW]
        cop_w = pot_yr["COP"].values
        ext = pot_yr["extractable"].values

        # --- LSWSHP + electric backup ---
        covered = np.minimum(dem, dlv)
        deficit = dem - covered
        el_hp = np.where(ext, covered * (1 + cfg.q_loss) / cop_w, 0.0)
        el_backup = deficit * (1 + cfg.q_loss) / BACKUP_EFF
        out_hp = covered * (1 + cfg.q_loss)
        spf_w = out_hp.sum() / el_hp.sum() if el_hp.sum() > 0 else np.nan
        spf_w_sys = dem.sum() * (1 + cfg.q_loss) / (el_hp.sum() + el_backup.sum())

        # --- ASHP (decentral, no network losses) ---
        ta = t_air.reindex(idx).interpolate(limit_direction="both").values
        sh = demand_sh.reindex(idx).fillna(0).values
        dhw = demand_dhw.reindex(idx).fillna(0).values
        cop_sh = calc_cop_ashp(ta, ASHP_SINK_SH_BASE + ASHP_SINK_SH_SLOPE * ta)
        cop_dhw = calc_cop_ashp(ta, np.full_like(ta, ASHP_SINK_DHW))
        el_ashp = sh / cop_sh + dhw / cop_dhw
        spf_a = dem.sum() / el_ashp.sum()

        E_h = dem.sum()                                       # kWh/a
        E_w = E_h * (1 + cfg.q_loss)
        capex_a = cfg.CAPEX_Air * (E_h / cfg.hours_Air) * af_dec
        capex_w = cfg.CAPEX_Water * (E_w / cfg.hours_Water) * af_cen

        A_ashp = capex_a + cfg.c_el_L * el_ashp.sum()
        A_lsw = capex_w + cfg.c_el_W * (el_hp.sum() + el_backup.sum())
        # base case: constant COPs, no backup (as in the national EI)
        A_ashp_base = capex_a + cfg.c_el_L * E_h / cfg.COP_Air
        A_lsw_base = capex_w + cfg.c_el_W * E_w / cfg.COP_Water

        rows.append({
            "Year": year,
            "Mean_T_air_C": round(float(np.mean(ta)), 2),
            "SPF_ASHP": round(spf_a, 3),
            "SPF_LSWSHP": round(spf_w, 3),
            "SPF_LSWSHP_incl_backup": round(spf_w_sys, 3),
            "Backup_share_pct": round(deficit.sum() / E_h * 100, 2),
            "El_ASHP_GWh": round(el_ashp.sum() / 1e6, 3),
            "El_LSWSHP_HP_GWh": round(el_hp.sum() / 1e6, 3),
            "El_backup_GWh": round(el_backup.sum() / 1e6, 3),
            "A_ASHP_MEUR": round(A_ashp / 1e6, 3),
            "A_LSWSHP_MEUR": round(A_lsw / 1e6, 3),
            "Advantage_MEUR": round((A_ashp - A_lsw) / 1e6, 3),
            "Advantage_base_MEUR": round((A_ashp_base - A_lsw_base) / 1e6, 3),
            # EI = advantage / pipeline annuity -> same factor applies to EI
            "EI_factor": round((A_ashp - A_lsw) / (A_ashp_base - A_lsw_base), 3),
        })

    df = pd.DataFrame(rows)

    print(f"\n  {'Year':<6}{'T_air':<8}{'SPF_A':<8}{'SPF_W':<8}{'SPF_W+B':<10}"
          f"{'Backup%':<9}{'Adv.':<9}{'EI fac.':<8}")
    print(f"  {'':6}{'[°C]':<8}{'':8}{'':8}{'':10}{'':9}{'[M€]':<9}{'':8}")
    print(f"  {'-' * 66}")
    for _, r in df.iterrows():
        print(f"  {int(r['Year']):<6}{r['Mean_T_air_C']:<8.1f}{r['SPF_ASHP']:<8.2f}"
              f"{r['SPF_LSWSHP']:<8.2f}{r['SPF_LSWSHP_incl_backup']:<10.2f}"
              f"{r['Backup_share_pct']:<9.1f}{r['Advantage_MEUR']:<9.2f}{r['EI_factor']:<8.2f}")
    print(f"  {'-' * 66}")
    print(f"  Mean  SPF_ASHP {df['SPF_ASHP'].mean():.2f} (range {df['SPF_ASHP'].min():.2f}-"
          f"{df['SPF_ASHP'].max():.2f}; base case COP_Air = {cfg.COP_Air})")
    print(f"        SPF_LSWSHP {df['SPF_LSWSHP'].mean():.2f} (range {df['SPF_LSWSHP'].min():.2f}-"
          f"{df['SPF_LSWSHP'].max():.2f}; base case COP_Water = {cfg.COP_Water})")
    print(f"        EI factor {df['EI_factor'].mean():.2f} (range {df['EI_factor'].min():.2f}-"
          f"{df['EI_factor'].max():.2f}; 1 = base case)")

    csv_path = OUTPUT_DIR / "climate_year_comparison_ASHP_LSWSHP.csv"
    df.to_csv(csv_path, index=False)
    print(f"  > Table saved: {csv_path}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    x = df["Year"].values
    ax = axes[0]
    ax.plot(x, df["SPF_ASHP"], "o-", color="#c00000", lw=2, label="ASHP (SPF)")
    ax.plot(x, df["SPF_LSWSHP"], "s-", color="#1f4e79", lw=2, label="LSWSHP (SPF)")
    ax.axhline(cfg.COP_Air, color="#c00000", ls=":", lw=1.5, label=f"Base case COP ASHP = {cfg.COP_Air}")
    ax.axhline(cfg.COP_Water, color="#1f4e79", ls=":", lw=1.5, label=f"Base case COP LSWSHP = {cfg.COP_Water}")
    ax.set_ylabel("Seasonal performance factor SPF [-]")
    ax.legend(loc="best", fontsize=12)
    ax.text(-0.02, 1.05, "A", transform=ax.transAxes, fontsize=22, fontweight="bold",
            va="bottom", ha="right")

    ax = axes[1]
    colors = np.where(df["EI_factor"] >= 1, "#228b22", "#dc143c")
    ax.bar(x, df["EI_factor"], color=colors, alpha=0.8)
    ax.axhline(1.0, color="black", ls="--", lw=1.5, label="Base case")
    ax.set_ylabel("Cost advantage LSWSHP\nrelative to base case [-]")
    ax.set_xlabel("Year")
    ax.legend(loc="best", fontsize=12)
    ax.text(-0.02, 1.05, "B", transform=ax.transAxes, fontsize=22, fontweight="bold",
            va="bottom", ha="right")
    plt.tight_layout()
    save_case_study_plot(fig, "climate_year_comparison_ASHP_LSWSHP")
    return df




# ============================================================
# MAIN – shared inputs (independent of scenario)
# ============================================================

def load_lake_geometry():
    """Volume and area of the (combined) lake."""
    lakes_gdf = read_lakes(cfg, cfg.target_crs)
    lakes_gdf["Hylak_id"] = lakes_gdf["Hylak_id"].astype(int)

    lake_vol_mcm = 0.0
    lake_area_km2 = 0.0
    print()
    for lid in LAKE_IDS:
        lake_row = lakes_gdf[lakes_gdf["Hylak_id"] == lid]
        if lake_row.empty:
            raise ValueError(f"Lake {lid} not found!")
        vol_hl = float(lake_row.iloc[0]["Vol_total"])        # HydroLAKES: million m3
        vol_override = LAKE_VOLUME_OVERRIDE_MCM.get(lid)
        vol = float(vol_override) if vol_override is not None else vol_hl
        area = lake_row.iloc[0].geometry.area / 1e6
        src = "hard-coded" if vol_override is not None else "HydroLAKES"
        print(f"  Lake ID {lid}: area {area:.2f} km2, volume {vol:.2f} Mio m3 ({src}"
              + (f", HydroLAKES: {vol_hl:.2f}" if vol_override is not None else "") + ")")
        lake_vol_mcm += vol
        lake_area_km2 += area

    if len(LAKE_IDS) > 1:
        print(f"  Combined ({len(LAKE_IDS)} lakes):")
    print(f"  Area: {lake_area_km2:.2f} km2")
    print(f"  Volume: {lake_vol_mcm:.2f} Mio m3 ({lake_vol_mcm/1e3:.4f} km3)")
    return lake_vol_mcm, lake_area_km2


def load_economic_demand_kWh():
    """Annual demand = economic potential (Heat_EI_ge1_MWh) of the lake(s)."""
    economic_csv_path = Path("Data/Processed/economic_potential_comparison.csv")
    df_economic = pd.read_csv(economic_csv_path)
    df_economic["Hylak_id"] = df_economic["Hylak_id"].astype(int)
    heat_ei_MWh = 0.0
    for lid in LAKE_IDS:
        lake_economic_row = df_economic[df_economic["Hylak_id"] == lid]
        if lake_economic_row.empty:
            print(f"  WARNING: Lake {lid} not in {economic_csv_path} -> 0 MWh economic heat")
            continue
        heat_lid = float(lake_economic_row.iloc[0]["Heat_EI_ge1_MWh"])
        print(f"  Lake ID {lid}: Heat_EI_ge1 = {heat_lid/1e3:.3f} GWh")
        heat_ei_MWh += heat_lid
    if heat_ei_MWh <= 0:
        raise ValueError(f"No economic heat for lakes {LAKE_IDS} in {economic_csv_path}")
    return heat_ei_MWh * 1e3  # MWh -> kWh


# ============================================================
# STEP 0: COMPARISON OF THE LAKE TEMPERATURE SERIES
# ============================================================

def compare_lake_temperatures(selected_year=None):
    """Compare the lake temperature without heat-pump feedback (reference)
    with the coupled HP-feedback runs (one per T_min).

    Difference convention: dT = T_hp_feedback(T_min) - T_no_hp_feedback
    (negative = lake cooled by heat extraction).
    Output in OUTPUT_BASE_DIR: plot, annual and monthly CSV tables.
    """
    print("\n" + "=" * 60)
    print("  STEP 0: LAKE TEMPERATURE COMPARISON (no feedback vs. HP feedback)")
    print("=" * 60)

    ref_csv = LAKE_TEMP_SCENARIOS["no_hp_feedback"][T_MIN_SCENARIOS[0]]
    df_ref, yrs = load_hourly_complete_years(ref_csv, "ref (no feedback)")
    series = {"ref": df_ref["temp_C"]}
    years = set(yrs)
    for t in T_MIN_SCENARIOS:
        d, y = load_hourly_complete_years(
            LAKE_TEMP_SCENARIOS["hp_feedback"][t], f"HP feedback Tmin{t:g}C")
        series[t] = d["temp_C"]
        years &= set(y)
    years = sorted(years)
    if not years:
        raise ValueError("No common complete years in the lake temperature series")

    df = pd.concat(series, axis=1, join="inner")
    df = df[df.index.year.isin(years)]
    df["year"] = df.index.year
    df["month"] = df.index.month
    for t in T_MIN_SCENARIOS:
        df["dT_%g" % t] = df[t] - df["ref"]

    # --- Overall statistics ---
    print(f"  Common years: {years[0]}-{years[-1]} ({len(years)} yrs, {len(df)} h)")
    print(f"\n  {'HP run':<12}{'mean dT':<10}{'min dT':<10}{'max dT':<10}{'RMSE':<8}{'r':<8}")
    print(f"  {'':12}{'[K]':<10}{'[K]':<10}{'[K]':<10}{'[K]':<8}")
    print(f"  {'-' * 56}")
    for t in T_MIN_SCENARIOS:
        dT = df["dT_%g" % t]
        print(f"  {f'Tmin {t:g} °C':<12}{dT.mean():<+10.3f}{dT.min():<+10.3f}"
              f"{dT.max():<+10.3f}{np.sqrt((dT ** 2).mean()):<8.3f}"
              f"{df['ref'].corr(df[t]):<8.4f}")

    # --- Annual table ---
    rows = []
    for y, g in df.groupby("year"):
        row = {"Year": y,
               "Mean_T_ref_C": g["ref"].mean(),
               "Min_T_ref_C": g["ref"].min()}
        for t in T_MIN_SCENARIOS:
            row[f"Mean_T_hp_Tmin{t:g}C"] = g[t].mean()
            row[f"Min_T_hp_Tmin{t:g}C"] = g[t].min()
            row[f"Mean_dT_Tmin{t:g}C_K"] = g["dT_%g" % t].mean()
            row[f"Min_dT_Tmin{t:g}C_K"] = g["dT_%g" % t].min()
            # hours at/below the extraction limit of this run
            row[f"h_below_{t:g}C_ref"] = int((g["ref"] <= t).sum())
            row[f"h_below_{t:g}C_hp"] = int((g[t] <= t).sum())
        rows.append(row)
    df_yr = pd.DataFrame(rows).round(3)

    hdr = "".join(f"{f'dT {t:g}C':<10}{f'<={t:g}C ref/hp':<16}" for t in T_MIN_SCENARIOS)
    print(f"\n  {'Year':<6}{'T_ref':<8}{'Tmin ref':<10}{hdr}")
    print(f"  {'-' * (24 + 26 * len(T_MIN_SCENARIOS))}")
    for _, r in df_yr.iterrows():
        h = ""
        for t in T_MIN_SCENARIOS:
            n_ref = int(r["h_below_%gC_ref" % t])
            n_hp = int(r["h_below_%gC_hp" % t])
            h += f"{r['Mean_dT_Tmin%gC_K' % t]:<+10.3f}{f'{n_ref}/{n_hp}':<16}"
        print(f"  {int(r['Year']):<6}{r['Mean_T_ref_C']:<8.2f}{r['Min_T_ref_C']:<10.2f}{h}")
    for t in T_MIN_SCENARIOS:
        print(f"  Mean hours <= {t:g} °C: ref {df_yr['h_below_%gC_ref' % t].mean():.0f} h/yr, "
              f"HP feedback (Tmin {t:g} °C) {df_yr['h_below_%gC_hp' % t].mean():.0f} h/yr")

    # --- Monthly table (mean over all years, spread across years) ---
    mon = {"Month": list(range(1, 13)),
           "Mean_T_ref_C": df.groupby("month")["ref"].mean().values}
    for t in T_MIN_SCENARIOS:
        ym = df.groupby(["year", "month"])["dT_%g" % t].mean().unstack("year")
        mon[f"Mean_T_hp_Tmin{t:g}C"] = df.groupby("month")[t].mean().values
        mon[f"Mean_dT_Tmin{t:g}C_K"] = ym.mean(axis=1).values
        mon[f"Min_year_dT_Tmin{t:g}C_K"] = ym.min(axis=1).values
        mon[f"Max_year_dT_Tmin{t:g}C_K"] = ym.max(axis=1).values
    df_mon = pd.DataFrame(mon).round(3)

    df_yr.to_csv(OUTPUT_BASE_DIR / "lake_temperature_comparison_annual.csv", index=False)
    df_mon.to_csv(OUTPUT_BASE_DIR / "lake_temperature_comparison_monthly.csv", index=False)
    print(f"  > Tables saved: lake_temperature_comparison_annual/monthly.csv")

    # --- Plot ---
    if selected_year not in years:
        selected_year = years[len(years) // 2]
    d = df[df["year"] == selected_year]
    hoy = np.arange(len(_drop_feb29(d.index, d["ref"].values)))
    col_hp = dict(zip(T_MIN_SCENARIOS, ["#ff8c00", "#008080", "#6a0dad", "#228b22"]))
    ls_tmin = dict(zip(T_MIN_SCENARIOS, ["--", ":", "-.", "--"]))

    fig, axes = plt.subplots(3, 1, figsize=(16, 15),
                             gridspec_kw={"height_ratios": [2, 1.3, 1.5]})

    # A: all series, selected year
    ax = axes[0]
    ax.plot(hoy, _drop_feb29(d.index, d["ref"].values), lw=1.6,
            color="#1f4e79", label="Without HP feedback")
    for t in T_MIN_SCENARIOS:
        ax.plot(hoy, _drop_feb29(d.index, d[t].values), lw=1.4, color=col_hp[t],
                label=f"With HP feedback ($T_{{min}}$ = {t:g} °C)")
    for t in T_MIN_SCENARIOS:
        ax.axhline(t, color="#c00000", ls=ls_tmin[t], lw=1.8,
                   label=f"$T_{{min}}$ = {t:g} °C")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Lake Temperature [°C]")
    ax.set_title(f"{LAKE_NAME} – Lake temperature at {LAKE_DEPTH_LABEL} ({selected_year})",
                 fontweight="bold")
    ax.legend(loc="upper left", ncol=2, fontsize=13)
    set_month_xaxis(ax)
    ax.text(-0.02, 1.05, "A", transform=ax.transAxes, fontsize=22,
            fontweight="bold", va="bottom", ha="right")

    # B: hourly difference, selected year
    ax = axes[1]
    for t in T_MIN_SCENARIOS:
        ax.plot(hoy, _drop_feb29(d.index, d["dT_%g" % t].values), lw=1.0,
                color=col_hp[t], label=f"$T_{{min}}$ = {t:g} °C")
    ax.axhline(0, color="black", lw=1.0)
    ax.set_ylabel(r"$\Delta T$ [K]" "\n(HP − ref)")
    ax.legend(loc="lower right")
    set_month_xaxis(ax)
    ax.set_xlabel("Hour of Year")
    ax.text(-0.02, 1.05, "B", transform=ax.transAxes, fontsize=22,
            fontweight="bold", va="bottom", ha="right")

    # C: monthly mean difference, all years (bar = mean, whiskers = range of years)
    ax = axes[2]
    months = df_mon["Month"].values
    n = len(T_MIN_SCENARIOS)
    width = 0.8 / n
    for i, t in enumerate(T_MIN_SCENARIOS):
        m = df_mon[f"Mean_dT_Tmin{t:g}C_K"]
        x = months + (i - (n - 1) / 2) * width
        ax.bar(x, m, width, color=col_hp[t], alpha=0.8, edgecolor="k", lw=0.3,
               label=f"$T_{{min}}$ = {t:g} °C")
        ax.errorbar(x, m, yerr=[m - df_mon[f"Min_year_dT_Tmin{t:g}C_K"],
                                df_mon[f"Max_year_dT_Tmin{t:g}C_K"] - m],
                    fmt="none", ecolor="black", capsize=3, lw=1.0)
    ax.axhline(0, color="black", lw=1.0)
    ax.set_xticks(months)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel(r"Monthly mean $\Delta T$ [K]")
    ax.set_xlabel(f"Month (bars: mean {years[0]}–{years[-1]}, "
                  f"whiskers: range of individual years)")
    ax.legend(loc="lower right")
    ax.text(-0.02, 1.05, "C", transform=ax.transAxes, fontsize=22,
            fontweight="bold", va="bottom", ha="right")

    plt.tight_layout()
    stem = "lake_temperature_comparison"
    fig.savefig(OUTPUT_BASE_DIR / f"{stem}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_BASE_DIR / f"{stem}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  > Plot saved: {stem}.pdf and {stem}.svg")

    return df_yr, df_mon


# ============================================================
# SCENARIO RUN
# ============================================================

def run_scenario(temp_key, temp_csv, t_min, lake_vol_mcm, lake_area_km2,
                 static, df_air, complete_years_air, annual_demand_kWh,
                 demand_cache):
    """Run STEP 2-9 for one lake temperature series and one T_min.

    The helper functions read the module-level T_MIN_EXTRACTION and
    OUTPUT_DIR, so both are set here for the duration of the scenario.
    """
    global OUTPUT_DIR, T_MIN_EXTRACTION
    T_MIN_EXTRACTION = float(t_min)
    scenario_name = f"{temp_key}_Tmin{t_min:g}C"
    OUTPUT_DIR = OUTPUT_BASE_DIR / scenario_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 60)
    print(f"  SCENARIO: {scenario_name}")
    print(f"  Lake temperature: {temp_csv}")
    print(f"  T_min extraction: {T_MIN_EXTRACTION} °C")
    print("#" * 60)

    # --- Lake temperature data ---
    print()
    df_lake, complete_years_lake = load_hourly_complete_years(temp_csv, "Lake temp")
    print(f"  Lake temp data: {complete_years_lake[0]}-{complete_years_lake[-1]}")

    common_years = sorted(set(complete_years_lake) & set(complete_years_air))
    print(f"  Common years:   {common_years[0]}-{common_years[-1]} ({len(common_years)} yrs)")

    # =========================================================
    # STEP 2: Hourly dynamic potential
    # =========================================================
    df_hourly, df_annual, max_annual_kWh = compute_hourly_potential(
        lake_vol_mcm, df_lake, complete_years_lake
    )

    # =========================================================
    # STEP 2b: Useful-heat conversion
    # =========================================================
    df_hourly["lake_kW"] = df_hourly["capacity_kW"]
    df_hourly["useful_kW"] = np.where(
        df_hourly["extractable"],
        lake_to_useful(df_hourly["lake_kW"], df_hourly["COP"], cfg.q_loss),
        0.0,
    )

    n_years = df_hourly["year"].nunique()
    mean_cop_extractable = float(df_hourly.loc[df_hourly["extractable"], "COP"].mean()) if df_hourly["extractable"].any() else 0.0
    mean_lake_heat_GWh_yr = df_hourly.loc[df_hourly["extractable"], "lake_kW"].sum() / 1e6 / n_years
    mean_useful_heat_GWh_yr = df_hourly.loc[df_hourly["extractable"], "useful_kW"].sum() / 1e6 / n_years

    print("\n" + "=" * 60)
    print("  STEP 2b: USEFUL-HEAT CONVERSION")
    print("=" * 60)
    print(f"  Mean hourly COP (extractable hours): {mean_cop_extractable:.3f}")
    print(f"  Mean annual lake heat potential:    {mean_lake_heat_GWh_yr:.3f} GWh/yr")
    print(f"  Mean annual useful heat (uncapped):  {mean_useful_heat_GWh_yr:.3f} GWh/yr")

    # =========================================================
    # STEP 3: Outage analysis
    # =========================================================
    outage_info = analyse_outages(df_hourly, complete_years_lake)

    # =========================================================
    # STEP 4: Demand profile (cached: depends only on the years)
    # =========================================================
    print(f"\n  >>> ANNUAL_DEMAND_TOTAL_KWH from CSV = "
          f"{annual_demand_kWh:,.0f} kWh ({annual_demand_kWh/1e6:.3f} GWh)")
    Installed_Capacity_kW = annual_demand_kWh / cfg.hours_Water
    print(f"  >>> Corresponding Installed Capacity: {Installed_Capacity_kW/1e3:.3f} MW")

    key = tuple(common_years)
    if key not in demand_cache:
        demand_cache[key] = generate_demand_profile(
            annual_demand_kWh, df_air, common_years
        )
    else:
        print("\n  STEP 4: demand profile reused from previous scenario")
    demand_series, demand_sh, demand_dhw = demand_cache[key]

    # =========================================================
    # CAP: Limit hourly lake potential at installed capacity
    # =========================================================
    print(f"\n  >>> Capping lake thermal capacity at Installed Capacity: "
          f"{Installed_Capacity_kW/1e3:.3f} MW ({Installed_Capacity_kW:.1f} kW)")

    pre_cap_peak = df_hourly["useful_kW"].max()
    df_hourly["useful_uncapped_kW"] = df_hourly["useful_kW"]  # lake limit only
    df_hourly["useful_kW"] = df_hourly["useful_kW"].clip(upper=Installed_Capacity_kW)
    post_cap_peak = df_hourly["useful_kW"].max()

    hours_capped = int((df_hourly["useful_kW"] == Installed_Capacity_kW).sum())
    print(f"  Pre-cap peak:   {pre_cap_peak/1e3:.3f} MW")
    print(f"  Post-cap peak:  {post_cap_peak/1e3:.3f} MW")
    print(f"  Hours capped:   {hours_capped} "
          f"({hours_capped / len(df_hourly) * 100:.1f}%)")

    # =========================================================
    # STEP 5: Select year with outages and plot
    # =========================================================
    df_annual_common = df_annual[df_annual["year"].isin(common_years)]
    years_with_outages = df_annual_common[df_annual_common["hours_outage"] > 0]

    if SELECTED_YEAR_OVERRIDE is not None and SELECTED_YEAR_OVERRIDE in common_years:
        # fixed year -> identical year in all scenarios (comparable plots)
        selected_year = SELECTED_YEAR_OVERRIDE
        sel_hours = int(df_annual_common.loc[
            df_annual_common["year"] == selected_year, "hours_outage"].iloc[0])
        print(f"\n  >>> Selected year (override): {selected_year} ({sel_hours} h outage)")
    elif len(years_with_outages) > 0:
        mean_outage_hours = years_with_outages["hours_outage"].mean()
        closest_idx = (years_with_outages["hours_outage"] - mean_outage_hours).abs().idxmin()
        # Representative year: outage hours closest to the mean of all outage years
        selected_year = int(years_with_outages.loc[closest_idx, "year"])
        sel_hours = int(years_with_outages.loc[closest_idx, "hours_outage"])
        print(f"\n  >>> Selected year: {selected_year} "
              f"({sel_hours} h outage, mean of outage years: {mean_outage_hours:.0f} h)")
    else:
        median_idx = (df_annual_common["constrained_potential_GWh"] -
                      df_annual_common["constrained_potential_GWh"].median()).abs().idxmin()
        selected_year = int(df_annual_common.loc[median_idx, "year"])
        print(f"\n  >>> No outages found; using median year: {selected_year}")

    results = plot_potential_vs_demand(df_hourly, df_annual, demand_series, selected_year)

    # =========================================================
    # STEP 6 / 7: Heatmaps
    # =========================================================
    plot_daily_coverage_heatmap(df_hourly, demand_series, common_years)
    plot_hourly_deficit_heatmap(df_hourly, demand_series, common_years)

    # =========================================================
    # STEP 8: Annual Summary Table (Potential, Demand, Deficit, Utilization)
    # =========================================================
    print("\n" + "=" * 60)
    print("  STEP 8: ANNUAL SUMMARY TABLE")
    print("=" * 60)

    annual_table_rows = []
    for year in common_years:
        pot_yr, dem_kW = _align_year(df_hourly, demand_series, year)
        if len(pot_yr) == 0:
            continue

        pot_kW = pot_yr["useful_kW"].values
        unc_kW = pot_yr["useful_uncapped_kW"].values

        cop = pot_yr["COP"].values
        extractable = pot_yr["extractable"].values

        total_potential_GWh = pot_kW.sum() / 1e6
        total_demand_GWh = dem_kW.sum() / 1e6
        total_deficit_GWh = np.maximum(0, dem_kW - pot_kW).sum() / 1e6
        utilization_potential_GWh = total_demand_GWh - total_deficit_GWh

        # Deficit attribution (sums to total deficit):
        #   lake:     demand the lake cannot supply even with unlimited HP capacity
        #   capacity: demand the lake could supply but exceeds installed HP capacity
        deficit_lake_kW = np.maximum(0, dem_kW - unc_kW)
        deficit_cap_kW = np.maximum(0, np.minimum(dem_kW, unc_kW) - Installed_Capacity_kW)
        deficit_lake_GWh = deficit_lake_kW.sum() / 1e6
        deficit_cap_GWh = deficit_cap_kW.sum() / 1e6
        hours_no_extraction_deficit = int(((~extractable) & (dem_kW > 0)).sum())

        # Source side: lake heat actually extracted to supply the covered demand
        covered_kW = np.minimum(dem_kW, pot_kW)
        lake_extracted_kW = np.where(
            extractable, useful_to_lake(covered_kW, cop, cfg.q_loss), 0.0
        )
        lake_extracted_GWh = lake_extracted_kW.sum() / 1e6

        # Seasonal performance factor (SPF): HP heat output / electricity, weighted by actual operation
        hp_output_kW = np.where(extractable, covered_kW * (1 + cfg.q_loss), 0.0)
        electricity_kW = np.where(extractable, hp_output_kW / cop, 0.0)
        spf = hp_output_kW.sum() / electricity_kW.sum() if electricity_kW.sum() > 0 else np.nan

        annual_table_rows.append({
            "Year": year,
            "Useful_Heat_GWh": round(total_potential_GWh, 4),
            "Demand_GWh": round(total_demand_GWh, 4),
            "Deficit_GWh": round(total_deficit_GWh, 4),
            "Deficit_lake_GWh": round(deficit_lake_GWh, 4),
            "Deficit_capacity_GWh": round(deficit_cap_GWh, 4),
            "Hours_no_extraction": hours_no_extraction_deficit,
            "Utilization_Potential_GWh": round(utilization_potential_GWh, 4),
            "Coverage_pct": round(utilization_potential_GWh / total_demand_GWh * 100, 2),
            "Lake_Heat_Extracted_GWh": round(lake_extracted_GWh, 4),
            "Electricity_GWh": round(electricity_kW.sum() / 1e6, 4),
            "SPF": round(spf, 3),
        })

    df_annual_table = pd.DataFrame(annual_table_rows)

    print(f"\n  {'Year':<6}{'Useful':<10}{'Demand':<10}{'Deficit':<10}{'Def.lake':<10}"
          f"{'Def.cap':<10}{'Utiliz.':<10}{'Cov.%':<8}{'Lake extr.':<12}{'SPF':<6}")
    print(f"  {'':6}{'[GWh]':<10}{'[GWh]':<10}{'[GWh]':<10}{'[GWh]':<10}"
          f"{'[GWh]':<10}{'[GWh]':<10}{'':8}{'[GWh]':<12}{'':6}")
    print(f"  {'-' * 92}")
    for _, row in df_annual_table.iterrows():
        print(f"  {int(row['Year']):<6}{row['Useful_Heat_GWh']:<10.2f}"
              f"{row['Demand_GWh']:<10.2f}{row['Deficit_GWh']:<10.2f}"
              f"{row['Deficit_lake_GWh']:<10.2f}{row['Deficit_capacity_GWh']:<10.2f}"
              f"{row['Utilization_Potential_GWh']:<10.2f}{row['Coverage_pct']:<8.1f}"
              f"{row['Lake_Heat_Extracted_GWh']:<12.2f}{row['SPF']:<6.2f}")

    mean_lake_extracted = df_annual_table["Lake_Heat_Extracted_GWh"].mean()
    mean_potential = df_annual_table["Useful_Heat_GWh"].mean()
    mean_demand = df_annual_table["Demand_GWh"].mean()
    mean_deficit = df_annual_table["Deficit_GWh"].mean()
    mean_utilization = df_annual_table["Utilization_Potential_GWh"].mean()
    mean_def_lake = df_annual_table["Deficit_lake_GWh"].mean()
    mean_def_cap = df_annual_table["Deficit_capacity_GWh"].mean()
    mean_cov = df_annual_table["Coverage_pct"].mean()
    # SPF over all years (energy-weighted)
    hp_out_all = (df_annual_table["Utilization_Potential_GWh"] * (1 + cfg.q_loss)).sum()
    spf_all = hp_out_all / df_annual_table["Electricity_GWh"].sum()

    print(f"  {'-' * 92}")
    print(f"  {'Mean':<6}{mean_potential:<10.2f}{mean_demand:<10.2f}"
          f"{mean_deficit:<10.2f}{mean_def_lake:<10.2f}{mean_def_cap:<10.2f}"
          f"{mean_utilization:<10.2f}{mean_cov:<8.1f}{mean_lake_extracted:<12.2f}{spf_all:<6.2f}")
    print(f"\n  SPF (Lorenz, eta_L = {LORENZ_EFF}, DH {T_SUPPLY_C:.0f}/{T_RETURN_C:.0f} °C): "
          f"{spf_all:.2f}  (annual range {df_annual_table['SPF'].min():.2f}"
          f"-{df_annual_table['SPF'].max():.2f}; national constant COP_Water = {cfg.COP_Water})")
    if mean_deficit > 0:
        print(f"\n  Deficit share lake-limited:     {mean_def_lake / mean_deficit * 100:.1f}%")
        print(f"  Deficit share capacity-limited: {mean_def_cap / mean_deficit * 100:.1f}%")

    print(f"\n  >>> Mean Utilization Potential: {mean_utilization:.4f} GWh/yr "
          f"({mean_utilization/(annual_demand_kWh/1e6)*100:.1f}% of annual demand / economic potential)")

    csv_path = OUTPUT_DIR / "annual_potential_demand_utilization.csv"
    df_annual_table.to_csv(csv_path, index=False)
    print(f"  > Table saved: {csv_path}")

    # =========================================================
    # STEP 8b: Hourly export for lake feedback modelling
    # =========================================================
    export_hourly_lake_interaction(
        df_hourly, demand_series, common_years, lake_area_km2 * 1e6
    )

    # =========================================================
    # STEP 9: Climate-year comparison LSWSHP vs. ASHP (appendix)
    # =========================================================
    df_climate = climate_year_comparison(
        df_hourly, demand_series, demand_sh, demand_dhw,
        df_air, common_years, Installed_Capacity_kW
    )

    # =========================================================
    # Save combined results of this scenario
    # =========================================================
    df_ann_c = df_annual[df_annual["year"].isin(common_years)]
    combined_summary = {
        "scenario": scenario_name,
        "lake_temp_series": temp_key,
        "lake_temp_csv": str(temp_csv),
        "T_min_C": T_MIN_EXTRACTION,
        "years": f"{common_years[0]}-{common_years[-1]}",
        "n_years": len(common_years),
        "lake_ids": ";".join(str(i) for i in LAKE_IDS),
        "lake_volume_mcm": lake_vol_mcm,
        "lake_area_km2": lake_area_km2,
        "static_lake_heat_GWh": static["potential_GWh"],
        "annual_demand_GWh": annual_demand_kWh / 1e6,
        # all-years means
        "mean_lake_temp_C": df_ann_c["mean_temp_C"].mean(),
        "mean_min_lake_temp_C": df_ann_c["min_temp_C"].mean(),
        "mean_outage_h_yr": df_ann_c["hours_outage"].mean(),
        "max_outage_h_yr": df_ann_c["hours_outage"].max(),
        "max_outage_event_h": int(np.max(outage_info["outage_runs"])),
        "mean_useful_heat_GWh": mean_potential,
        "mean_covered_GWh": mean_utilization,
        "mean_coverage_pct": mean_cov,
        "min_coverage_pct": df_annual_table["Coverage_pct"].min(),
        "mean_deficit_GWh": mean_deficit,
        "mean_deficit_lake_GWh": mean_def_lake,
        "mean_deficit_capacity_GWh": mean_def_cap,
        "mean_lake_extracted_GWh": mean_lake_extracted,
        "SPF_all_years": spf_all,
        "SPF_min_year": df_annual_table["SPF"].min(),
        "SPF_max_year": df_annual_table["SPF"].max(),
        "mean_backup_share_pct": df_climate["Backup_share_pct"].mean(),
        "mean_EI_factor": df_climate["EI_factor"].mean(),
        "min_EI_factor": df_climate["EI_factor"].min(),
        # selected-year results (STEP 5), prefixed to avoid name clashes
        "selected_year": selected_year,
        **{f"sel_{k}": v for k, v in results.items() if k != "year"},
    }
    pd.DataFrame([combined_summary]).to_csv(
        OUTPUT_DIR / "combined_summary.csv", index=False
    )
    print(f"\n  > Scenario {scenario_name} complete -> {OUTPUT_DIR}")
    return combined_summary


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  COMBINED ANALYSIS: THERMAL POTENTIAL vs. HEATING DEMAND")
    print(f"  {LAKE_NAME} (Hylak_id={', '.join(str(i) for i in LAKE_IDS)})")
    print(f"  Scenarios: {list(LAKE_TEMP_SCENARIOS)} x T_min {T_MIN_SCENARIOS} °C")
    print("=" * 60)

    # --- Shared inputs (identical in all scenarios) ---
    lake_vol_mcm, lake_area_km2 = load_lake_geometry()

    print()
    df_air, complete_years_air = load_hourly_complete_years(AIR_TEMP_CSV, "Air temp")
    print(f"  Air temp data:  {complete_years_air[0]}-{complete_years_air[-1]}")

    static = compute_static_potential(lake_vol_mcm)   # STEP 1 (no temperature)
    annual_demand_kWh = load_economic_demand_kWh()

    # --- STEP 0: how do the two lake temperature series differ? ---
    compare_lake_temperatures(SELECTED_YEAR_OVERRIDE)

    # --- Scenario loop ---
    demand_cache = {}
    summaries = []
    for temp_key, csv_by_tmin in LAKE_TEMP_SCENARIOS.items():
        for t_min in T_MIN_SCENARIOS:
            summaries.append(run_scenario(
                temp_key, csv_by_tmin[t_min], t_min, lake_vol_mcm, lake_area_km2,
                static, df_air, complete_years_air, annual_demand_kWh,
                demand_cache,
            ))

    # --- Scenario comparison ---
    df_sc = pd.DataFrame(summaries)
    csv_path = OUTPUT_BASE_DIR / "scenario_comparison.csv"
    df_sc.to_csv(csv_path, index=False)

    print("\n" + "=" * 60)
    print("  SCENARIO COMPARISON (means over all common years)")
    print("=" * 60)
    print(f"  {'Scenario':<26}{'Outage':<9}{'Cov.%':<8}{'Deficit':<9}"
          f"{'Lake ex.':<10}{'SPF':<7}{'EI fac.':<8}")
    print(f"  {'':26}{'[h/yr]':<9}{'':8}{'[GWh]':<9}{'[GWh]':<10}{'':7}{'':8}")
    print(f"  {'-' * 77}")
    for _, r in df_sc.iterrows():
        print(f"  {r['scenario']:<26}{r['mean_outage_h_yr']:<9.0f}"
              f"{r['mean_coverage_pct']:<8.1f}{r['mean_deficit_GWh']:<9.2f}"
              f"{r['mean_lake_extracted_GWh']:<10.2f}{r['SPF_all_years']:<7.2f}"
              f"{r['mean_EI_factor']:<8.2f}")
    print(f"  > Table saved: {csv_path}")

    print("\n" + "=" * 60)
    print("  > COMBINED ANALYSIS COMPLETE")
    print(f"  Output: {OUTPUT_BASE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()