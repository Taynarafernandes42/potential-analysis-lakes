"""
Case Study: Lake Schwerin (Hylak_id=13463) – COMPLETE ANALYSIS

Includes:
  - Technical extractable potential (hourly, variable ΔT)
  - BDEW demand profile
  - Temperature-dependent COP
  - Supply-demand matching (hourly)
  - Outage analysis
  - EI spatial calculation
  - MANY plots
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from scipy import ndimage as ndi
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from shapely.geometry import box
from demandlib import bdew
import holidays as hol
import warnings

from config import Config

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

cfg = Config()

LAKE_ID = 13463
BUFFER_M = 2000
OUTPUT_DIR = Path(cfg.results_dir) / "CaseStudy_Schwerin"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAKE_TEMP_CSV = Path("Data/Raw/schwerin_temperature_13.5m_hourly.csv")
AIR_TEMP_CSV = Path("Data/Raw/schwerin_airtemp_1994_2020.csv")
ZENSUS_PATH = Path("Data/Raw/Zensus2022_Gebaeude_nach_Energietraeger_der_Heizung_100m-Gitter.csv")
DH_THRESHOLD = 0.5

# Temperature constraints
T_MIN_EXTRACTION_C = 3.0
DELTA_T_HEX_MAX = 5.0

# COP coefficients (water-source HP)
T_SUPPLY_C = 90.0
COP_C1 = 6.7
COP_C2 = 0.03697
COP_C3 = -0.03697
COP_C4 = 1.605e-7

# BDEW demand profile
BUILDING_TYPE = "MFH"
BUILDING_CLASS = 2
WIND_CLASS = 0
STATE = "MV"

# Physical constants
RHO = cfg.density_water
CP = cfg.cp_water
DELTA_T_TECH = cfg.deltaT_technical_extractable


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _pixel_center_xy(transform, rows, cols):
    x = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    y = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    return x, y


def _extent_from_transform(transform, shape):
    rows, cols = shape
    px = abs(transform.a)
    py = abs(transform.e)
    left = transform.c
    right = left + cols * px
    top = transform.f
    bottom = top - rows * py
    return [left, right, bottom, top]


def clip_to_extent(data, full_transform, bounds):
    xmin, ymin, xmax, ymax = bounds
    px = abs(full_transform.a)
    py = abs(full_transform.e)
    col_start = max(0, int((xmin - full_transform.c) / px))
    col_end = min(data.shape[1], int(np.ceil((xmax - full_transform.c) / px)))
    row_start = max(0, int((full_transform.f - ymax) / py))
    row_end = min(data.shape[0], int(np.ceil((full_transform.f - ymin) / py)))
    clipped = data[row_start:row_end, col_start:col_end]
    new_transform = rasterio.transform.from_origin(
        full_transform.c + col_start * px,
        full_transform.f - row_start * py,
        px, py
    )
    return clipped, new_transform


def calc_cop_water(temp_lake, t_supply=T_SUPPLY_C):
    """COP = f(T_lake, T_supply)"""
    temp_lake = np.asarray(temp_lake, dtype="float64")
    return COP_C1 + COP_C2 * temp_lake + COP_C3 * t_supply + COP_C4 * temp_lake * t_supply


def calc_delta_t_hex(lake_temp_C):
    """
    ΔT_hex:
      T ≤ 3°C  → 0 (no extraction)
      3 < T < 8 → T - 3
      T ≥ 8    → 5 (capped)
    """
    lake_temp_C = np.asarray(lake_temp_C, dtype="float64")
    return np.where(
        lake_temp_C <= T_MIN_EXTRACTION_C, 0.0,
        np.where(
            lake_temp_C < T_MIN_EXTRACTION_C + DELTA_T_HEX_MAX,
            lake_temp_C - T_MIN_EXTRACTION_C,
            DELTA_T_HEX_MAX,
        ),
    )


# ============================================================
# STEP 1: LOAD DATA & FILTER COMPLETE YEARS
# ============================================================

def load_data():
    """Load lake and air temperature, filter to common complete years."""
    print("\n[1] LOADING DATA")
    print("-" * 40)

    df_lake = pd.read_csv(LAKE_TEMP_CSV, parse_dates=["datetime"], index_col="datetime")
    df_air = pd.read_csv(AIR_TEMP_CSV, parse_dates=["datetime"], index_col="datetime")

    hours_air = df_air.groupby(df_air.index.year).size()
    complete_air = set(hours_air[hours_air >= 8760].index)
    hours_lake = df_lake.groupby(df_lake.index.year).size()
    complete_lake = set(hours_lake[hours_lake >= 8760].index)
    complete_years = sorted(complete_air & complete_lake)

    df_lake = df_lake[df_lake.index.year.isin(complete_years)]
    df_air = df_air[df_air.index.year.isin(complete_years)]

    print(f"  Lake temp: {LAKE_TEMP_CSV}")
    print(f"  Air temp:  {AIR_TEMP_CSV}")
    print(f"  Complete years: {complete_years[0]}–{complete_years[-1]} ({len(complete_years)} yrs)")
    print(f"  Lake temp range: {df_lake['temp_C'].min():.1f} – {df_lake['temp_C'].max():.1f} °C")

    return df_lake, df_air, complete_years


# ============================================================
# STEP 2: TECHNICAL POTENTIAL (hourly, variable ΔT)
# ============================================================

def compute_technical_potential(df_lake, lake_vol_km3, complete_years):
    """Compute hourly capacity with variable ΔT_hex and annual constraint."""
    print("\n[2] TECHNICAL POTENTIAL")
    print("-" * 40)

    seconds_per_year = 365.25 * 24 * 3600
    flow_m3s = lake_vol_km3 * 1e9 / seconds_per_year
    max_annual_kWh = lake_vol_km3 * 1e9 * RHO * CP * DELTA_T_TECH / 3.6e6

    print(f"  Flow rate: {flow_m3s:.4f} m³/s")
    print(f"  ΔT_tech: {DELTA_T_TECH} K → Max annual: {max_annual_kWh/1e6:.2f} GWh")

    df = df_lake.copy()
    df["deltaT_hex"] = calc_delta_t_hex(df["temp_C"].values)
    df["extractable"] = df["deltaT_hex"] > 0
    df["COP"] = calc_cop_water(df["temp_C"].values)

    # Hourly capacity [kW] – VARIABLE
    df["capacity_kW"] = np.where(
        df["extractable"],
        flow_m3s * RHO * CP * df["deltaT_hex"] / 1000.0,
        0.0,
    )

    # Annual constraint
    df["year"] = df.index.year
    annual_results = []

    for year in complete_years:
        mask_yr = df["year"] == year
        raw_kWh = df.loc[mask_yr, "capacity_kW"].sum()

        if raw_kWh > max_annual_kWh:
            scale = max_annual_kWh / raw_kWh
            df.loc[mask_yr, "capacity_kW"] *= scale
            actual_kWh = max_annual_kWh
        else:
            scale = 1.0
            actual_kWh = raw_kWh

        annual_results.append({
            "year": year,
            "raw_potential_GWh": raw_kWh / 1e6,
            "constrained_potential_GWh": actual_kWh / 1e6,
            "scale_factor": scale,
            "limited": raw_kWh > max_annual_kWh,
            "hours_extractable": int(df.loc[mask_yr, "extractable"].sum()),
            "hours_outage_temp": int((~df.loc[mask_yr, "extractable"]).sum()),
            "mean_temp_C": df.loc[mask_yr, "temp_C"].mean(),
            "min_temp_C": df.loc[mask_yr, "temp_C"].min(),
            "mean_deltaT_hex": float(df.loc[mask_yr & df["extractable"], "deltaT_hex"].mean()),
            "peak_capacity_MW": df.loc[mask_yr, "capacity_kW"].max() / 1000,
        })

    df_annual_pot = pd.DataFrame(annual_results)

    print(f"\n  {'Year':<6}{'Raw[GWh]':<10}{'Constr.[GWh]':<14}{'Limited':<9}"
          f"{'Outage[h]':<11}{'Peak[MW]'}")
    print(f"  {'-'*60}")
    for _, r in df_annual_pot.iterrows():
        print(f"  {int(r['year']):<6}{r['raw_potential_GWh']:<10.2f}"
              f"{r['constrained_potential_GWh']:<14.2f}"
              f"{'YES' if r['limited'] else 'no':<9}"
              f"{int(r['hours_outage_temp']):<11}{r['peak_capacity_MW']:<.2f}")

    mean_pot = df_annual_pot["constrained_potential_GWh"].mean()
    print(f"\n  Mean annual potential: {mean_pot:.2f} GWh/yr")
    print(f"  Mean outage hours:    {df_annual_pot['hours_outage_temp'].mean():.0f} h/yr")

    return df, df_annual_pot, max_annual_kWh, flow_m3s


# ============================================================
# STEP 3: DEMAND PROFILE (BDEW)
# ============================================================

def generate_demand_profile(complete_years, df_air, annual_demand_kWh):
    """Generate hourly BDEW demand profile."""
    print(f"\n[3] DEMAND PROFILE (BDEW)")
    print("-" * 40)
    print(f"  Annual demand: {annual_demand_kWh/1e6:.2f} GWh")

    heat_share = 0.85
    hw_share = 0.15
    profiles = []

    for year in complete_years:
        year_air = df_air.loc[df_air.index.year == year]
        holidays_dict = dict(hol.Germany(state=STATE, years=year))

        heating = bdew.HeatBuilding(
            year_air.index,
            holidays=holidays_dict,
            temperature=year_air["airtemp_C"],
            shlp_type=BUILDING_TYPE,
            building_class=BUILDING_CLASS,
            wind_class=WIND_CLASS,
            annual_heat_demand=annual_demand_kWh * heat_share,
            name="heating",
            ww_incl=False,
        ).get_bdew_profile()

        hotwater = bdew.HeatBuilding(
            year_air.index,
            holidays=holidays_dict,
            temperature=year_air["airtemp_C"],
            shlp_type=BUILDING_TYPE,
            building_class=BUILDING_CLASS,
            wind_class=WIND_CLASS,
            annual_heat_demand=annual_demand_kWh * hw_share,
            name="hotwater",
            ww_incl=True,
        ).get_bdew_profile()

        combined = (heating + hotwater).to_frame(name="demand_kW")
        profiles.append(combined)

    df_demand = pd.concat(profiles)
    print(f"  Generated {len(df_demand)} hourly values")
    print(f"  Peak demand: {df_demand['demand_kW'].max()/1000:.2f} MW")
    return df_demand


# ============================================================
# STEP 4: SUPPLY-DEMAND MATCHING (hourly)
# ============================================================

def hourly_matching(df_capacity, df_demand, complete_years):
    """Merge demand with capacity, compute outages and COP."""
    print(f"\n[4] HOURLY SUPPLY-DEMAND MATCHING")
    print("-" * 40)

    merged = df_demand.join(
        df_capacity[["temp_C", "deltaT_hex", "COP", "capacity_kW", "extractable"]],
        how="inner",
    )

    # Outage classification
    merged["outage_temp"] = ~merged["extractable"]
    merged["outage_capacity"] = (
        merged["extractable"] & (merged["capacity_kW"] < merged["demand_kW"])
    )
    merged["outage_any"] = merged["outage_temp"] | merged["outage_capacity"]

    # Supply/deficit
    merged["supplied_kW"] = np.minimum(merged["demand_kW"], merged["capacity_kW"])
    merged["deficit_kW"] = np.maximum(0, merged["demand_kW"] - merged["capacity_kW"])
    merged["surplus_kW"] = np.maximum(0, merged["capacity_kW"] - merged["demand_kW"])

    # Demand-weighted COP
    merged["year"] = merged.index.year
    supply_mask = merged["supplied_kW"] > 0
    if supply_mask.sum() > 0:
        cop_weighted = (merged.loc[supply_mask, "supplied_kW"].sum() /
                        (merged.loc[supply_mask, "supplied_kW"] /
                         merged.loc[supply_mask, "COP"]).sum())
    else:
        cop_weighted = np.nan

    # Annual breakdown
    annual_list = []
    for year in complete_years:
        ym = merged[merged["year"] == year]
        ym_supply = ym[ym["supplied_kW"] > 0]
        if len(ym_supply) > 0:
            cop_y = ym_supply["supplied_kW"].sum() / (ym_supply["supplied_kW"] / ym_supply["COP"]).sum()
        else:
            cop_y = np.nan

        annual_list.append({
            "year": year,
            "demand_MWh": ym["demand_kW"].sum() / 1000,
            "supplied_MWh": ym["supplied_kW"].sum() / 1000,
            "deficit_MWh": ym["deficit_kW"].sum() / 1000,
            "surplus_MWh": ym["surplus_kW"].sum() / 1000,
            "capacity_MWh": ym["capacity_kW"].sum() / 1000,
            "hours_outage_temp": int(ym["outage_temp"].sum()),
            "hours_outage_cap": int(ym["outage_capacity"].sum()),
            "hours_outage_any": int(ym["outage_any"].sum()),
            "COP_weighted": cop_y,
            "peak_demand_MW": ym["demand_kW"].max() / 1000,
            "peak_capacity_MW": ym["capacity_kW"].max() / 1000,
        })

    df_annual = pd.DataFrame(annual_list)
    df_annual["coverage_pct"] = df_annual["supplied_MWh"] / df_annual["demand_MWh"] * 100

    # Print summary
    n_total = len(merged)
    n_years = len(complete_years)
    print(f"  Hours analyzed: {n_total} ({n_years} years)")
    print(f"  Outage T≤{T_MIN_EXTRACTION_C}°C:    {int(merged['outage_temp'].sum())} "
          f"→ {merged['outage_temp'].sum()/n_years:.0f} h/yr")
    print(f"  Outage cap<demand: {int(merged['outage_capacity'].sum())} "
          f"→ {merged['outage_capacity'].sum()/n_years:.0f} h/yr")
    print(f"  Total outage:      {int(merged['outage_any'].sum())} "
          f"→ {merged['outage_any'].sum()/n_years:.0f} h/yr")
    print(f"  Demand-weighted COP: {cop_weighted:.3f}")
    print(f"  Mean coverage: {df_annual['coverage_pct'].mean():.1f}%")

    print(f"\n  {'Year':<6}{'Demand':<9}{'Supplied':<10}{'Deficit':<9}"
          f"{'Cover%':<8}{'Out_T':<7}{'Out_C':<7}{'COP'}")
    print(f"  {'-'*63}")
    for _, r in df_annual.iterrows():
        print(f"  {int(r['year']):<6}{r['demand_MWh']:<9.0f}{r['supplied_MWh']:<10.0f}"
              f"{r['deficit_MWh']:<9.0f}{r['coverage_pct']:<8.1f}"
              f"{int(r['hours_outage_temp']):<7}{int(r['hours_outage_cap']):<7}"
              f"{r['COP_weighted']:.3f}")

    stats = {
        "cop_weighted": cop_weighted,
        "mean_coverage_pct": df_annual["coverage_pct"].mean(),
        "mean_deficit_MWh": df_annual["deficit_MWh"].mean(),
        "mean_supplied_GWh": df_annual["supplied_MWh"].mean() / 1000,
        "outage_temp_h_yr": merged["outage_temp"].sum() / n_years,
        "outage_cap_h_yr": merged["outage_capacity"].sum() / n_years,
        "outage_any_h_yr": merged["outage_any"].sum() / n_years,
        "annual_breakdown": df_annual,
    }

    return merged, stats


# ============================================================
# STEP 5: SPATIAL EI CALCULATION
# ============================================================

def compute_spatial_ei(cfg, lake_geom, cop_weighted, buffer_m):
    """Compute EI map using demand-weighted COP."""
    print(f"\n[5] SPATIAL EI CALCULATION")
    print("-" * 40)
    print(f"  COP_Water = {cop_weighted:.3f}")

    xmin_l, ymin_l, xmax_l, ymax_l = lake_geom.bounds
    bounds = (xmin_l - buffer_m, ymin_l - buffer_m,
              xmax_l + buffer_m, ymax_l + buffer_m)

    with rasterio.open(cfg.heat_path) as src:
        heat_full = src.read(1).astype("float32")
        heat_nodata = src.nodata
        full_transform = src.transform
    if heat_nodata is not None:
        heat_full[heat_full == heat_nodata] = np.nan

    heat_clip, local_tf = clip_to_extent(heat_full, full_transform, bounds)
    rows_local, cols_local = heat_clip.shape
    px = abs(local_tf.a)
    py = abs(local_tf.e)

    boundary_raster = features.rasterize(
        shapes=[(lake_geom.boundary, 1)],
        out_shape=(rows_local, cols_local),
        transform=local_tf, fill=0, all_touched=True, dtype="uint8"
    )
    lake_interior = features.rasterize(
        shapes=[(lake_geom, 1)],
        out_shape=(rows_local, cols_local),
        transform=local_tf, fill=0, all_touched=False, dtype="uint8"
    )
    dist_to_shore_m = ndi.distance_transform_edt(
        ~(boundary_raster > 0), sampling=(py, px)
    )

    dh_mask = load_district_heating_mask(bounds, local_tf, (rows_local, cols_local))

    # Distance calculation
    D_local = np.full((rows_local, cols_local), np.nan, dtype="float64")
    valid_mask = (
        np.isfinite(heat_clip) & (heat_clip > 0)
        & (lake_interior == 0)
        & (dist_to_shore_m <= buffer_m)
    )

    if cfg.distance_mode == "pixel":
        D_local[valid_mask] = dist_to_shore_m[valid_mask]
    else:
        valid_rows, valid_cols = np.where(valid_mask)
        if valid_rows.size > 0:
            fx = max(1, int(round(cfg.hub_grid_m / px)))
            fy = max(1, int(round(cfg.hub_grid_m / py)))
            row_grp = np.arange(rows_local) // fy
            col_grp = np.arange(cols_local) // fx

            group_pixels = defaultdict(list)
            for r, c in zip(valid_rows, valid_cols):
                group_pixels[(row_grp[r], col_grp[c])].append((r, c))

            for (gr, gc), pix_list in group_pixels.items():
                rr = np.array([p[0] for p in pix_list], dtype=int)
                cc = np.array([p[1] for p in pix_list], dtype=int)
                heat_grp = heat_clip[rr, cc].astype("float64")
                x_grp, y_grp = _pixel_center_xy(local_tf, rr, cc)

                w = np.where(heat_grp > 0, heat_grp, 0.0)
                wsum = w.sum()
                if wsum <= 0:
                    x_h, y_h = float(np.mean(x_grp)), float(np.mean(y_grp))
                else:
                    x_h = float(np.sum(x_grp * w) / wsum)
                    y_h = float(np.sum(y_grp * w) / wsum)

                r_h = int(round((local_tf.f - y_h) / py))
                c_h = int(round((x_h - local_tf.c) / px))
                r_h = np.clip(r_h, 0, rows_local - 1)
                c_h = np.clip(c_h, 0, cols_local - 1)
                d_hub_shore = float(dist_to_shore_m[r_h, c_h])

                d_ph = np.hypot(x_grp - x_h, y_grp - y_h)
                total_MWh = float(heat_grp.sum())
                if cfg.hub_share_mode == "equal":
                    share = np.full(rr.shape, 1.0 / rr.size, dtype="float64")
                else:
                    share = heat_grp / (total_MWh if total_MWh > 0 else 1.0)
                D_local[rr, cc] = d_ph + share * d_hub_shore

    D_local = np.where(
        np.isfinite(D_local) & (D_local <= buffer_m),
        np.maximum(D_local, cfg.min_pipe_length_m), np.nan
    )

    # Existing DH → no pipe cost
    dh_and_valid = dh_mask & valid_mask
    n_dh_applied = int(dh_and_valid.sum())
    D_local[dh_and_valid] = cfg.min_pipe_length_m
    print(f"  Pixels with existing DH (no pipe cost): {n_dh_applied}")

    # EI calculation
    def _discounted_op_cost(c_el, E_h, COP, infl, lt):
        total = 0.0
        for year in range(1, int(lt) + 1):
            total += (c_el * E_h / COP) / ((1 + infl) ** year)
        return total

    mask_calc = valid_mask & np.isfinite(D_local)
    E_h_kWh = heat_clip[mask_calc].astype("float64") * 1000.0
    E_h_W = E_h_kWh * (1 + cfg.q_loss)

    disc_Air = _discounted_op_cost(cfg.c_el_L, E_h_kWh, cfg.COP_Air,
                                    cfg.inflation_rate, cfg.lifetime)
    disc_Water = _discounted_op_cost(cfg.c_el_W, E_h_W, cop_weighted,
                                      cfg.inflation_rate, cfg.lifetime)

    numerator = (
        (cfg.CAPEX_Air * E_h_kWh / cfg.hours_Air) + disc_Air
        - (cfg.CAPEX_Water * E_h_W / cfg.hours_Water) - disc_Water
    )
    denominator = cfg.c_pipe * D_local[mask_calc]
    denominator = np.where(denominator == 0, np.nan, denominator)

    EI_local = np.full((rows_local, cols_local), np.nan, dtype="float64")
    EI_local[mask_calc] = numerator / denominator
    EI_local[lake_interior > 0] = np.nan

    mask_viable = mask_calc & (EI_local >= cfg.ei_threshold)
    heat_viable_GWh = float(np.nansum(heat_clip[mask_viable])) / 1000
    heat_total_GWh = float(np.nansum(heat_clip[mask_calc])) / 1000

    mask_viable_dh = mask_viable & dh_mask
    mask_viable_no_dh = mask_viable & ~dh_mask
    heat_viable_dh_GWh = float(np.nansum(heat_clip[mask_viable_dh])) / 1000
    heat_viable_no_dh_GWh = float(np.nansum(heat_clip[mask_viable_no_dh])) / 1000
    n_viable = int(mask_viable.sum())
    n_total = int(mask_calc.sum())

    print(f"  Heat within {buffer_m/1000:.0f} km: {heat_total_GWh:.1f} GWh/yr")
    print(f"  Heat viable (EI≥1):  {heat_viable_GWh:.1f} GWh/yr")
    print(f"    ├─ existing DH:    {heat_viable_dh_GWh:.1f} GWh/yr")
    print(f"    └─ new pipe:       {heat_viable_no_dh_GWh:.1f} GWh/yr")
    print(f"  Viable pixels: {n_viable}/{n_total} ({n_viable/n_total*100:.1f}%)")

    return {
        "EI": EI_local,
        "heat": heat_clip,
        "mask_calc": mask_calc,
        "mask_viable": mask_viable,
        "lake_interior": lake_interior,
        "dh_mask": dh_mask,
        "transform": local_tf,
        "bounds": bounds,
        "D_local": D_local,
        "heat_viable_GWh": heat_viable_GWh,
        "heat_viable_dh_GWh": heat_viable_dh_GWh,
        "heat_viable_no_dh_GWh": heat_viable_no_dh_GWh,
        "heat_total_GWh": heat_total_GWh,
        "n_viable": n_viable,
        "n_total": n_total,
        "n_dh_applied": n_dh_applied,
    }


def load_district_heating_mask(bounds, local_transform, shape):
    """Load Zensus 2022 → rasterize DH cells."""
    if not ZENSUS_PATH.exists():
        return np.zeros(shape, dtype=bool)

    df = pd.read_csv(ZENSUS_PATH, encoding="utf-8", sep=";")
    energy_cols = ["Insgesamt_Energietraeger", "Fernwaerme"]
    df[energy_cols] = df[energy_cols].replace("–", 0).astype(int)
    df = df[df["Insgesamt_Energietraeger"] > 0].copy()
    df["dh_share"] = df["Fernwaerme"] / df["Insgesamt_Energietraeger"]
    df["has_dh"] = df["dh_share"] >= DH_THRESHOLD

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["x_mp_100m"], df["y_mp_100m"]),
        crs="EPSG:3035",
    )
    if cfg.target_crs != "EPSG:3035":
        gdf = gdf.to_crs(cfg.target_crs)

    xmin, ymin, xmax, ymax = bounds
    gdf = gdf.cx[xmin:xmax, ymin:ymax]
    dh_cells = gdf[gdf["has_dh"]].copy()

    if dh_cells.empty:
        return np.zeros(shape, dtype=bool)

    dh_cells["geometry"] = dh_cells.geometry.buffer(50, cap_style=3)
    dh_shapes = [(geom, 1) for geom in dh_cells.geometry if geom is not None]
    dh_raster = features.rasterize(
        shapes=dh_shapes, out_shape=shape, transform=local_transform,
        fill=0, all_touched=True, dtype="uint8",
    )
    return dh_raster > 0


# ============================================================
# PLOTS (MANY!)
# ============================================================

def plot_01_technical_potential_year(df_capacity, df_annual_pot, complete_years):
    """Representative year: capacity, ΔT_hex, temperature, outages."""
    median_year = int(df_annual_pot.loc[
        (df_annual_pot["constrained_potential_GWh"] -
         df_annual_pot["constrained_potential_GWh"].median()).abs().idxmin(), "year"])

    df_yr = df_capacity[df_capacity.index.year == median_year].copy()
    df_yr["hoy"] = np.arange(len(df_yr))

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 1.2, 1]})

    ax1 = axes[0]
    ax1.fill_between(df_yr["hoy"], df_yr["capacity_kW"] / 1000,
                     color="teal", alpha=0.6, linewidth=0, label="Capacity")
    outage = ~df_yr["extractable"].values
    ax1.fill_between(df_yr["hoy"], 0, df_yr["capacity_kW"].max() / 1000 * 1.1,
                     where=outage, color="red", alpha=0.15,
                     label=f"Outage (T ≤ {T_MIN_EXTRACTION_C}°C)", step="mid")
    ax1.set_ylabel("Capacity [MW]")
    yr_data = df_annual_pot[df_annual_pot["year"] == median_year].iloc[0]
    ax1.set_title(f"Technical Potential – Representative Year ({median_year})\n"
                  f"Annual: {yr_data['constrained_potential_GWh']:.2f} GWh | "
                  f"Peak: {yr_data['peak_capacity_MW']:.2f} MW | "
                  f"Outage: {int(yr_data['hours_outage_temp'])} h")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2 = axes[1]
    ax2.fill_between(df_yr["hoy"], df_yr["deltaT_hex"], color="orange", alpha=0.5, lw=0)
    ax2.axhline(DELTA_T_HEX_MAX, ls="--", color="grey", label=f"Max = {DELTA_T_HEX_MAX} K")
    ax2.set_ylabel("ΔT_hex [K]")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.plot(df_yr["hoy"], df_yr["temp_C"], lw=0.5, color="steelblue")
    ax3.axhline(T_MIN_EXTRACTION_C, color="red", ls="--", lw=1.5,
                label=f"T_min = {T_MIN_EXTRACTION_C}°C")
    ax3.axhline(T_MIN_EXTRACTION_C + DELTA_T_HEX_MAX, color="orange", ls="--", lw=1,
                label=f"Full ΔT = {T_MIN_EXTRACTION_C + DELTA_T_HEX_MAX}°C")
    ax3.set_ylabel("Lake T [°C]")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    _set_month_xaxis(axes)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_technical_potential_year.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_02_annual_potential_bar(df_annual_pot, max_annual_kWh):
    """Annual potential bar chart."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(df_annual_pot["year"], df_annual_pot["constrained_potential_GWh"],
           color="teal", edgecolor="k", lw=0.3, label="Extractable")
    excess = df_annual_pot["raw_potential_GWh"] - df_annual_pot["constrained_potential_GWh"]
    ax.bar(df_annual_pot["year"], excess,
           bottom=df_annual_pot["constrained_potential_GWh"],
           color="grey", alpha=0.4, edgecolor="k", lw=0.3, label="Capped by ΔT limit")
    ax.axhline(max_annual_kWh / 1e6, ls="--", color="red", lw=1.2,
               label=f"Max (ΔT_tech = {DELTA_T_TECH} K)")
    ax.axhline(df_annual_pot["constrained_potential_GWh"].mean(), ls=":", color="black",
               label=f"Mean = {df_annual_pot['constrained_potential_GWh'].mean():.2f} GWh")
    ax.set_xlabel("Year")
    ax.set_ylabel("Technical Potential [GWh/yr]")
    ax.set_title("Annual Technical Extractable Potential – Lake Schwerin")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_annual_potential_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_03_supply_demand_year(merged, complete_years):
    """Supply vs demand for representative year (median)."""
    annual_coverage = merged.groupby(merged.index.year)["supplied_kW"].sum()
    median_year = int(annual_coverage.index[
        (annual_coverage - annual_coverage.median()).abs().argmin()])

    df_yr = merged[merged.index.year == median_year].copy()
    df_yr["hoy"] = np.arange(len(df_yr))

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.2, 1]})

    ax1 = axes[0]
    ax1.plot(df_yr["hoy"], df_yr["demand_kW"] / 1000, lw=0.4, color="firebrick",
             label="Demand", zorder=2)
    ax1.plot(df_yr["hoy"], df_yr["capacity_kW"] / 1000, lw=0.4, color="teal",
             alpha=0.7, label="Capacity", zorder=1)
    ax1.fill_between(df_yr["hoy"], 0, df_yr["demand_kW"].max() / 1000,
                     where=df_yr["outage_temp"].values,
                     color="blue", alpha=0.1, label="Outage: T", step="mid")
    ax1.fill_between(df_yr["hoy"], 0, df_yr["demand_kW"].max() / 1000,
                     where=(df_yr["outage_capacity"] & ~df_yr["outage_temp"]).values,
                     color="orange", alpha=0.15, label="Outage: cap", step="mid")
    ax1.set_ylabel("Power [MW]")
    cov = df_yr["supplied_kW"].sum() / df_yr["demand_kW"].sum() * 100
    ax1.set_title(f"Supply-Demand Matching – {median_year} | Coverage: {cov:.1f}%")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2 = axes[1]
    ax2.fill_between(df_yr["hoy"], df_yr["deficit_kW"] / 1000, color="red", alpha=0.5, lw=0)
    ax2.set_ylabel("Deficit [MW]")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.plot(df_yr["hoy"], df_yr["temp_C"], lw=0.5, color="steelblue")
    ax3.axhline(T_MIN_EXTRACTION_C, color="red", ls="--", lw=1)
    ax3.set_ylabel("Lake T [°C]")
    ax3.grid(True, alpha=0.3)

    _set_month_xaxis(axes)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_supply_demand_year.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_04_supply_demand_all_years(merged):
    """Supply vs demand – ALL years."""
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.2, 1]})

    ax1 = axes[0]
    ax1.plot(merged.index, merged["demand_kW"] / 1000, lw=0.15, color="firebrick",
             label="Demand")
    ax1.plot(merged.index, merged["capacity_kW"] / 1000, lw=0.15, color="teal",
             alpha=0.7, label="Capacity")
    ax1.fill_between(merged.index, 0, merged["demand_kW"].max() / 1000,
                     where=merged["outage_temp"].values,
                     color="blue", alpha=0.08, step="mid")
    ax1.fill_between(merged.index, 0, merged["demand_kW"].max() / 1000,
                     where=(merged["outage_capacity"] & ~merged["outage_temp"]).values,
                     color="orange", alpha=0.12, step="mid")
    ax1.set_ylabel("Power [MW]")
    ax1.set_title("Supply-Demand – All Years")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2 = axes[1]
    ax2.fill_between(merged.index, merged["deficit_kW"] / 1000, color="red", alpha=0.5, lw=0)
    ax2.set_ylabel("Deficit [MW]")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.plot(merged.index, merged["temp_C"], lw=0.15, color="steelblue")
    ax3.axhline(T_MIN_EXTRACTION_C, color="red", ls="--", lw=1)
    ax3.set_ylabel("Lake T [°C]")
    ax3.set_xlabel("Time")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_supply_demand_all_years.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_05_annual_cop_outage(df_annual_match):
    """Annual COP + stacked outage hours."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1 = axes[0]
    bars = ax1.bar(df_annual_match["year"], df_annual_match["COP_weighted"],
                   color="teal", edgecolor="k", lw=0.3)
    mean_cop = df_annual_match["COP_weighted"].mean()
    ax1.axhline(mean_cop, ls="--", color="grey", label=f"Mean = {mean_cop:.3f}")
    ax1.set_ylabel("Demand-weighted COP")
    ax1.set_title(f"Annual COP & Outage (T_supply = {T_SUPPLY_C}°C)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(df_annual_match["year"], df_annual_match["hours_outage_temp"],
            color="blue", alpha=0.6, label=f"T ≤ {T_MIN_EXTRACTION_C}°C")
    ax2.bar(df_annual_match["year"], df_annual_match["hours_outage_cap"],
            bottom=df_annual_match["hours_outage_temp"],
            color="orange", alpha=0.6, label="Capacity < Demand")
    ax2.set_ylabel("Outage [h/yr]")
    ax2.set_xlabel("Year")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_annual_cop_outage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_06_duration_curves(merged, complete_years):
    """Duration curves: demand, capacity, deficit."""
    # Use one representative year
    median_year = int(pd.Series(complete_years).iloc[len(complete_years)//2])
    df_yr = merged[merged.index.year == median_year].copy()

    demand_sorted = np.sort(df_yr["demand_kW"].values)[::-1] / 1000
    capacity_sorted = np.sort(df_yr["capacity_kW"].values)[::-1] / 1000
    deficit_sorted = np.sort(df_yr["deficit_kW"].values)[::-1] / 1000
    hours = np.arange(1, len(demand_sorted) + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hours, demand_sorted, color="firebrick", lw=1.5, label="Demand")
    ax.plot(hours, capacity_sorted, color="teal", lw=1.5, label="Capacity")
    ax.fill_between(hours, deficit_sorted, color="red", alpha=0.3, label="Deficit")
    ax.set_xlabel("Hours (sorted)")
    ax.set_ylabel("Power [MW]")
    ax.set_title(f"Duration Curves – {median_year}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 8760)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_duration_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_07_monthly_statistics(merged):
    """Monthly boxplots: capacity, demand, COP."""
    merged_copy = merged.copy()
    merged_copy["month"] = merged_copy.index.month

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Capacity vs demand
    ax1 = axes[0]
    monthly_cap = [merged_copy[merged_copy["month"] == m]["capacity_kW"].values / 1000
                   for m in range(1, 13)]
    monthly_dem = [merged_copy[merged_copy["month"] == m]["demand_kW"].values / 1000
                   for m in range(1, 13)]
    bp1 = ax1.boxplot(monthly_cap, positions=np.arange(1, 13) - 0.2, widths=0.35,
                      patch_artist=True, boxprops=dict(facecolor="teal", alpha=0.5),
                      medianprops=dict(color="black"), flierprops=dict(markersize=1))
    bp2 = ax1.boxplot(monthly_dem, positions=np.arange(1, 13) + 0.2, widths=0.35,
                      patch_artist=True, boxprops=dict(facecolor="firebrick", alpha=0.5),
                      medianprops=dict(color="black"), flierprops=dict(markersize=1))
    ax1.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax1.set_ylabel("Power [MW]")
    ax1.set_title("Monthly Distribution: Capacity (teal) vs Demand (red)")
    ax1.grid(True, alpha=0.3)

    # COP
    ax2 = axes[1]
    monthly_cop = [merged_copy[(merged_copy["month"] == m) &
                               merged_copy["extractable"]]["COP"].values for m in range(1, 13)]
    bp3 = ax2.boxplot(monthly_cop, patch_artist=True,
                      boxprops=dict(facecolor="purple", alpha=0.4),
                      medianprops=dict(color="black"), flierprops=dict(markersize=1))
    ax2.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax2.set_ylabel("COP")
    ax2.set_title("Monthly COP Distribution")
    ax2.grid(True, alpha=0.3)

    # Deficit
    ax3 = axes[2]
    monthly_def = merged_copy.groupby([merged_copy.index.year, "month"])["deficit_kW"].sum() / 1000
    monthly_def = monthly_def.reset_index()
    monthly_def.columns = ["year", "month", "deficit_MWh"]
    monthly_def_pivot = monthly_def.groupby("month")["deficit_MWh"].apply(list)
    bp4 = ax3.boxplot([monthly_def_pivot.get(m, [0]) for m in range(1, 13)],
                      patch_artist=True, boxprops=dict(facecolor="red", alpha=0.4),
                      medianprops=dict(color="black"), flierprops=dict(markersize=1))
    ax3.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax3.set_ylabel("Deficit [MWh/month]")
    ax3.set_title("Monthly Energy Deficit")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_monthly_statistics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_08_cop_vs_temperature(merged):
    """Scatter: COP vs lake temperature."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sample = merged[merged["extractable"]].sample(min(50000, len(merged)), random_state=42)
    sc = ax.scatter(sample["temp_C"], sample["COP"], c=sample["demand_kW"] / 1000,
                    s=1, alpha=0.3, cmap="hot_r")
    plt.colorbar(sc, ax=ax, label="Demand [MW]")
    ax.set_xlabel("Lake Temperature [°C]")
    ax.set_ylabel("COP")
    ax.set_title("COP vs. Lake Temperature (color = demand)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_cop_vs_temperature.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_09_outage_heatmap(merged, complete_years):
    """Heatmap: outage per month/year."""
    merged_copy = merged.copy()
    merged_copy["month"] = merged_copy.index.month
    merged_copy["year"] = merged_copy.index.year

    pivot_temp = merged_copy.groupby(["year", "month"])["outage_temp"].sum().unstack(fill_value=0)
    pivot_cap = merged_copy.groupby(["year", "month"])["outage_capacity"].sum().unstack(fill_value=0)
    pivot_total = pivot_temp + pivot_cap

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    for ax, data, title, cmap in zip(
        axes,
        [pivot_temp, pivot_cap, pivot_total],
        ["Outage: T ≤ 3°C", "Outage: Capacity", "Total Outage"],
        ["Blues", "Oranges", "Reds"]
    ):
        im = ax.imshow(data.values, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels(data.index.astype(int))
        ax.set_xticks(range(12))
        ax.set_xticklabels(months)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="Hours")

    plt.suptitle("Outage Hours per Month/Year", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "09_outage_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_10_coverage_and_deficit(df_annual_match):
    """Annual coverage % + deficit energy."""
    fig, ax1 = plt.subplots(figsize=(12, 5))

    color1 = "teal"
    ax1.bar(df_annual_match["year"], df_annual_match["coverage_pct"],
            color=color1, alpha=0.7, edgecolor="k", lw=0.3)
    ax1.axhline(100, ls="--", color="green", lw=1, alpha=0.5)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Coverage [%]", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "red"
    ax2.plot(df_annual_match["year"], df_annual_match["deficit_MWh"] / 1000,
             "s-", color=color2, markersize=5, lw=1.5, label="Deficit [GWh]")
    ax2.set_ylabel("Deficit [GWh/yr]", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Annual Coverage & Energy Deficit")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "10_coverage_deficit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_11_viable_area_map(spatial, lake_geom):
    """Viable area map."""
    EI = spatial["EI"]
    mask_viable = spatial["mask_viable"]
    mask_calc = spatial["mask_calc"]
    heat = spatial["heat"]
    dh_mask = spatial["dh_mask"]
    tf = spatial["transform"]
    bounds = spatial["bounds"]
    extent = _extent_from_transform(tf, EI.shape)

    fig, ax = plt.subplots(figsize=(12, 10))

    h_disp = heat.copy()
    h_disp[h_disp <= 0] = np.nan
    ax.imshow(h_disp, extent=extent, origin="upper", cmap="Greys",
              alpha=0.3, norm=LogNorm(vmin=0.1, vmax=max(1, np.nanmax(h_disp))),
              interpolation="nearest")

    cat = np.zeros(EI.shape, dtype="uint8")
    not_viable = mask_calc & ~mask_viable & np.isfinite(EI)
    cat[not_viable] = 3
    cat[mask_viable & ~dh_mask] = 1
    cat[mask_viable & dh_mask] = 2

    rgba = np.zeros((*EI.shape, 4), dtype="float32")
    rgba[cat == 1] = [0.17, 0.63, 0.17, 0.7]
    rgba[cat == 2] = [0.58, 0.40, 0.74, 0.7]
    rgba[cat == 3] = [1.0, 0.6, 0.6, 0.25]
    ax.imshow(rgba, extent=extent, origin="upper", interpolation="nearest")

    lake_gdf = gpd.GeoDataFrame(geometry=[lake_geom], crs=cfg.target_crs)
    lake_gdf.plot(ax=ax, color="#4292c6", alpha=0.7, zorder=5)
    lake_gdf.boundary.plot(ax=ax, color="black", linewidth=2.5, zorder=6)

    lakes_gdf = gpd.read_file(cfg.lakes_path).to_crs(cfg.target_crs)
    lakes_gdf["Hylak_id"] = lakes_gdf["Hylak_id"].astype(int)
    bbox_poly = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=cfg.target_crs)
    other_lakes = gpd.sjoin(lakes_gdf[lakes_gdf["Hylak_id"] != LAKE_ID],
                            bbox_poly, how="inner", predicate="intersects")
    if not other_lakes.empty:
        other_lakes.plot(ax=ax, color="#9ecae1", alpha=0.5, edgecolor="gray",
                         linewidth=0.5, zorder=3)

    ax.legend(handles=[
        mpatches.Patch(color="#4292c6", label="Schwerin Lake"),
        mpatches.Patch(color="#9ecae1", label="Other lakes"),
        mpatches.Patch(color="#2ca02c", alpha=0.7, label="Viable – new pipe (EI ≥ 1)"),
        mpatches.Patch(color="#9467bd", alpha=0.7, label="Viable – existing DH"),
        mpatches.Patch(color="#ff9999", alpha=0.5, label="Not viable (EI < 1)"),
    ], loc="lower right", fontsize=9, framealpha=0.95)

    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    ax.set_title(f"Viable Supply Area – Lake Schwerin ({LAKE_ID})")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "11_viable_area_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_12_ei_histogram(spatial):
    """EI distribution histogram."""
    EI = spatial["EI"]
    mask_calc = spatial["mask_calc"]
    ei_vals = EI[mask_calc & np.isfinite(EI)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ei_vals, bins=100, color="steelblue", edgecolor="k", lw=0.3, alpha=0.7)
    ax.axvline(cfg.ei_threshold, color="red", ls="--", lw=2, label=f"Threshold (EI = {cfg.ei_threshold})")
    ax.axvline(np.median(ei_vals), color="orange", ls="--", lw=1.5,
               label=f"Median = {np.median(ei_vals):.2f}")
    ax.set_xlabel("Economic Index (EI)")
    ax.set_ylabel("Pixel count")
    ax.set_title("EI Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, max(5, np.percentile(ei_vals, 99)))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "12_ei_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_13_summary_dashboard(stats, spatial, df_annual_match):
    """Summary dashboard with key metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Case Study Summary – Lake Schwerin", fontsize=14, fontweight="bold")

    # 1: Key numbers
    ax = axes[0, 0]
    ax.axis("off")
    text = (
        f"Lake Volume: {stats['lake_vol_km3']:.4f} km³\n"
        f"Buffer: {BUFFER_M/1000:.0f} km\n"
        f"T_supply: {T_SUPPLY_C}°C\n"
        f"\n"
        f"COP (weighted): {stats['cop_weighted']:.3f}\n"
        f"Annual capacity: {stats['mean_annual_cap_GWh']:.2f} GWh\n"
        f"Assigned demand: {stats['assigned_demand_GWh']:.2f} GWh\n"
        f"Supplied: {stats['mean_supplied_GWh']:.2f} GWh\n"
        f"Coverage: {stats['mean_coverage_pct']:.1f}%\n"
        f"\n"
        f"Outage (T): {stats['outage_temp_h_yr']:.0f} h/yr\n"
        f"Outage (cap): {stats['outage_cap_h_yr']:.0f} h/yr\n"
        f"Deficit: {stats['mean_deficit_MWh']:.0f} MWh/yr"
    )
    ax.text(0.1, 0.95, text, transform=ax.transAxes, va="top", fontsize=11,
            fontfamily="monospace", bbox=dict(facecolor="lightyellow", alpha=0.8))
    ax.set_title("Key Metrics")

    # 2: Coverage per year
    ax = axes[0, 1]
    ax.bar(df_annual_match["year"], df_annual_match["coverage_pct"], color="teal", edgecolor="k", lw=0.3)
    ax.axhline(100, ls="--", color="green", alpha=0.5)
    ax.set_ylabel("Coverage [%]")
    ax.set_title("Annual Coverage")
    ax.grid(True, alpha=0.3)

    # 3: COP per year
    ax = axes[0, 2]
    ax.bar(df_annual_match["year"], df_annual_match["COP_weighted"], color="purple",
           edgecolor="k", lw=0.3, alpha=0.7)
    ax.set_ylabel("COP")
    ax.set_title("Annual COP")
    ax.grid(True, alpha=0.3)

    # 4: Outage stacked
    ax = axes[1, 0]
    ax.bar(df_annual_match["year"], df_annual_match["hours_outage_temp"],
           color="blue", alpha=0.6, label="T outage")
    ax.bar(df_annual_match["year"], df_annual_match["hours_outage_cap"],
           bottom=df_annual_match["hours_outage_temp"],
           color="orange", alpha=0.6, label="Cap outage")
    ax.set_ylabel("Hours")
    ax.set_title("Outage Hours")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5: Demand vs supplied
    ax = axes[1, 1]
    ax.bar(df_annual_match["year"], df_annual_match["demand_MWh"] / 1000,
           color="firebrick", alpha=0.4, label="Demand")
    ax.bar(df_annual_match["year"], df_annual_match["supplied_MWh"] / 1000,
           color="teal", alpha=0.7, label="Supplied")
    ax.set_ylabel("Energy [GWh]")
    ax.set_title("Demand vs Supplied")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6: EI spatial info
    ax = axes[1, 2]
    ax.axis("off")
    spatial_text = (
        f"Heat within {BUFFER_M/1000:.0f} km: {spatial['heat_total_GWh']:.1f} GWh\n"
        f"Viable (EI≥1): {spatial['heat_viable_GWh']:.1f} GWh\n"
        f"  existing DH: {spatial['heat_viable_dh_GWh']:.1f} GWh\n"
        f"  new pipe: {spatial['heat_viable_no_dh_GWh']:.1f} GWh\n"
        f"\n"
        f"Viable pixels: {spatial['n_viable']}/{spatial['n_total']}\n"
        f"  ({spatial['n_viable']/spatial['n_total']*100:.1f}%)\n"
        f"DH pixels: {spatial['n_dh_applied']}"
    )
    ax.text(0.1, 0.95, spatial_text, transform=ax.transAxes, va="top", fontsize=11,
            fontfamily="monospace", bbox=dict(facecolor="lightcyan", alpha=0.8))
    ax.set_title("Spatial EI Results")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "13_summary_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _set_month_xaxis(axes):
    """Set month labels on x-axis."""
    month_starts = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for ax in axes:
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_labels)
    axes[-1].set_xlabel("Month")


# ============================================================
# MAIN
# ============================================================

def main():
    # --- Load data ---
    df_lake, df_air, complete_years = load_data()

    # --- Lake geometry ---
    lakes_gdf = gpd.read_file(cfg.lakes_path).to_crs(cfg.target_crs)
    lakes_gdf["Hylak_id"] = lakes_gdf["Hylak_id"].astype(int)
    lake_row = lakes_gdf[lakes_gdf["Hylak_id"] == LAKE_ID]
    if lake_row.empty:
        raise ValueError(f"Lake {LAKE_ID} not found!")
    lake_geom = lake_row.iloc[0].geometry
    lake_vol_km3 = lake_row.iloc[0]["Vol_total"]
    print(f"  Lake area: {lake_geom.area / 1e6:.2f} km² | Volume: {lake_vol_km3:.4f} km³")

    # --- Step 2: Technical potential ---
    df_capacity, df_annual_pot, max_annual_kWh, flow_m3s = compute_technical_potential(
        df_lake, lake_vol_km3, complete_years
    )
    mean_annual_cap_GWh = df_annual_pot["constrained_potential_GWh"].mean()

    # --- Step 5: Spatial EI (first pass with mean COP) ---
    mean_cop_prelim = float(calc_cop_water(
        df_capacity.loc[df_capacity["extractable"], "temp_C"].mean()
    ))
    print(f"\n  Preliminary COP (mean lake T): {mean_cop_prelim:.3f}")

    spatial = compute_spatial_ei(cfg, lake_geom, mean_cop_prelim, BUFFER_M)

    # --- Step 3: Assign demand ---
    assigned_demand_GWh = min(spatial["heat_viable_GWh"], mean_annual_cap_GWh)
    assigned_demand_kWh = assigned_demand_GWh * 1e6
    print(f"\n  ► Assigned demand: min({spatial['heat_viable_GWh']:.1f}, "
          f"{mean_annual_cap_GWh:.2f}) = {assigned_demand_GWh:.2f} GWh/yr")

    # --- Step 3: Generate demand profile ---
    df_demand = generate_demand_profile(complete_years, df_air, assigned_demand_kWh)

    # --- Step 4: Hourly matching ---
    merged, match_stats = hourly_matching(df_capacity, df_demand, complete_years)

    # --- Update EI with final COP if different ---
    cop_weighted = match_stats["cop_weighted"]
    if abs(cop_weighted - mean_cop_prelim) / mean_cop_prelim > 0.01:
        print(f"\n  Recalculating EI with final COP: {mean_cop_prelim:.3f} → {cop_weighted:.3f}")
        spatial = compute_spatial_ei(cfg, lake_geom, cop_weighted, BUFFER_M)
        # Re-check assigned demand
        new_assigned = min(spatial["heat_viable_GWh"], mean_annual_cap_GWh)
        if abs(new_assigned - assigned_demand_GWh) / max(assigned_demand_GWh, 0.01) > 0.05:
            print(f"  Assigned demand changed: {assigned_demand_GWh:.2f} → {new_assigned:.2f} GWh")
            assigned_demand_GWh = new_assigned

    # --- Collect all stats ---
    all_stats = {
        "lake_vol_km3": lake_vol_km3,
        "flow_m3s": flow_m3s,
        "mean_annual_cap_GWh": mean_annual_cap_GWh,
        "assigned_demand_GWh": assigned_demand_GWh,
        "cop_weighted": cop_weighted,
        "mean_coverage_pct": match_stats["mean_coverage_pct"],
        "mean_supplied_GWh": match_stats["mean_supplied_GWh"],
        "mean_deficit_MWh": match_stats["mean_deficit_MWh"],
        "outage_temp_h_yr": match_stats["outage_temp_h_yr"],
        "outage_cap_h_yr": match_stats["outage_cap_h_yr"],
        "outage_any_h_yr": match_stats["outage_any_h_yr"],
    }

    # --- Print final summary ---
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    for k, v in all_stats.items():
        if isinstance(v, float):
            print(f"  {k:<30}: {v:.4f}")
        else:
            print(f"  {k:<30}: {v}")

    # --- Save CSVs ---
    print(f"\n[SAVING CSVs]")
    pd.DataFrame([all_stats]).to_csv(OUTPUT_DIR / "summary_statistics.csv", index=False)
    df_annual_pot.to_csv(OUTPUT_DIR / "annual_technical_potential.csv", index=False)
    match_stats["annual_breakdown"].to_csv(OUTPUT_DIR / "annual_supply_demand.csv", index=False)

    hourly_cols = ["demand_kW", "capacity_kW", "supplied_kW", "deficit_kW",
                   "surplus_kW", "temp_C", "deltaT_hex", "COP",
                   "outage_temp", "outage_capacity", "outage_any"]
    merged[hourly_cols].to_csv(OUTPUT_DIR / "hourly_data.csv")
    print(f"  Saved to {OUTPUT_DIR}")

    # --- PLOTS ---
    print(f"\n[GENERATING PLOTS]")
    df_annual_match = match_stats["annual_breakdown"]

    plot_01_technical_potential_year(df_capacity, df_annual_pot, complete_years)
    print("  ✓ 01 Technical potential (year)")

    plot_02_annual_potential_bar(df_annual_pot, max_annual_kWh)
    print("  ✓ 02 Annual potential bar")

    plot_03_supply_demand_year(merged, complete_years)
    print("  ✓ 03 Supply-demand (year)")

    plot_04_supply_demand_all_years(merged)
    print("  ✓ 04 Supply-demand (all years)")

    plot_05_annual_cop_outage(df_annual_match)
    print("  ✓ 05 Annual COP + outage")

    plot_06_duration_curves(merged, complete_years)
    print("  ✓ 06 Duration curves")

    plot_07_monthly_statistics(merged)
    print("  ✓ 07 Monthly statistics")

    plot_08_cop_vs_temperature(merged)
    print("  ✓ 08 COP vs temperature")

    plot_09_outage_heatmap(merged, complete_years)
    print("  ✓ 09 Outage heatmap")

    plot_10_coverage_and_deficit(df_annual_match)
    print("  ✓ 10 Coverage & deficit")

    plot_11_viable_area_map(spatial, lake_geom)
    print("  ✓ 11 Viable area map")

    plot_12_ei_histogram(spatial)
    print("  ✓ 12 EI histogram")

    plot_13_summary_dashboard(all_stats, spatial, df_annual_match)
    print("  ✓ 13 Summary dashboard")

    print(f"\n  ✓ ALL DONE → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()