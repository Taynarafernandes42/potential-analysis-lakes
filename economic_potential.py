from __future__ import annotations
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import Affine
from scipy import ndimage as ndi
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm

from config import lake_to_useful
from lake_io import read_lakes


# ============================================================
# Utilities
# ============================================================

def _pixel_center_xy(transform, rows, cols):
    x = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    y = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    return x, y


def _rowcol_from_xy(transform, x, y):
    col = (x - transform.c) / transform.a - 0.5
    row = (y - transform.f) / transform.e - 0.5
    return int(round(row)), int(round(col))


def _rasterize_boundaries(lakes_gdf, shape, transform):
    """Rasterize lake boundaries into a label array."""
    boundary_shapes = []
    for _, row in lakes_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        boundary_shapes.append((geom.boundary, int(row["Hylak_id"])))
    return features.rasterize(
        shapes=boundary_shapes, out_shape=shape, transform=transform,
        fill=0, all_touched=True, dtype="int32",
    )


def _annuity_factor(wacc, lifetime):
    """Kapitalwiedergewinnungsfaktor (Annuitätsfaktor).
    
    Wandelt eine Einmalinvestition in gleichmäßige jährliche Zahlungen um.
    Bei WACC=0 ergibt sich einfach 1/lifetime (lineare Abschreibung).
    """
    if wacc == 0:
        return 1.0 / lifetime
    return wacc * (1 + wacc) ** lifetime / ((1 + wacc) ** lifetime - 1)


# ============================================================
# Core EI Calculation (Annuity Method)
# ============================================================

def _calculate_ei_raster(cfg):
    """Compute Economic Index raster using Annuity method. Returns arrays + metadata.
    
    EI = (Jährliche Gesamtkosten Air - Jährliche Gesamtkosten Water) / Jährliche Pipeline-Kosten
    
    Alle Kosten (CAPEX und OPEX) werden als jährliche Äquivalente verglichen.
    CAPEX wird über den jeweiligen WACC annualisiert.
    """

    # Load heat raster
    with rasterio.open(cfg.heat_path) as src:
        heat_profile = src.profile.copy()
        heat_data = src.read(1)
        transform = src.transform
        crs_raster = src.crs
        nodata_heat = src.nodata

    # Load lakes
    lakes = read_lakes(cfg, crs_raster)

    # Rasterize lake boundaries and compute Euclidean distance transform
    boundary_labels = _rasterize_boundaries(lakes, heat_data.shape, transform)
    boundary_bool = boundary_labels > 0
    px, py = abs(transform.a), abs(transform.e)

    dist_to_shore_m, inds = ndi.distance_transform_edt(
        ~boundary_bool, sampling=(py, px), return_indices=True
    )
    nearest_rows, nearest_cols = inds
    nearest_lake_id_map = boundary_labels[nearest_rows, nearest_cols]

    rows, cols = heat_data.shape
    EI = np.full((rows, cols), np.nan, dtype="float64")
    HI = np.zeros((rows, cols), dtype="int32")
    D = np.full((rows, cols), np.nan, dtype="float32")
    HUB = np.zeros((rows, cols), dtype="int32")
    # Trunk length hub -> lake (hub mode) or distance to shore (pixel mode); used for the distance limit
    D_TRUNK = np.full((rows, cols), np.nan, dtype="float32")

    E_h_kWh = heat_data.astype("float64") * 1000.0  # MWh to kWh
    d_pixel = np.maximum(dist_to_shore_m, cfg.min_pipe_length_m)

    # --- Distance computation ---
    if cfg.distance_mode == "pixel":
        D = d_pixel.astype("float32")
        HI = nearest_lake_id_map.astype("int32")
        D_TRUNK = dist_to_shore_m.astype("float32")
    else:
        # Hub-based distance
        if nodata_heat is not None:
            valid_mask = (heat_data != nodata_heat) & np.isfinite(heat_data) & (heat_data > 0)
        else:
            valid_mask = np.isfinite(heat_data) & (heat_data > 0)

        valid_rows, valid_cols = np.where(valid_mask)

        if valid_rows.size > 0:
            fx = max(1, int(round(cfg.hub_grid_m / px)))
            fy = max(1, int(round(cfg.hub_grid_m / py)))
            row_grp = np.arange(rows) // fy
            col_grp = np.arange(cols) // fx

            group_pixels = defaultdict(list)
            for r, c in zip(valid_rows, valid_cols):
                group_pixels[(row_grp[r], col_grp[c])].append((r, c))

            hub_counter = 0
            for (gr, gc), pix_list in group_pixels.items():
                hub_counter += 1
                rr = np.array([p[0] for p in pix_list], dtype=int)
                cc = np.array([p[1] for p in pix_list], dtype=int)
                heat_grp = heat_data[rr, cc].astype("float64")
                total_MWh = float(heat_grp.sum())
                x_grp, y_grp = _pixel_center_xy(transform, rr, cc)

                # Heat-weighted hub centroid
                w = np.where(heat_grp > 0, heat_grp, 0.0)
                wsum = w.sum()
                if wsum <= 0:
                    x_h, y_h = float(np.mean(x_grp)), float(np.mean(y_grp))
                else:
                    x_h = float(np.sum(x_grp * w) / wsum)
                    y_h = float(np.sum(y_grp * w) / wsum)

                r_h, c_h = _rowcol_from_xy(transform, x_h, y_h)
                r_h = np.clip(r_h, 0, rows - 1)
                c_h = np.clip(c_h, 0, cols - 1)

                d_hub_shore = float(dist_to_shore_m[r_h, c_h])
                lake_id_hub = int(nearest_lake_id_map[r_h, c_h])
                d_ph = np.hypot(x_grp - x_h, y_grp - y_h)

                if cfg.hub_share_mode == "equal":
                    share = np.full(rr.shape, 1.0 / rr.size, dtype="float64")
                else:
                    denom = total_MWh if total_MWh > 0 else np.nan
                    share = heat_grp / denom

                D[rr, cc] = (d_ph + share * d_hub_shore).astype("float32")
                HI[rr, cc] = lake_id_hub
                HUB[rr, cc] = hub_counter
                D_TRUNK[rr, cc] = d_hub_shore

            # Fill non-valid pixels with pixel-based distance
            D[~valid_mask] = d_pixel[~valid_mask].astype("float32")
            HI[~valid_mask] = nearest_lake_id_map[~valid_mask].astype("int32")
            D_TRUNK[~valid_mask] = dist_to_shore_m[~valid_mask].astype("float32")

    # --- Apply max distance limit ---
    # The limit applies to the trunk pipeline from the hub to the lake (hub mode) or to the
    # distance to the shore (pixel mode), i.e. to a real distance, not to the effective
    # distance d_i (which contains only the demand-weighted share of the trunk pipeline).
    D = np.where(
        np.isfinite(D) & np.isfinite(D_TRUNK) & (D_TRUNK <= cfg.ei_max_dist_m),
        np.maximum(D, cfg.min_pipe_length_m), np.nan
    ).astype("float32")
    mask_valid = np.isfinite(D) & (HI > 0)

    E_h_L = E_h_kWh[mask_valid]                          # Heat demand for air-source [kWh]
    E_h_W = E_h_kWh[mask_valid] * (1 + cfg.q_loss)       # Heat demand incl. losses for water-source [kWh]

    # --- Annuity-based EI calculation ---
    # Annuity factors for CAPEX
    af_decentral = _annuity_factor(cfg.wacc_decentral, cfg.lifetime)
    af_central = _annuity_factor(cfg.wacc_central, cfg.lifetime)

    # Jährliche Gesamtkosten Air-source (dezentral):
    #   = CAPEX_Air * Leistung * Annuitätsfaktor_dezentral + Stromkosten/Jahr
    annual_capex_Air = cfg.CAPEX_Air * (E_h_L / cfg.hours_Air) * af_decentral
    annual_opex_Air = cfg.c_el_L * E_h_L / cfg.COP_Air
    annual_total_Air = annual_capex_Air + annual_opex_Air

    # Jährliche Gesamtkosten Water-source (zentral):
    #   = CAPEX_Water * Leistung * Annuitätsfaktor_zentral + Stromkosten/Jahr
    annual_capex_Water = cfg.CAPEX_Water * (E_h_W / cfg.hours_Water) * af_central
    annual_opex_Water = cfg.c_el_W * E_h_W / cfg.COP_Water
    annual_total_Water = annual_capex_Water + annual_opex_Water

    # Jährliche Pipeline-Kosten (zentral finanziert):
    #   = Einmalinvestition * Annuitätsfaktor_zentral
    annual_pipeline = cfg.c_pipe * D[mask_valid].astype("float64") * af_central
    annual_pipeline = np.where(annual_pipeline == 0, np.nan, annual_pipeline)

    # EI = Jährliche Einsparung / Jährliche Pipeline-Kosten
    EI[mask_valid] = (annual_total_Air - annual_total_Water) / annual_pipeline

    # Clean up invalid pixels
    if nodata_heat is not None:
        invalid = (heat_data == nodata_heat) | ~np.isfinite(heat_data)
    else:
        invalid = ~np.isfinite(heat_data)
    EI[invalid] = np.nan
    D[invalid] = np.nan
    HI[~np.isfinite(EI)] = 0

    # --- Save to GeoTIFF ---
    out_profile = heat_profile.copy()
    out_profile.update({"count": 5, "dtype": "float32", "nodata": np.nan})
    Path(cfg.ei_output_tif).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(cfg.ei_output_tif, "w", **out_profile) as dst:
        dst.write(EI.astype("float32"), 1)
        dst.write(HI.astype("float32"), 2)
        dst.write(D.astype("float32"), 3)
        dst.write(HUB.astype("float32"), 4)
        dst.write(D_TRUNK.astype("float32"), 5)

    print(f"  EI raster saved: {cfg.ei_output_tif}")
    return EI, HI, D, heat_data, transform, crs_raster


# ============================================================
# Per-Lake Aggregation
# ============================================================

def _load_ei_results(cfg):
    """Load previously computed EI raster."""
    with rasterio.open(cfg.ei_output_tif) as src:
        ei_data = src.read(1).astype("float32")
        hylak_data = src.read(2).astype("int32")
        dist_data = src.read(3).astype("float32")
        try:
            hub_data = src.read(4).astype("int32")
        except IndexError:
            hub_data = np.zeros_like(hylak_data)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    if nodata is not None:
        ei_data[ei_data == nodata] = np.nan
        dist_data[dist_data == nodata] = np.nan

    with rasterio.open(cfg.heat_path) as src:
        heat_data = src.read(1).astype("float32")
        heat_nodata = src.nodata
    if heat_nodata is not None:
        heat_data[heat_data == heat_nodata] = np.nan

    actual_dist_data = _compute_actual_distance_to_lake(cfg, transform, crs, heat_data.shape)
    return ei_data, hylak_data, dist_data, hub_data, heat_data, actual_dist_data, transform, crs


def _compute_actual_distance_to_lake(cfg, transform, crs, shape):
    lakes = read_lakes(cfg, crs)

    boundary_labels = _rasterize_boundaries(lakes, shape, transform)
    boundary_bool = boundary_labels > 0
    px, py = abs(transform.a), abs(transform.e)
    dist_to_shore_m = ndi.distance_transform_edt(~boundary_bool, sampling=(py, px))
    return dist_to_shore_m.astype("float32")


def _get_lake_ids(cfg, crs):
    """Load lake GeoDataFrame, apply NUTS filter, return filtered GDF."""
    lakes = read_lakes(cfg, crs)

    if cfg.nuts1_filter:
        nuts1 = gpd.read_file(cfg.nuts1_path).to_crs(crs)
        nuts1["NUTS_CODE"] = nuts1["NUTS_CODE"].astype(str)
        filtered = nuts1[nuts1["NUTS_CODE"].isin(cfg.nuts1_filter)]
        if filtered["NUTS_CODE"].duplicated().any():
            filtered = filtered[filtered["GF"] == 4]
        lakes = gpd.sjoin(lakes, filtered[["NUTS_CODE", "geometry"]],
                          how="inner", predicate="intersects")
        if "index_right" in lakes.columns:
            lakes = lakes.drop(columns="index_right")

    lakes["Hylak_id"] = lakes["Hylak_id"].astype(int)
    return lakes


def _compute_lake_heat(lake_id, ei_data, hylak_data, heat_data, threshold):
    mask = (hylak_data == lake_id) & (ei_data >= threshold) & np.isfinite(heat_data)
    total_MWh = float(np.nansum(heat_data[mask]))
    return {"Hylak_id": int(lake_id), "Heat_EI_ge1_MWh": total_MWh,
            "Heat_EI_ge1_GWh": total_MWh / 1000.0}


def _compute_heat_per_lake(cfg, ei_data, hylak_data, heat_data, lakes_gdf):
    """Compute total heat demand with EI >= threshold for each lake."""
    ids = [int(x) for x in lakes_gdf["Hylak_id"].tolist()]
    results = Parallel(n_jobs=cfg.n_jobs)(
        delayed(_compute_lake_heat)(hid, ei_data, hylak_data, heat_data, cfg.ei_threshold)
        for hid in tqdm(ids, desc="Per-lake heat aggregation")
    )
    return pd.DataFrame(results)


def _merge_with_potential(heat_df, cfg):
    """Merge heat demand (EI>=1) with technical potential to obtain economic potential."""
    pot_df = pd.read_csv(cfg.technical_extractable_csv)
    pot_df["Hylak_id"] = pot_df["Hylak_id"].astype(int)
    pot_df["technical_extractable_Thermal_Potential_GWh"] = pot_df["potential_Wh"] / 1e9
    df = pd.merge(
        heat_df,
        pot_df[["Hylak_id", "technical_extractable_Thermal_Potential_GWh"]],
        on="Hylak_id", how="outer",
    )
    df["Heat_EI_ge1_GWh"] = df["Heat_EI_ge1_GWh"].fillna(0)
    df["technical_extractable_Thermal_Potential_GWh"] = df["technical_extractable_Thermal_Potential_GWh"].fillna(0)
    df["technical_useful_Heat_GWh"] = lake_to_useful(
        df["technical_extractable_Thermal_Potential_GWh"], cfg.COP_Water, cfg.q_loss)
    df["Covered_Heat_GWh"] = np.minimum(df["Heat_EI_ge1_GWh"], df["technical_useful_Heat_GWh"])
    df["Lake_Heat_Extracted_GWh"] = df["Covered_Heat_GWh"] * (1 + cfg.q_loss) * (1 - 1 / cfg.COP_Water)

    total_GWh = df["Covered_Heat_GWh"].sum()
    n_no_demand = int((df["Heat_EI_ge1_GWh"] == 0).sum())
    df_with_demand = df[df["Heat_EI_ge1_GWh"] > 0]
    n_pot_gt = int((df_with_demand["technical_useful_Heat_GWh"] > df_with_demand["Heat_EI_ge1_GWh"]).sum())
    n_pot_lt = int((df_with_demand["technical_useful_Heat_GWh"] < df_with_demand["Heat_EI_ge1_GWh"]).sum())

    print(f"  Lakes with no economic potential: {n_no_demand}")
    print(f"  Lakes where technical > economic demand: {n_pot_gt}")
    print(f"  Lakes where technical < economic demand: {n_pot_lt}")
    print(f"  Total Economic Potential: {total_GWh / 1000:.2f} TWh/year")

    Path(cfg.economic_compare_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.economic_compare_csv, index=False)
    return df


REQUIRED_COMPARE_COLUMNS = {
    "Hylak_id",
    "Heat_EI_ge1_MWh",
    "Heat_EI_ge1_GWh",
    "technical_extractable_Thermal_Potential_GWh",
    "technical_useful_Heat_GWh",
    "Covered_Heat_GWh",
    "Lake_Heat_Extracted_GWh",
}


def _has_required_compare_columns(compare_df):
    """Return False for stale/incomplete cache files missing summary columns."""
    if not isinstance(compare_df, pd.DataFrame):
        return False
    return REQUIRED_COMPARE_COLUMNS.issubset(set(compare_df.columns))


def load_comparison_data(cfg):
    """Load economic comparison results created by economic_potential.calculate()."""
    compare_path = Path(cfg.economic_compare_csv)
    if compare_path.exists():
        compare_df = pd.read_csv(compare_path)
        if not _has_required_compare_columns(compare_df):
            missing = sorted(REQUIRED_COMPARE_COLUMNS.difference(compare_df.columns))
            raise ValueError(
                f"Economic comparison cache is stale/incomplete at {compare_path}. "
                f"Missing columns: {missing}. Re-run economic_potential.calculate(cfg)."
            )
        return compare_df
    raise FileNotFoundError(
        f"Economic comparison CSV not found: {compare_path}. "
        "Run economic_potential.calculate(cfg) first."
    )


# ============================================================
# Utilization Selection (Hub-Based)
# ============================================================

def _select_utilized_hubs(cfg, ei_data, hylak_data, hub_data, heat_data):
    """Select pixels/hubs to utilize, capped by each lake's technical potential."""
    mask_valid = (
        np.isfinite(ei_data) & np.isfinite(heat_data) & (heat_data > 0)
        & (hylak_data > 0) & (ei_data >= cfg.ei_threshold)
    )
    selected = np.zeros(ei_data.shape, dtype=bool)

    pot_df = pd.read_csv(cfg.technical_extractable_csv)
    pot_df["Hylak_id"] = pot_df["Hylak_id"].astype(int)
    pot_df["pot_MWh"] = lake_to_useful(pot_df["potential_Wh"] / 1e6, cfg.COP_Water, cfg.q_loss)
    pot_map = dict(zip(pot_df["Hylak_id"], pot_df["pot_MWh"]))

    lake_ids = np.unique(hylak_data[mask_valid])
    total_used = 0.0

    for lid in tqdm(lake_ids, desc="Utilization selection"):
        lid_int = int(lid)
        pot_MWh = pot_map.get(lid_int, 0.0)
        if pot_MWh <= 0:
            continue

        mask_lake = mask_valid & (hylak_data == lid_int)
        if not np.any(mask_lake):
            continue

        ei_l = ei_data[mask_lake]
        heat_l = heat_data[mask_lake]
        hubs_l = hub_data[mask_lake].astype(int)
        rows_l, cols_l = np.where(mask_lake)

        # Group by hub
        group_heat, group_ei, group_pixels = [], [], []

        mask_hub = hubs_l > 0
        if np.any(mask_hub):
            hubs_unique, inv_idx = np.unique(hubs_l[mask_hub], return_inverse=True)
            heat_per_hub = np.bincount(inv_idx, weights=heat_l[mask_hub]).astype(float)
            ei_wsum = np.bincount(inv_idx, weights=ei_l[mask_hub] * heat_l[mask_hub]).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                ei_per_hub = np.where(heat_per_hub > 0, ei_wsum / heat_per_hub, 0.0)
            idx_hub_pixels = np.where(mask_hub)[0]
            for j in range(len(hubs_unique)):
                pix_local = idx_hub_pixels[inv_idx == j]
                group_heat.append(heat_per_hub[j])
                group_ei.append(ei_per_hub[j])
                group_pixels.append(pix_local)

        # Single pixels (no hub assignment)
        mask_single = ~mask_hub
        if np.any(mask_single):
            for k in np.where(mask_single)[0]:
                group_heat.append(float(heat_l[k]))
                group_ei.append(float(ei_l[k]))
                group_pixels.append(np.array([k], dtype=int))

        if not group_heat:
            continue

        group_heat = np.array(group_heat)
        group_ei = np.array(group_ei)
        order = np.argsort(group_ei)[::-1]  # highest EI first

        used_MWh = 0.0
        sel_local = np.zeros(ei_l.shape, dtype=bool)
        for gid in order:
            remaining = pot_MWh - used_MWh
            if group_heat[gid] <= remaining + 1e-6:
                sel_local[group_pixels[gid]] = True
                used_MWh += group_heat[gid]
            if pot_MWh - used_MWh <= 1e-6:
                break

        selected[rows_l[sel_local], cols_l[sel_local]] = True
        total_used += used_MWh

    total_pot = sum(pot_map.get(int(lid), 0.0) for lid in lake_ids)
    if total_pot > 0:
        print(f"  Utilization: {total_used:,.0f} MWh of {total_pot:,.0f} MWh "
              f"({total_used / total_pot * 100:.1f}%)")
    return selected


# ============================================================
# Sensitivity Analysis (Annuity Method)
# ============================================================

def sensitivity_analysis(cfg, ei_data, hylak_data, dist_data, heat_data):
    """Vary EI parameters and compute resulting economic potential.
    Uses Annuity method: all costs annualized with respective WACC."""

    mask_base = np.isfinite(dist_data) & (hylak_data > 0) & np.isfinite(heat_data) & (heat_data > 0)
    E_h_base = heat_data[mask_base].astype("float64") * 1000.0  # kWh
    D_base = dist_data[mask_base].astype("float64")
    HI_base = hylak_data[mask_base].astype("int64")
    heat_MWh_base = heat_data[mask_base].astype("float64")

    # Technical potential map
    pot_df = pd.read_csv(cfg.technical_extractable_csv)
    pot_df["Hylak_id"] = pot_df["Hylak_id"].astype(int)
    pot_df["pot_GWh"] = pot_df["potential_Wh"] / 1e9
    pot_map = dict(zip(pot_df["Hylak_id"], pot_df["pot_GWh"]))

    params_base = {
        "c_el_L": cfg.c_el_L, "c_el_W": cfg.c_el_W,
        "c_pipe": cfg.c_pipe, "COP_Water": cfg.COP_Water, "COP_Air": cfg.COP_Air,
        "q_loss": cfg.q_loss, "CAPEX_Air": cfg.CAPEX_Air, "CAPEX_Water": cfg.CAPEX_Water,
        "hours_Air": cfg.hours_Air, "hours_Water": cfg.hours_Water,
        "wacc_decentral": cfg.wacc_decentral,
        "wacc_central": cfg.wacc_central,
    }

    results = []
    for param_name, (low, high) in cfg.sensitivity_param_ranges.items():
        values = np.linspace(low, high, cfg.sensitivity_n_steps)
        # Ensure base value is included in the sweep
        if param_name in params_base:
            idx_close = np.argmin(np.abs(values - params_base[param_name]))
            values[idx_close] = params_base[param_name]

        for val in values:
            p = params_base.copy()
            p[param_name] = float(val)

            # Annuity factors
            af_decentral = _annuity_factor(p["wacc_decentral"], cfg.lifetime)
            af_central = _annuity_factor(p["wacc_central"], cfg.lifetime)

            E_h_W = E_h_base * (1 + p["q_loss"])

            # Jährliche Gesamtkosten Air (dezentral)
            annual_capex_Air = p["CAPEX_Air"] * (E_h_base / p["hours_Air"]) * af_decentral
            annual_opex_Air = p["c_el_L"] * E_h_base / p["COP_Air"]
            annual_total_Air = annual_capex_Air + annual_opex_Air

            # Jährliche Gesamtkosten Water (zentral)
            annual_capex_Water = p["CAPEX_Water"] * (E_h_W / p["hours_Water"]) * af_central
            annual_opex_Water = p["c_el_W"] * E_h_W / p["COP_Water"]
            annual_total_Water = annual_capex_Water + annual_opex_Water

            # Jährliche Pipeline-Kosten (zentral finanziert)
            annual_pipeline = p["c_pipe"] * D_base * af_central
            annual_pipeline = np.where(annual_pipeline == 0, np.nan, annual_pipeline)

            # EI
            EI_var = (annual_total_Air - annual_total_Water) / annual_pipeline

            mask_ei = EI_var >= cfg.ei_threshold

            if np.any(mask_ei):
                uids, inv = np.unique(HI_base[mask_ei], return_inverse=True)
                sums = np.bincount(inv, weights=heat_MWh_base[mask_ei]).astype(float)
                covered = sum(
                    min(s / 1000.0, lake_to_useful(pot_map.get(int(uid), 0.0), p["COP_Water"], p["q_loss"]))
                    for uid, s in zip(uids, sums)
                )
            else:
                covered = 0.0

            results.append({"parameter": param_name, "tested_value": float(val),
                            "Utilization_GWh": covered})

    df = pd.DataFrame(results)
    df["Utilization_TWh"] = df["Utilization_GWh"] / 1000.0
    return df, params_base


# ============================================================
# Base Case Summary
# ============================================================

def _print_base_case_summary(cfg, compare_df, ei_data, hylak_data, heat_data):
    """Print a comprehensive summary of the economic potential base case."""

    total_covered_GWh = compare_df["Covered_Heat_GWh"].sum()
    total_covered_TWh = total_covered_GWh / 1000.0
    total_demand_GWh = compare_df["Heat_EI_ge1_GWh"].sum()
    total_technical_GWh = compare_df["technical_extractable_Thermal_Potential_GWh"].sum()
    total_technical_useful_GWh = compare_df["technical_useful_Heat_GWh"].sum()
    total_lake_extracted_GWh = compare_df["Lake_Heat_Extracted_GWh"].sum()

    n_lakes_total = len(compare_df)
    n_lakes_with_potential = int((compare_df["Covered_Heat_GWh"] > 0).sum())
    n_lakes_demand_limited = int(
        ((compare_df["Covered_Heat_GWh"] == compare_df["Heat_EI_ge1_GWh"])
         & (compare_df["Covered_Heat_GWh"] > 0)).sum()
    )

    # EI statistics
    mask_valid_ei = np.isfinite(ei_data) & (hylak_data > 0) & np.isfinite(heat_data) & (heat_data > 0)
    ei_valid = ei_data[mask_valid_ei]
    n_pixels_ei_ge1 = int((ei_valid >= cfg.ei_threshold).sum())
    n_pixels_total = int(mask_valid_ei.sum())
    ei_median = float(np.nanmedian(ei_valid)) if ei_valid.size > 0 else 0.0
    ei_mean = float(np.nanmean(ei_valid)) if ei_valid.size > 0 else 0.0

    # Annuity factors for display
    af_decentral = _annuity_factor(cfg.wacc_decentral, cfg.lifetime)
    af_central = _annuity_factor(cfg.wacc_central, cfg.lifetime)

    # Top lakes
    top_lakes = compare_df.nlargest(5, "Covered_Heat_GWh")[
        ["Hylak_id", "Heat_EI_ge1_GWh", "technical_extractable_Thermal_Potential_GWh", "Covered_Heat_GWh"]
    ]

    print("\n" + "=" * 65)
    print("  ECONOMIC POTENTIAL – BASE CASE SUMMARY (Annuity Method)")
    print("=" * 65)
    print(f"  Method:                           Annuity ")
    print(f"  Lifetime:                         {cfg.lifetime} years")
    print(f"  EI threshold:                     {cfg.ei_threshold}")
    print(f"  Max pipe distance:                {cfg.ei_max_dist_m:,.0f} m")
    print("-" * 65)
    print(f"  Total economic potential:         {total_covered_TWh:.2f} TWh/yr")
    print(f"                                    ({total_covered_GWh:,.0f} GWh/yr)")
    print(f"  Total heat demand (EI>={cfg.ei_threshold}):      {total_demand_GWh / 1000:.2f} TWh/yr")
    print(f"  Total technical potential (lake): {total_technical_GWh / 1000:.2f} TWh/yr")
    print(f"  Total technical potential (useful): {total_technical_useful_GWh / 1000:.2f} TWh/yr")
    print(f"  Lake heat extracted (econ.):      {total_lake_extracted_GWh / 1000:.2f} TWh/yr")
    print("-" * 65)
    print(f"  Lakes total:                      {n_lakes_total}")
    print(f"  Lakes with economic potential:    {n_lakes_with_potential}")
    print(f"  Lakes demand-limited:             {n_lakes_demand_limited}")
    if n_pixels_total > 0:
        print(f"  Pixels with EI >= {cfg.ei_threshold}:             {n_pixels_ei_ge1:,} / {n_pixels_total:,} "
              f"({n_pixels_ei_ge1 / n_pixels_total * 100:.1f}%)")
    print(f"  EI mean (valid pixels):           {ei_mean:.2f}")
    print(f"  EI median (valid pixels):         {ei_median:.2f}")
    print("-" * 65)
    print("  Top 5 lakes by economic potential:")
    print(f"  {'Hylak_id':>10} {'EI Demand GWh':>15} {'Tech. Pot. GWh':>20} {'Econ. Pot. GWh':>15}")
    for _, row in top_lakes.iterrows():
        print(f"  {int(row['Hylak_id']):>10} {row['Heat_EI_ge1_GWh']:>15.1f} "
              f"{row['technical_extractable_Thermal_Potential_GWh']:>20.1f} "
              f"{row['Covered_Heat_GWh']:>15.1f}")
    print("=" * 65 + "\n")


# ============================================================
# Minimum project size (applied at the end of the assessment)
# ============================================================

def _hp_capacity_kW(cfg, covered_GWh):
    """Heat-pump thermal output capacity per lake [kW] (same sizing as CAPEX in the EI)."""
    return covered_GWh * 1e6 * (1 + cfg.q_loss) / cfg.hours_Water


def apply_min_project_size(cfg, compare_df):
    """Final step: exclude lakes whose project would be smaller than the minimum size.

    All previous steps (EI, demand matching, sensitivity) are computed without this
    threshold. Lakes below it keep their values in Covered_Heat_GWh, but contribute
    zero to the final Economic_Potential_GWh.
    """
    df = compare_df.copy()
    df["HP_Capacity_kW"] = _hp_capacity_kW(cfg, df["Covered_Heat_GWh"])
    df["Above_Min_Size"] = df["HP_Capacity_kW"] >= cfg.min_project_capacity_kW
    df["Economic_Potential_GWh"] = np.where(df["Above_Min_Size"], df["Covered_Heat_GWh"], 0.0)

    has_pot = df["Covered_Heat_GWh"] > 0
    n_before = int(has_pot.sum())
    n_after = int((has_pot & df["Above_Min_Size"]).sum())
    pot_before = df["Covered_Heat_GWh"].sum() / 1000.0
    pot_after = df["Economic_Potential_GWh"].sum() / 1000.0
    cap = df.loc[has_pot, "HP_Capacity_kW"]

    print("\n" + "=" * 65)
    print(f"  MINIMUM PROJECT SIZE  (>= {cfg.min_project_capacity_kW:.0f} kW, "
          f"FLH = {cfg.hours_Water} h)")
    print("=" * 65)
    print(f"  HP capacity per lake  median / mean / max: "
          f"{cap.median():,.0f} kW / {cap.mean():,.0f} kW / {cap.max()/1000:,.1f} MW")
    print(f"  Lakes with economic potential:  {n_before:>6} -> {n_after:>6} "
          f"({n_after / n_before * 100 if n_before else 0:.1f}%)")
    print(f"  Economic potential [TWh/yr]:    {pot_before:>6.2f} -> {pot_after:>6.2f} "
          f"({pot_after / pot_before * 100 if pot_before else 0:.1f}%)")
    print(f"  Lake heat extracted [TWh/yr]:   "
          f"{df['Lake_Heat_Extracted_GWh'].sum()/1000:>6.2f} -> "
          f"{df.loc[df['Above_Min_Size'], 'Lake_Heat_Extracted_GWh'].sum()/1000:>6.2f}")
    print("=" * 65 + "\n")

    df.to_csv(cfg.economic_compare_csv, index=False)
    return df


def plot_project_size_distribution(cfg, compare_df):
    """Histogram of HP capacity per lake, split by limiting factor, with minimum size."""
    df = compare_df[compare_df["Covered_Heat_GWh"] > 0].copy()
    if df.empty:
        return
    cap_MW = df["HP_Capacity_kW"] / 1000.0
    limited_by = np.where(df["Covered_Heat_GWh"] >= df["Heat_EI_ge1_GWh"] - 1e-9,
                          "demand", "lake")
    thr_MW = cfg.min_project_capacity_kW / 1000.0
    lang = cfg.plot_language

    plt.rcParams.update({"font.size": 14, "font.family": "serif", "mathtext.fontset": "cm"})
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.logspace(np.log10(max(cap_MW.min(), 1e-4)), np.log10(cap_MW.max()), 40)
    groups = [("lake", "#1976d2"), ("demand", "#f58220")]
    if lang == "en":
        labels = [f"{g}-limited (n={int((limited_by == g).sum())})" for g, _ in groups]
    else:
        names = {"lake": "potenzialbegrenzt", "demand": "bedarfsbegrenzt"}
        labels = [f"{names[g]} (n={int((limited_by == g).sum())})" for g, _ in groups]
    ax.hist([cap_MW[limited_by == g] for g, _ in groups], bins=bins, stacked=True,
            color=[c for _, c in groups], label=labels)

    ax.axvspan(bins[0], thr_MW, color="grey", alpha=0.15, zorder=0)
    n_after = int((cap_MW >= thr_MW).sum())
    pot_share = df.loc[cap_MW >= thr_MW, "Covered_Heat_GWh"].sum() / df["Covered_Heat_GWh"].sum() * 100
    thr_label = (f"min. project size {cfg.min_project_capacity_kW:.0f} kW: "
                 f"{n_after} lakes, {pot_share:.0f}% of potential")
    if lang != "en":
        thr_label = (f"Mindestgröße {cfg.min_project_capacity_kW:.0f} kW: "
                     f"{n_after} Seen, {pot_share:.0f}% des Potenzials")
    ax.axvline(thr_MW, color="k", ls="--", lw=1.8, label=thr_label)

    ax.set_xscale("log")
    xlabel = (f"Heat-pump capacity per lake [MW]  (FLH = {cfg.hours_Water} h)" if lang == "en"
              else f"Wärmepumpenleistung je See [MW]  (VLS = {cfg.hours_Water} h)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of lakes" if lang == "en" else "Anzahl Seen")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=11, loc="upper right")
    plt.tight_layout()

    out = Path(cfg.project_size_plot_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Project size plot saved: {out}")


# ============================================================
# Plots
# ============================================================

def plot_scatter(cfg, ei_data, dist_data, heat_data, utilized_mask=None, actual_dist_data=None):
    """EI scatter plot: effective connection distance and actual lake distance."""
    base_font = 11
    factor = 1 / 0.65
    plt.rcParams.update({
        "font.size": base_font * factor,
        "axes.labelsize": base_font * factor,
        "legend.fontsize": base_font * factor,
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

    mask_common = np.isfinite(ei_data) & np.isfinite(heat_data) & (heat_data > 0) & (ei_data > 0)
    if utilized_mask is not None:
        mask_common &= utilized_mask

    if actual_dist_data is None:
        raise ValueError("actual_dist_data must be provided for scatter plotting.")

    ei_vals = np.clip(ei_data[mask_common], 0, 5)
    dist_eff_vals = dist_data[mask_common] / 1000.0
    dist_actual_vals = actual_dist_data[mask_common] / 1000.0
    heat_vals = heat_data[mask_common]

    mask_gray = ei_vals < 1
    mask_color = ei_vals >= 1

    fig, axes = plt.subplots(1, 2, figsize=(24, 8), sharey=True)
    x_labels = ["Effective connection distance", "Distance to lake"]
    dist_arrays = [dist_eff_vals, dist_actual_vals]

    for ax, dist_vals, label in zip(axes, dist_arrays, x_labels):
        if np.any(mask_gray):
            ax.scatter(dist_vals[mask_gray], heat_vals[mask_gray],
                       c="lightgray", s=25, alpha=0.7, linewidths=0, label="EI < 1")
        if np.any(mask_color):
            sc = ax.scatter(dist_vals[mask_color], heat_vals[mask_color],
                            c=ei_vals[mask_color], cmap="plasma_r",
                            s=25, alpha=1, linewidths=0, norm=LogNorm(vmin=1, vmax=5))
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label("EI (clipped at 5)")
            cbar.set_ticks([1, 2, 3, 4, 5])
            cbar.ax.yaxis.set_major_formatter(plt.ScalarFormatter())

        ax.set_xlim(left=0, right=10)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(f"{label} [km]")
        if label == x_labels[0]:
            ax.set_ylabel("Heat Demand [MWh/yr]")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(markerscale=3, loc="upper right")
        ax.set_title(label)

    plt.tight_layout()
    out = Path(cfg.economic_scatter_plot_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, format="png", dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".svg"), format="svg", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_map(cfg, compare_df):
    """Map of economic potential per lake with colored districts."""
    target_crs = cfg.target_crs
    lang = cfg.plot_language
    base_font = 11
    factor = 1 / 0.5
    plt.rcParams.update({
        "font.size": base_font * factor,
        "axes.titlesize": base_font * factor * 1.2,
        "axes.labelsize": base_font * factor,
        "legend.fontsize": base_font * factor * 0.9,
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

    lakes_gdf = read_lakes(cfg, target_crs)
    germany_gdf = gpd.read_file(cfg.germany_path).to_crs(target_crs)
    states_gdf = gpd.read_file(cfg.nuts3_path).to_crs(target_crs)

    if cfg.nuts1_filter:
        nuts1 = gpd.read_file(cfg.nuts1_path).to_crs(target_crs)
        nuts1["NUTS_CODE"] = nuts1["NUTS_CODE"].astype(str)
        filtered_nuts = nuts1[nuts1["NUTS_CODE"].isin(cfg.nuts1_filter)]
        if not filtered_nuts.empty:
            germany_gdf = filtered_nuts
        states_gdf["NUTS_CODE"] = states_gdf["NUTS_CODE"].astype(str)
        states_gdf = states_gdf[states_gdf["NUTS_CODE"].str[:3].isin(cfg.nuts1_filter)]

    # Final economic potential (after minimum project size) if available
    val_col = "Economic_Potential_GWh" if "Economic_Potential_GWh" in compare_df.columns \
        else "Covered_Heat_GWh"

    # Merge comparison data with lake geometries
    df = compare_df.copy()
    df["Hylak_id"] = df["Hylak_id"].astype(int)
    lakes_gdf["Hylak_id"] = lakes_gdf["Hylak_id"].astype(int)
    lakes_with_data = lakes_gdf.merge(
        df[["Hylak_id", val_col]], on="Hylak_id", how="left"
    )
    lakes_with_data[val_col] = lakes_with_data[val_col].fillna(0)
    lakes_with_data["centroid"] = lakes_with_data.geometry.centroid

    if cfg.nuts1_filter:
        lakes_with_data = gpd.sjoin(
            lakes_with_data, filtered_nuts[["NUTS_CODE", "geometry"]],
            how="inner", predicate="intersects"
        )
        if "index_right" in lakes_with_data.columns:
            lakes_with_data = lakes_with_data.drop(columns="index_right")

    # Highlight districts containing lakes with economic potential
    relevant = lakes_with_data[lakes_with_data[val_col] > 0.0001]
    name_col = next((c for c in ["NUTS_NAME", "GEN", "NAME_3", "NAME"]
                     if c in states_gdf.columns), None)

    if name_col and not relevant.empty:
        if "index_right" in relevant.columns:
            relevant = relevant.drop(columns="index_right")
        joined = gpd.sjoin(relevant, states_gdf[[name_col, "geometry"]], how="inner", predicate="intersects")
        agg = joined.groupby(name_col)[val_col].sum().reset_index()
        agg.columns = [name_col, "Total_GWh"]
        states_gdf = states_gdf.merge(agg, on=name_col, how="left")
        states_gdf["Total_GWh"] = states_gdf["Total_GWh"].fillna(0)
        states_gdf["highlight"] = np.select(
            [states_gdf["Total_GWh"] > 100, states_gdf["Total_GWh"] > 0.0001],
            [2, 1], default=0
        )
    else:
        states_gdf["highlight"] = 0

    # Plot
    fig, ax = plt.subplots(figsize=(12, 15))
    germany_gdf.boundary.plot(ax=ax, color="grey", linewidth=0.8, zorder=0)
    states_gdf[states_gdf["highlight"] == 2].plot(ax=ax, color="#41ab5d", alpha=0.5, zorder=0)
    states_gdf[states_gdf["highlight"] == 1].plot(ax=ax, color="#c7e9c0", alpha=0.7, zorder=0)
    states_gdf.boundary.plot(ax=ax, color="lightgrey", linewidth=0.3, zorder=1)
    lakes_with_data.plot(ax=ax, color="#08306b", edgecolor="none", markersize=3, zorder=4)

    bubble_color = "#f58223"
    scale_factor = 2500

    for _, row in lakes_with_data.iterrows():
        val = row[val_col]
        if val <= 0 or row["centroid"] is None:
            continue
        x, y = row["centroid"].x, row["centroid"].y

        if cfg.label_bodensee and int(row["Hylak_id"]) == cfg.bodensee_id:
            label = f"{val:.0f} GWh/yr" if lang == "en" else f"{val:.0f} GWh/a"
            ax.text(x, y, label, ha="center", va="center",
                    fontsize=base_font * factor * 0.9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7), zorder=7)
            continue

        radius = np.sqrt(val) * scale_factor
        ax.add_patch(plt.Circle((x, y), radius, color=bubble_color, alpha=0.4,
                                ec=bubble_color, lw=0.2, zorder=6))

    # Set extent and add reference circles for legend
    xmin, ymin, xmax, ymax = germany_gdf.total_bounds
    width, height = xmax - xmin, ymax - ymin
    ax.set_xlim(xmin - 0.05 * width, xmax + 0.4 * width)
    ax.set_ylim(ymin - 0.05 * height, ymax + 0.2 * height)

    example_values = [5, 10, 50]
    x_ref = xmax + 0.1 * width
    text_x = x_ref + max(np.sqrt(v) * scale_factor for v in example_values) * 1.2
    y_ref = ymin + 0.3 * height
    for j, val in enumerate(example_values):
        r = np.sqrt(val) * scale_factor
        if j == 0:
            y_ref += r
        else:
            y_ref += np.sqrt(example_values[j - 1]) * scale_factor + r + 0.03 * height
        ax.add_patch(plt.Circle((x_ref, y_ref), r, color=bubble_color, alpha=0.4, ec="none"))
        ax.text(text_x, y_ref, f"{val:.0f} GWh/yr", va="center")

    # Legend
    econ_label = "Economic potential" if lang == "en" else "Wirtschaftliches Potential"
    lakes_label = "Lakes" if lang == "en" else "Seen"
    dist_lo = "Districts with potential (>100 kWh)" if lang == "en" else "Kreise mit Potential (>100 kWh)"
    dist_hi = "Districts with potential (>100 GWh)" if lang == "en" else "Kreise mit Potential (>100 GWh)"
    ax.legend(handles=[
        mpatches.Patch(color=bubble_color, label=econ_label),
        mpatches.Patch(color="#08306b", label=lakes_label),
        mpatches.Patch(color="#c7e9c0", label=dist_lo),
        mpatches.Patch(color="#41ab5d", label=dist_hi),
    ], loc="upper left", frameon=True)

    germany_gdf.boundary.plot(ax=ax, color="grey", linewidth=0.8, zorder=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    plt.tight_layout()

    Path(cfg.economic_map_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cfg.economic_map_plot_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(cfg.economic_map_plot_path.replace(".pdf", ".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(cfg, df_sens, params_base):
    """Normalized sensitivity plot: % parameter variation vs % economic potential variation."""
    base_font = 11
    factor = 1 / 0.5
    plt.rcParams.update({
        "font.size": base_font * factor * 0.9,
        "axes.labelsize": base_font * factor,
        "legend.fontsize": base_font * factor * 0.9,
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

    # --- Configuration ---
    math_labels = {
        "c_el_L": r"$c_{el,ASHP}$",
        "c_el_W": r"$c_{el,LSWSHP}$",
        "c_pipe": r"$c_{pipe}$",
        "COP_Water": r"$COP_{LSWSHP}$",
        "COP_Air": r"$COP_{ASHP}$",
        "q_loss": r"$q_{loss}$",
        "CAPEX_Air": r"$CAPEX_{ASHP}$",
        "CAPEX_Water": r"$CAPEX_{LSWSHP}$",
        "hours_Air": r"$h_{ASHP}$",
        "hours_Water": r"$h_{LSWSHP}$",
        "wacc_decentral": r"$WACC_{ASHP}$",
        "wacc_central": r"$WACC_{LSWSHP}$",
    }

    air_params = ['c_el_L', 'COP_Air', 'CAPEX_Air', 'hours_Air', 'wacc_decentral']
    water_params = ['c_el_W', 'COP_Water', 'CAPEX_Water', 'hours_Water', 'wacc_central']
    other_params = ['c_pipe', 'q_loss']

    param_colors = {
        'c_el_L': '#1b5e20',
        'COP_Air': '#43a047',
        'CAPEX_Air': '#7cb342',
        'hours_Air': '#a5d6a7',
        'wacc_decentral': "#de5d1c",
        'c_el_W': '#0d47a1',
        'COP_Water': '#1976d2',
        'CAPEX_Water': '#42a5f5',
        'hours_Water': '#90caf9',
        'wacc_central': '#d32f2f',
        'c_pipe': '#424242',
        'q_loss': '#757575',
    }

    param_marker = {
        **{param: 'o' for param in air_params},
        **{param: 's' for param in water_params},
        **{param: 'D' for param in other_params},
    }

    # --- Determine baseline utilization ---
    baseline_values = []
    for param_name, base_val in params_base.items():
        sub = df_sens[df_sens["parameter"] == param_name]
        if sub.empty:
            continue
        idx = (sub["tested_value"] - base_val).abs().idxmin()
        baseline_values.append(sub.loc[idx, "Utilization_GWh"])

    if baseline_values:
        baseline = float(np.median(baseline_values))
    else:
        baseline = df_sens["Utilization_GWh"].median()

    # --- Plot ---
    from matplotlib.lines import Line2D

    all_params = df_sens['parameter'].unique()
    fig, ax = plt.subplots(figsize=(11, 7))

    nominal_wacc_offset = 0.02  # Add 2 percentage points inflation to display nominal WACC on x-axis.

    for param in all_params:
        subset = df_sens[df_sens['parameter'] == param].copy()

        tested_values_for_plot = subset['tested_value'].astype(float)
        if param in {'wacc_decentral', 'wacc_central'}:
            tested_values_for_plot = tested_values_for_plot + nominal_wacc_offset

        # Find baseline tested_value for this parameter
        baseline_idx = (subset['Utilization_GWh'] - baseline).abs().idxmin()
        baseline_tested = tested_values_for_plot.loc[baseline_idx]

        # Normalize: % change from baseline
        if baseline_tested != 0:
            subset['x_norm'] = (tested_values_for_plot - baseline_tested) / abs(baseline_tested) * 100
        else:
            subset['x_norm'] = 0.0
        subset['y_norm'] = (subset['Utilization_GWh'] - baseline) / baseline * 100

        color = param_colors.get(param, '#333333')
        marker = param_marker.get(param, 'o')
        ax.scatter(subset['x_norm'], subset['y_norm'], color=color, marker=marker,
                   s=40, linewidth=0, zorder=3, alpha=0.85)
        ax.plot(subset['x_norm'], subset['y_norm'], color=color, linewidth=2.0,
                alpha=0.8, linestyle='-')

    ax.axhline(y=0, color='black', linewidth=1, linestyle='-')
    ax.axvline(x=0, color='black', linewidth=1, linestyle='-')

    if cfg.plot_language == "en":
        ax.set_xlabel('Parameter Variation from Baseline [%]', fontsize=12)
        ax.set_ylabel('Economic Potential Variation from Baseline [%]', fontsize=12)
    else:
        ax.set_xlabel('Parametervariation vom Basiswert [%]', fontsize=12)
        ax.set_ylabel('Wirtschaftliches Potential – Variation vom Basiswert [%]', fontsize=12)

    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.6, alpha=0.4)
    ax.minorticks_on()
    ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.4, alpha=0.25)

    # --- Grouped legend ---
    legend_handles = []
    legend_labels = []

    air_title = (r'$\bf{Air\text{-}source\ HP}$' if cfg.plot_language == "en"
                 else r'$\bf{Luft\text{-}WP\ (dezentral)}$')
    legend_handles.append(Line2D([0], [0], color='none'))
    legend_labels.append(air_title)
    for param in air_params:
        if param in param_colors:
            legend_handles.append(Line2D([0], [0], color=param_colors[param], marker=param_marker.get(param, 'o'),
                                         markerfacecolor=param_colors[param], linestyle='-', linewidth=2.5,
                                         markersize=5))
            legend_labels.append(math_labels.get(param, param))

    water_title = (r'$\bf{Water\text{-}source\ HP}$' if cfg.plot_language == "en"
                   else r'$\bf{Wasser\text{-}WP\ (zentral)}$')
    legend_handles.append(Line2D([0], [0], color='none'))
    legend_labels.append(water_title)
    for param in water_params:
        if param in param_colors:
            legend_handles.append(Line2D([0], [0], color=param_colors[param], marker=param_marker.get(param, 's'),
                                         markerfacecolor=param_colors[param], linestyle='-', linewidth=2.5,
                                         markersize=5))
            legend_labels.append(math_labels.get(param, param))

    other_title = r'$\bf{Other}$' if cfg.plot_language == "en" else r'$\bf{Sonstige}$'
    legend_handles.append(Line2D([0], [0], color='none'))
    legend_labels.append(other_title)
    for param in other_params:
        if param in param_colors:
            legend_handles.append(Line2D([0], [0], color=param_colors[param], marker=param_marker.get(param, 'D'),
                                         markerfacecolor=param_colors[param], linestyle='-', linewidth=2.5,
                                         markersize=5))
            legend_labels.append(math_labels.get(param, param))
    max_col_len = max(len(air_params), len(water_params), len(other_params))
    for _ in range(max_col_len - len(other_params)):
        legend_handles.append(Line2D([0], [0], color='none'))
        legend_labels.append('')

    ax.legend(legend_handles, legend_labels, loc='upper center', fontsize=10,
              framealpha=0.95, ncol=3, columnspacing=2.0, handlelength=1.5,
              bbox_to_anchor=(0.5, -0.12))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.24)

    Path(cfg.economic_sensitivity_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cfg.economic_sensitivity_plot_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(cfg.economic_sensitivity_plot_path.replace(".pdf", ".svg"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sensitivity plot saved: {cfg.economic_sensitivity_plot_path}")


# ============================================================
# Distance-Band Economic Viability Analysis
# ============================================================

def _distance_band_analysis(cfg, ei_data, heat_data, actual_dist_data, hylak_data):
    """Analyse how much heat demand within distance bands (1 km, 10 km) is economically viable (EI >= threshold).
    
    Uses actual Euclidean distance to nearest lake shore (not effective/hub distance).
    """
    
    mask_valid = (
        np.isfinite(ei_data) & np.isfinite(heat_data) & (heat_data > 0)
        & np.isfinite(actual_dist_data) & (hylak_data > 0)
    )
    
    bands_m = [1000, 10000]  # 1 km, 10 km
    
    results = []
    
    for max_dist_m in bands_m:
        # All pixels within this distance band
        mask_band = mask_valid & (actual_dist_data <= max_dist_m)
        # Pixels within band AND economically viable
        mask_viable = mask_band & (ei_data >= cfg.ei_threshold)
        
        total_demand_MWh = float(np.nansum(heat_data[mask_band]))
        viable_demand_MWh = float(np.nansum(heat_data[mask_viable]))
        
        n_pixels_band = int(mask_band.sum())
        n_pixels_viable = int(mask_viable.sum())
        
        share_demand = (viable_demand_MWh / total_demand_MWh * 100) if total_demand_MWh > 0 else 0.0
        share_pixels = (n_pixels_viable / n_pixels_band * 100) if n_pixels_band > 0 else 0.0
        
        results.append({
            "Distance_band_km": max_dist_m / 1000,
            "Total_demand_GWh": total_demand_MWh / 1000,
            "Viable_demand_GWh": viable_demand_MWh / 1000,
            "Share_demand_pct": share_demand,
            "N_pixels_total": n_pixels_band,
            "N_pixels_viable": n_pixels_viable,
            "Share_pixels_pct": share_pixels,
        })
    
    df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "-" * 65)
    print("  DISTANCE-BAND ECONOMIC VIABILITY ANALYSIS")
    print(f"  (EI threshold: {cfg.ei_threshold})")
    print("-" * 65)
    print(f"  {'Band':>8} | {'Total Demand':>14} | {'Viable (EI≥1)':>14} | {'Share':>8} | {'Pixels viable':>14}")
    print(f"  {'':>8} | {'[GWh/yr]':>14} | {'[GWh/yr]':>14} | {'[%]':>8} | {'[%]':>14}")
    print("  " + "-" * 63)
    for _, row in df.iterrows():
        print(f"  {row['Distance_band_km']:>6.0f} km | "
              f"{row['Total_demand_GWh']:>14,.1f} | "
              f"{row['Viable_demand_GWh']:>14,.1f} | "
              f"{row['Share_demand_pct']:>7.1f}% | "
              f"{row['Share_pixels_pct']:>12.1f}%")
    print("-" * 65)
    
    # Also show TWh for readability
    for _, row in df.iterrows():
        print(f"  Within {row['Distance_band_km']:.0f} km: "
              f"{row['Viable_demand_GWh']/1000:.2f} TWh of {row['Total_demand_GWh']/1000:.2f} TWh "
              f"economically viable ({row['Share_demand_pct']:.1f}%)")
    print("")
    
    # Save to CSV
    out_path = Path(cfg.economic_compare_csv).parent / "distance_band_viability.csv"
    df.to_csv(out_path, index=False)
    print(f"  Distance-band analysis saved: {out_path}")
    
    return df

# ============================================================
# Main Entry Point
# ============================================================

def calculate(cfg):
    """Full economic potential pipeline with optional caching."""

    # 1. Calculate or load EI raster
    ei_cache_exists = Path(cfg.ei_output_tif).exists()

    if cfg.use_cache and ei_cache_exists:
        print(f"  [CACHE] Loading existing EI raster: {cfg.ei_output_tif}")
    else:
        if cfg.use_cache and not ei_cache_exists:
            print(f"  [CACHE] EI raster not found, computing fresh...")
        else:
            print("  Computing EI raster (Annuity method)...")
        _calculate_ei_raster(cfg)

    # 2. Load results and get lake list
    ei_data, hylak_data, dist_data, hub_data, heat_data, actual_dist_data, transform, crs = _load_ei_results(cfg)
    lakes_gdf = _get_lake_ids(cfg, crs)

    # 3. Per-lake heat aggregation (with cache)
    compare_cache_exists = Path(cfg.economic_compare_csv).exists()
    compare_df = None
    compare_recomputed = False

    if cfg.use_cache and compare_cache_exists:
        print(f"  [CACHE] Loading existing economic comparison: {cfg.economic_compare_csv}")
        compare_df = pd.read_csv(cfg.economic_compare_csv)
        if not _has_required_compare_columns(compare_df):
            missing = sorted(REQUIRED_COMPARE_COLUMNS.difference(compare_df.columns))
            print(
                "  [CACHE] Economic comparison cache is stale/incomplete; "
                f"missing columns: {missing}. Recomputing..."
            )
            compare_df = None

    if compare_df is None:
        if cfg.use_cache and not compare_cache_exists:
            print(f"  [CACHE] Economic comparison CSV not found, computing fresh...")
        elif cfg.use_cache:
            print("  [CACHE] Economic comparison cache invalid, computing fresh...")
        else:
            print("  Computing per-lake economic heat demand...")
        heat_df = _compute_heat_per_lake(cfg, ei_data, hylak_data, heat_data, lakes_gdf)
        compare_df = _merge_with_potential(heat_df, cfg)
        compare_recomputed = True

    # 4. Base case summary output
    _print_base_case_summary(cfg, compare_df, ei_data, hylak_data, heat_data)

    # 4b. Distance-band economic viability analysis
    print("  Analysing economic viability by distance to lake...")
    dist_band_df = _distance_band_analysis(cfg, ei_data, heat_data, actual_dist_data, hylak_data)

    # 5. Utilization selection (with cache)
    utilized_cache_path = Path(cfg.utilized_mask_cache_path)
    utilized_cache_exists = utilized_cache_path.exists()

    # Reuse only if the comparison itself came from a valid cache
    if cfg.use_cache and utilized_cache_exists and not compare_recomputed:
        print(f"  [CACHE] Loading existing utilization mask: {utilized_cache_path}")
        utilized_mask = np.load(utilized_cache_path)
    else:
        if cfg.use_cache and not utilized_cache_exists:
            print(f"  [CACHE] Utilization mask not found, computing fresh...")
        else:
            print("  Selecting utilized hubs...")
        utilized_mask = _select_utilized_hubs(cfg, ei_data, hylak_data, hub_data, heat_data)
        utilized_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(utilized_cache_path, utilized_mask)
        print(f"  Utilization mask cached: {utilized_cache_path}")

    # 6. Scatter plot
    print("  Plotting EI scatter...")
    plot_scatter(cfg, ei_data, dist_data, heat_data, utilized_mask, actual_dist_data)

    # 8. Sensitivity analysis (with cache)
    if cfg.run_economic_sensitivity:
        sens_csv_path = Path(cfg.economic_sensitivity_csv)
        sens_cache_exists = sens_csv_path.exists()

        if cfg.use_cache and sens_cache_exists and not compare_recomputed:
            print(f"  [CACHE] Loading existing sensitivity results: {sens_csv_path}")
            df_sens = pd.read_csv(sens_csv_path)
            params_base = {
                "c_el_L": cfg.c_el_L, "c_el_W": cfg.c_el_W,
                "c_pipe": cfg.c_pipe, "COP_Water": cfg.COP_Water, "COP_Air": cfg.COP_Air,
                "q_loss": cfg.q_loss, "CAPEX_Air": cfg.CAPEX_Air, "CAPEX_Water": cfg.CAPEX_Water,
                "hours_Air": cfg.hours_Air, "hours_Water": cfg.hours_Water,
                "wacc_decentral": cfg.wacc_decentral,
                "wacc_central": cfg.wacc_central,
            }
        else:
            if cfg.use_cache and not sens_cache_exists:
                print(f"  [CACHE] Sensitivity CSV not found, computing fresh...")
            else:
                print("  Running sensitivity analysis...")
            df_sens, params_base = sensitivity_analysis(cfg, ei_data, hylak_data, dist_data, heat_data)
            sens_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_sens.to_csv(sens_csv_path, index=False)
            print(f"  Sensitivity results cached: {sens_csv_path}")

        plot_sensitivity(cfg, df_sens, params_base)

    # 9. Minimum project size (final step, all results above are without threshold)
    compare_df = apply_min_project_size(cfg, compare_df)
    plot_project_size_distribution(cfg, compare_df)

    # 10. Map plot (final economic potential incl. minimum project size)
    print("  Plotting economic potential map...")
    plot_map(cfg, compare_df)

    return compare_df