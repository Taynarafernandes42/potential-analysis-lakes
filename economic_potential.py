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
    """Rasterize lake boundaries → label array."""
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


# ============================================================
# Core EI Calculation
# ============================================================

def _calculate_ei_raster(cfg):
    """Compute Economic Index raster. Returns arrays + metadata."""

    def _discounted_op_cost(c_el, E_h, COP, infl, lt):
        total = 0.0
        for year in range(1, int(lt) + 1):
            total += (c_el * E_h / COP) / ((1 + infl) ** year)
        return total

    # Load heat raster
    with rasterio.open(cfg.heat_path) as src:
        heat_profile = src.profile.copy()
        heat_data = src.read(1)
        transform = src.transform
        crs_raster = src.crs
        nodata_heat = src.nodata

    # Load lakes
    lakes = gpd.read_file(cfg.lakes_path).to_crs(crs_raster)

    # Rasterize lake boundaries + EDT
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

    E_h_kWh = heat_data.astype("float64") * 1000.0
    d_pixel = np.maximum(dist_to_shore_m, cfg.min_pipe_length_m)

    # --- Distance computation ---
    if cfg.distance_mode == "pixel":
        D = d_pixel.astype("float32")
        HI = nearest_lake_id_map.astype("int32")
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

                # Weighted hub centroid
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

            # Fill non-valid pixels with pixel distance
            D[~valid_mask] = d_pixel[~valid_mask].astype("float32")
            HI[~valid_mask] = nearest_lake_id_map[~valid_mask].astype("int32")

    # --- Apply max distance limit ---
    D = np.where(
        np.isfinite(D) & (D <= cfg.ei_max_dist_m),
        np.maximum(D, cfg.min_pipe_length_m), np.nan
    ).astype("float32")
    mask_valid = np.isfinite(D) & (HI > 0)

    # --- EI Calculation ---
    E_h_L = E_h_kWh[mask_valid]
    E_h_W = E_h_kWh[mask_valid] * (1 + cfg.q_loss)

    disc_Air = _discounted_op_cost(cfg.c_el_L, E_h_L, cfg.COP_Air, cfg.inflation_rate, cfg.lifetime)
    disc_Water = _discounted_op_cost(cfg.c_el_W, E_h_W, cfg.COP_Water, cfg.inflation_rate, cfg.lifetime)

    numerator = (
        (cfg.CAPEX_Air * (E_h_L / cfg.hours_Air)) + disc_Air
        - (cfg.CAPEX_Water * (E_h_W / cfg.hours_Water)) - disc_Water
    )
    denominator = cfg.c_pipe * D[mask_valid].astype("float64")
    denominator = np.where(denominator == 0, np.nan, denominator)
    EI[mask_valid] = numerator / denominator

    # Clean up
    if nodata_heat is not None:
        invalid = (heat_data == nodata_heat) | ~np.isfinite(heat_data)
    else:
        invalid = ~np.isfinite(heat_data)
    EI[invalid] = np.nan
    D[invalid] = np.nan
    HI[~np.isfinite(EI)] = 0

    # --- Save to GeoTIFF ---
    out_profile = heat_profile.copy()
    out_profile.update({"count": 4, "dtype": "float32", "nodata": np.nan})
    Path(cfg.ei_output_tif).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(cfg.ei_output_tif, "w", **out_profile) as dst:
        dst.write(EI.astype("float32"), 1)
        dst.write(HI.astype("float32"), 2)
        dst.write(D.astype("float32"), 3)
        dst.write(HUB.astype("float32"), 4)

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

    return ei_data, hylak_data, dist_data, hub_data, heat_data, transform, crs


def _get_lake_ids(cfg, crs):
    """Load lake GeoDataFrame, apply NUTS filter, return filtered GDF."""
    lakes = gpd.read_file(cfg.lakes_path)
    if lakes.crs != crs:
        lakes = lakes.to_crs(crs)

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
    """Merge heat demand (EI>=1) with technical potential → economic potential."""
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
    df["Covered_Heat_GWh"] = np.minimum(df["Heat_EI_ge1_GWh"], df["technical_extractable_Thermal_Potential_GWh"])

    total_GWh = df["Covered_Heat_GWh"].sum()
    n_no_demand = int((df["Heat_EI_ge1_GWh"] == 0).sum())
    df_with_demand = df[df["Heat_EI_ge1_GWh"] > 0]
    n_pot_gt = int((df_with_demand["technical_extractable_Thermal_Potential_GWh"] > df_with_demand["Heat_EI_ge1_GWh"]).sum())
    n_pot_lt = int((df_with_demand["technical_extractable_Thermal_Potential_GWh"] < df_with_demand["Heat_EI_ge1_GWh"]).sum())

    print(f"  Lakes with no economic potential: {n_no_demand}")
    print(f"  Lakes where technical > economic demand: {n_pot_gt}")
    print(f"  Lakes where technical < economic demand: {n_pot_lt}")
    print(f"  Total Economic Potential: {total_GWh / 1000:.2f} TWh/year")

    Path(cfg.economic_compare_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.economic_compare_csv, index=False)
    return df


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
    pot_df["pot_MWh"] = pot_df["potential_Wh"] / 1e6
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

        # Single pixels (no hub)
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
# Sensitivity Analysis
# ============================================================

def sensitivity_analysis(cfg, ei_data, hylak_data, dist_data, heat_data):
    """Vary EI parameters and compute resulting economic potential."""

    def _discounted(c_el, E_h, COP):
        total = 0.0
        for yr in range(1, cfg.lifetime + 1):
            total += (c_el * E_h / COP) / ((1 + cfg.inflation_rate) ** yr)
        return total

    mask_base = np.isfinite(dist_data) & (hylak_data > 0) & np.isfinite(heat_data) & (heat_data > 0)
    E_h_base = heat_data[mask_base].astype("float64") * 1000.0
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
    }

    results = []
    for param_name, (low, high) in cfg.sensitivity_param_ranges.items():
        values = np.linspace(low, high, cfg.sensitivity_n_steps)
        # Ensure base value is included
        if param_name in params_base:
            idx_close = np.argmin(np.abs(values - params_base[param_name]))
            values[idx_close] = params_base[param_name]

        for val in values:
            p = params_base.copy()
            p[param_name] = float(val)

            E_h_W = E_h_base * (1 + p["q_loss"])
            disc_A = _discounted(p["c_el_L"], E_h_base, p["COP_Air"])
            disc_W = _discounted(p["c_el_W"], E_h_W, p["COP_Water"])

            num = ((p["CAPEX_Air"] * E_h_base / p["hours_Air"]) + disc_A
                   - (p["CAPEX_Water"] * E_h_W / p["hours_Water"]) - disc_W)
            den = np.where(p["c_pipe"] * D_base == 0, np.nan, p["c_pipe"] * D_base)
            EI_var = num / den
            mask_ei = EI_var >= cfg.ei_threshold

            if np.any(mask_ei):
                uids, inv = np.unique(HI_base[mask_ei], return_inverse=True)
                sums = np.bincount(inv, weights=heat_MWh_base[mask_ei]).astype(float)
                covered = sum(min(s / 1000.0, pot_map.get(int(uid), 0.0))
                              for uid, s in zip(uids, sums))
            else:
                covered = 0.0

            results.append({"parameter": param_name, "tested_value": float(val),
                            "Utilization_GWh": covered})

    df = pd.DataFrame(results)
    df["Utilization_TWh"] = df["Utilization_GWh"] / 1000.0
    return df, params_base


# ============================================================
# Plots
# ============================================================

def plot_scatter(cfg, ei_data, dist_data, heat_data, utilized_mask=None):
    """EI scatter plot: distance vs heat demand, colored by EI."""
    base_font = 11
    factor = 1 / 0.65
    plt.rcParams.update({
        "font.size": base_font * factor,
        "axes.labelsize": base_font * factor,
        "legend.fontsize": base_font * factor,
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

    mask_plot = np.isfinite(ei_data) & np.isfinite(dist_data) & np.isfinite(heat_data) & (heat_data > 0)
    if utilized_mask is not None:
        mask_plot &= utilized_mask
    mask_plot &= (ei_data > 0)

    ei_vals = np.clip(ei_data[mask_plot], 0, 5)
    dist_vals = dist_data[mask_plot]
    heat_vals = heat_data[mask_plot]

    mask_gray = ei_vals < 1
    mask_color = ei_vals >= 1

    fig, ax = plt.subplots(figsize=(12, 8))

    if np.any(mask_gray):
        ax.scatter(dist_vals[mask_gray], heat_vals[mask_gray],
                   c="lightgray", s=5, alpha=0.7, linewidths=0, label="EI < 1")
    if np.any(mask_color):
        sc = ax.scatter(dist_vals[mask_color], heat_vals[mask_color],
                        c=ei_vals[mask_color], cmap="plasma_r",
                        s=5, alpha=1, linewidths=0, norm=LogNorm(vmin=1, vmax=5))
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("EI (clipped at 5)")
        cbar.set_ticks([1, 2, 3, 4, 5])
        cbar.ax.yaxis.set_major_formatter(plt.ScalarFormatter())

    ax.set_xlim(left=0, right=cfg.ei_max_dist_m)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Distance to Lake (m)")
    ax.set_ylabel("Heat Demand [MWh/yr]")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(markerscale=3, loc="upper right")

    plt.tight_layout()
    Path(cfg.economic_scatter_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cfg.economic_scatter_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_map(cfg, compare_df):
    """Map of economic potential per lake, colored districts."""
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

    lakes_gdf = gpd.read_file(cfg.lakes_path).to_crs(target_crs)
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

    # Merge compare data with lakes
    df = compare_df.copy()
    df["Hylak_id"] = df["Hylak_id"].astype(int)
    lakes_gdf["Hylak_id"] = lakes_gdf["Hylak_id"].astype(int)
    lakes_with_data = lakes_gdf.merge(
        df[["Hylak_id", "Covered_Heat_GWh"]], on="Hylak_id", how="left"
    )
    lakes_with_data["Covered_Heat_GWh"] = lakes_with_data["Covered_Heat_GWh"].fillna(0)
    lakes_with_data["centroid"] = lakes_with_data.geometry.centroid

    if cfg.nuts1_filter:
        lakes_with_data = gpd.sjoin(
            lakes_with_data, filtered_nuts[["NUTS_CODE", "geometry"]],
            how="inner", predicate="intersects"
        )
        if "index_right" in lakes_with_data.columns:
            lakes_with_data = lakes_with_data.drop(columns="index_right")

    # District highlighting
    relevant = lakes_with_data[lakes_with_data["Covered_Heat_GWh"] > 0.0001]
    name_col = next((c for c in ["NUTS_NAME", "GEN", "NAME_3", "NAME"]
                     if c in states_gdf.columns), None)

    if name_col and not relevant.empty:
        if "index_right" in relevant.columns:
            relevant = relevant.drop(columns="index_right")
        joined = gpd.sjoin(relevant, states_gdf[[name_col, "geometry"]], how="inner", predicate="intersects")
        agg = joined.groupby(name_col)["Covered_Heat_GWh"].sum().reset_index()
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
        val = row["Covered_Heat_GWh"]
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

    # Extent & reference circles
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
    """Sensitivity plot: economic potential vs. parameter variations."""
    base_font = 11
    factor = 1 / 0.5
    plt.rcParams.update({
        "font.size": base_font * factor * 0.9,
        "axes.labelsize": base_font * factor,
        "legend.fontsize": base_font * factor * 0.9,
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

    plot_params = cfg.sensitivity_plot_params
    param_labels = {
        "CAPEX_Air": r"$C_{\mathrm{inv,Air}}$ (€/kW)",
        "CAPEX_Water": r"$C_{\mathrm{inv,Water}}$ (€/kW)",
        "c_el_L": r"$C_{\mathrm{el,Air}}$ (€/kWh)",
        "c_el_W": r"$C_{\mathrm{el,Water}}$ (€/kWh)",
        "c_pipe": r"$C_{\mathrm{pipe}}$ (€/m)",
        "COP_Air": r"$\mathrm{COP}_{\mathrm{Air}}$",
        "COP_Water": r"$\mathrm{COP}_{\mathrm{Water}}$",
        "q_loss": r"$q_{\mathrm{loss}}$ (%)",
        "hours_Air": r"$h_{\mathrm{Air}}$ (h/a)",
        "hours_Water": r"$h_{\mathrm{Water}}$ (h/a)",
    }

    # Find base values in results
    df_base_vals = []
    for p in plot_params:
        sub = df_sens[df_sens["parameter"] == p]
        if sub.empty:
            df_base_vals.append(np.nan)
            continue
        idx = (sub["tested_value"] - params_base[p]).abs().idxmin()
        df_base_vals.append(sub.loc[idx, "Utilization_TWh"])

    # Assign scenario numbers
    df = df_sens.copy()
    df["scenario"] = df.groupby("parameter").cumcount() + 1

    param_order = {p: i for i, p in enumerate(plot_params)}
    x_pos = np.arange(len(plot_params))

    palette = ["#0F5C4B", "#136C57", "#187964", "#1E876F", "#24957B",
               "#2AA288", "#30B194", "#36BF9F", "#3CCDAA", "#42DBB5"]

    fig, ax = plt.subplots(figsize=(14, 6))
    scenarios = sorted(df["scenario"].unique())

    for i, scen in enumerate(scenarios):
        sub = df[(df["scenario"] == scen) & (df["parameter"].isin(plot_params))]
        sub = sub.sort_values("parameter", key=lambda s: s.map(param_order))
        ax.plot(sub["parameter"].map(param_order).values, sub["Utilization_TWh"],
                color=palette[i % len(palette)], lw=1.5, marker="o", alpha=1)

    # Base line
    ax.plot(x_pos, df_base_vals, color="#f58220", lw=3, marker="o",
            label="Base values" if cfg.plot_language == "en" else "Basiswerte")

    ax.set_ylabel("Economic potential [TWh/a]")
    ax.set_xlabel("Parameter")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([param_labels.get(p, p) for p in plot_params], rotation=45, ha="right")
    ax.grid(True, linestyle="--", alpha=0.4)

    y_min, y_max = df["Utilization_TWh"].min(), df["Utilization_TWh"].max()
    if np.isfinite(y_min) and np.isfinite(y_max):
        span = max(1e-9, y_max - y_min)
        ax.set_ylim(max(0, y_min - 2 * span), y_max + 1.5 * span)

    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    Path(cfg.economic_sensitivity_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cfg.economic_sensitivity_plot_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(cfg.economic_sensitivity_plot_path.replace(".pdf", ".svg"), bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main Entry Point
# ============================================================

def calculate(cfg):
    """Full economic potential pipeline."""

    # 1. Calculate EI raster
    print("  Computing EI raster...")
    EI, HI, D, heat_data, transform, crs = _calculate_ei_raster(cfg)

    # 2. Load results & get lake list
    ei_data, hylak_data, dist_data, hub_data, heat_data, transform, crs = _load_ei_results(cfg)
    lakes_gdf = _get_lake_ids(cfg, crs)

    # 3. Per-lake heat aggregation
    print("  Computing per-lake economic heat demand...")
    heat_df = _compute_heat_per_lake(cfg, ei_data, hylak_data, heat_data, lakes_gdf)

    # 4. Merge with technical potential → economic potential
    compare_df = _merge_with_potential(heat_df, cfg)

    # 5. Utilization selection
    print("  Selecting utilized hubs...")
    utilized_mask = _select_utilized_hubs(cfg, ei_data, hylak_data, hub_data, heat_data)

    # 6. Scatter plot
    print("  Plotting EI scatter...")
    plot_scatter(cfg, ei_data, dist_data, heat_data, utilized_mask)

    # 7. Map plot
    print("  Plotting economic potential map...")
    plot_map(cfg, compare_df)

    # 8. Sensitivity analysis
    if cfg.run_economic_sensitivity:
        print("  Running sensitivity analysis...")
        df_sens, params_base = sensitivity_analysis(cfg, ei_data, hylak_data, dist_data, heat_data)
        plot_sensitivity(cfg, df_sens, params_base)

    return compare_df