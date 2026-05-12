# case_study.py
"""
Case Study: Recalculate EI for ALL demand pixels within 5 km
of the Schwerin Lake (Hylak_id=13463), using this lake as the
sole heat source.

Cells with existing district heating (>=50% share from Zensus 2022)
are treated as having zero additional pipe costs.
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
import warnings

from config import Config

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

cfg = Config()

LAKE_ID = 13463               # Schwerin Lake
BUFFER_M = 5000               # 5 km
OUTPUT_DIR = Path(cfg.results_dir) / "CaseStudy_Schwerin"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Zensus data path
ZENSUS_PATH = Path("Data/Raw/Zensus2022_Gebaeude_nach_Energietraeger_der_Heizung_100m-Gitter.csv")
DH_THRESHOLD = 0.5


# ============================================================
# HELPERS
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


# ============================================================
# DISTRICT HEATING CLASSIFICATION (ZENSUS)
# ============================================================

def load_district_heating_mask(bounds, local_transform, shape):
    """
    Load Zensus 2022 data, classify cells with >=50% district heating,
    and rasterize onto the local grid.
    Returns a boolean array: True = existing district heating.
    """
    print("  Loading Zensus 2022 district heating data...")

    if not ZENSUS_PATH.exists():
        print(f"  WARNING: Zensus file not found: {ZENSUS_PATH}")
        print("  → Proceeding without district heating adjustment.")
        return np.zeros(shape, dtype=bool)

    df = pd.read_csv(ZENSUS_PATH, encoding="utf-8", sep=";")
    energy_cols = ["Insgesamt_Energietraeger", "Fernwaerme"]
    df[energy_cols] = df[energy_cols].replace("–", 0).astype(int)
    df = df[df["Insgesamt_Energietraeger"] > 0].copy()
    df["dh_share"] = df["Fernwaerme"] / df["Insgesamt_Energietraeger"]
    df["has_dh"] = df["dh_share"] >= DH_THRESHOLD

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["x_mp_100m"], df["y_mp_100m"]),
        crs="EPSG:3035",
    )
    if cfg.target_crs != "EPSG:3035":
        gdf = gdf.to_crs(cfg.target_crs)

    xmin, ymin, xmax, ymax = bounds
    gdf = gdf.cx[xmin:xmax, ymin:ymax]
    dh_cells = gdf[gdf["has_dh"]].copy()

    if dh_cells.empty:
        print("  No cells with existing district heating in this area.")
        return np.zeros(shape, dtype=bool)

    dh_cells["geometry"] = dh_cells.geometry.buffer(50, cap_style=3)
    dh_shapes = [(geom, 1) for geom in dh_cells.geometry if geom is not None]
    dh_raster = features.rasterize(
        shapes=dh_shapes, out_shape=shape, transform=local_transform,
        fill=0, all_touched=True, dtype="uint8",
    )
    dh_mask = dh_raster > 0

    n_dh_cells = len(dh_cells)
    print(f"  Zensus cells in area: {len(gdf)}")
    print(f"  Cells with district heating (≥{DH_THRESHOLD*100:.0f}%): {n_dh_cells}")
    print(f"  Raster pixels marked as existing DH: {int(dh_mask.sum())}")

    return dh_mask


# ============================================================
# EI RECALCULATION
# ============================================================

def recalculate_ei_for_lake(cfg, lake_id, buffer_m):
    """Recalculate EI using ONLY this lake as heat source."""

    print(f"\n{'='*60}")
    print(f"  RECALCULATING EI for Lake {lake_id} (buffer={buffer_m}m)")
    print(f"  District heating cells → no pipe costs")
    print(f"{'='*60}")

    lakes_gdf = gpd.read_file(cfg.lakes_path).to_crs(cfg.target_crs)
    lakes_gdf["Hylak_id"] = lakes_gdf["Hylak_id"].astype(int)
    lake_row = lakes_gdf[lakes_gdf["Hylak_id"] == lake_id]
    if lake_row.empty:
        raise ValueError(f"Lake {lake_id} not found!")
    lake_geom = lake_row.iloc[0].geometry
    lake_vol_km3 = lake_row.iloc[0]["Vol_total"]

    print(f"  Lake area: {lake_geom.area / 1e6:.2f} km²")
    print(f"  Lake volume: {lake_vol_km3:.4f} km³")

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
    print(f"\n  Pixels with existing DH (pipe cost → 0): {n_dh_applied}")

    # --- EI Calculation ---
    print("  Calculating EI...")

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
    disc_Water = _discounted_op_cost(cfg.c_el_W, E_h_W, cfg.COP_Water,
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

    # --- Technical potential ---
    seconds_per_year = 365 * 24 * 3600
    flow_m3s = lake_vol_km3 * 1e6 / seconds_per_year
    withdrawal_m3s = flow_m3s * (cfg.deltaT_technical_extractable / cfg.deltaT_hex)
    capacity_W = withdrawal_m3s * cfg.density_water * cfg.cp_water * cfg.deltaT_hex
    tech_potential_GWh = capacity_W * cfg.hours_per_year / 1e9

    # --- Statistics ---
    mask_viable = mask_calc & (EI_local >= cfg.ei_threshold)
    heat_viable_GWh = float(np.nansum(heat_clip[mask_viable])) / 1000
    heat_total_GWh = float(np.nansum(heat_clip[mask_calc])) / 1000
    covered_GWh = min(heat_viable_GWh, tech_potential_GWh)
    n_viable = int(mask_viable.sum())
    n_total = int(mask_calc.sum())

    mask_viable_dh = mask_viable & dh_mask
    mask_viable_no_dh = mask_viable & ~dh_mask
    heat_viable_dh_GWh = float(np.nansum(heat_clip[mask_viable_dh])) / 1000
    heat_viable_no_dh_GWh = float(np.nansum(heat_clip[mask_viable_no_dh])) / 1000

    print(f"\n  --- Results ---")
    print(f"  Heat within {buffer_m/1000:.0f} km:      {heat_total_GWh:.1f} GWh/yr")
    print(f"  Heat with EI ≥ 1:           {heat_viable_GWh:.1f} GWh/yr")
    print(f"    ├─ via existing DH:       {heat_viable_dh_GWh:.1f} GWh/yr")
    print(f"    └─ new pipe required:     {heat_viable_no_dh_GWh:.1f} GWh/yr")
    print(f"  Technical potential:         {tech_potential_GWh:.2f} GWh/yr")
    print(f"  Economic potential:          {covered_GWh:.2f} GWh/yr")
    print(f"  Viable pixels: {n_viable}/{n_total} ({n_viable/n_total*100:.1f}%)")

    return {
        "EI": EI_local,
        "heat": heat_clip,
        "mask_calc": mask_calc,
        "mask_viable": mask_viable,
        "lake_interior": lake_interior,
        "dh_mask": dh_mask,
        "transform": local_tf,
        "lake_geom": lake_geom,
        "bounds": bounds,
        "tech_potential_GWh": tech_potential_GWh,
        "stats": {
            "Heat_viable_GWh": heat_viable_GWh,
            "Heat_viable_existing_DH_GWh": heat_viable_dh_GWh,
            "Heat_viable_new_pipe_GWh": heat_viable_no_dh_GWh,
            "Technical_potential_GWh": tech_potential_GWh,
            "Economic_potential_GWh": covered_GWh,
            "Pct_viable": n_viable / n_total * 100 if n_total > 0 else 0,
            "N_pixels_existing_DH": n_dh_applied,
        },
    }


# ============================================================
# PLOT
# ============================================================

def plot_viable_area_with_context(results):
    """Viable area with solid colors – no ambiguous colormap gradients."""
    EI = results["EI"]
    mask_viable = results["mask_viable"]
    mask_calc = results["mask_calc"]
    heat = results["heat"]
    dh_mask = results["dh_mask"]
    tf = results["transform"]
    lake_geom = results["lake_geom"]
    bounds = results["bounds"]
    extent = _extent_from_transform(tf, EI.shape)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Background: heat demand (grey)
    h_disp = heat.copy()
    h_disp[h_disp <= 0] = np.nan
    ax.imshow(h_disp, extent=extent, origin="upper", cmap="Greys",
              alpha=0.3, norm=LogNorm(vmin=0.1, vmax=max(1, np.nanmax(h_disp))),
              interpolation="nearest")

    # Category raster with solid colors
    # 0 = transparent (background)
    # 1 = viable, new pipe needed
    # 2 = viable, existing DH (no pipe cost)
    # 3 = not viable (EI < 1)
    cat = np.zeros(EI.shape, dtype="uint8")
    not_viable = mask_calc & ~mask_viable & np.isfinite(EI)
    cat[not_viable] = 3
    cat[mask_viable & ~dh_mask] = 1
    cat[mask_viable & dh_mask] = 2

    # Map categories to RGBA
    rgba = np.zeros((*EI.shape, 4), dtype="float32")
    # Cat 1: green (viable, new pipe)
    rgba[cat == 1] = [0.17, 0.63, 0.17, 0.7]   # #2ca02c
    # Cat 2: purple (viable, existing DH)
    rgba[cat == 2] = [0.58, 0.40, 0.74, 0.7]   # #9467bd
    # Cat 3: red (not viable)
    rgba[cat == 3] = [1.0, 0.6, 0.6, 0.25]     # light red

    ax.imshow(rgba, extent=extent, origin="upper", interpolation="nearest")

    # This lake
    lake_gdf = gpd.GeoDataFrame(geometry=[lake_geom], crs=cfg.target_crs)
    lake_gdf.plot(ax=ax, color="#4292c6", alpha=0.7, zorder=5)
    lake_gdf.boundary.plot(ax=ax, color="black", linewidth=2.5, zorder=6)

    # Other lakes in area
    lakes_gdf = gpd.read_file(cfg.lakes_path).to_crs(cfg.target_crs)
    lakes_gdf["Hylak_id"] = lakes_gdf["Hylak_id"].astype(int)
    bbox_poly = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=cfg.target_crs)
    other_lakes = gpd.sjoin(lakes_gdf[lakes_gdf["Hylak_id"] != LAKE_ID],
                            bbox_poly, how="inner", predicate="intersects")
    if not other_lakes.empty:
        other_lakes.plot(ax=ax, color="#9ecae1", alpha=0.5, edgecolor="gray",
                         linewidth=0.5, zorder=3)

    # Stats box
    s = results["stats"]
    info = (
        f"Economic Potential: {s['Economic_potential_GWh']:.2f} GWh/yr\n"
        f"Heat (EI≥1): {s['Heat_viable_GWh']:.1f} GWh/yr\n"
        f"  ├─ existing DH: {s['Heat_viable_existing_DH_GWh']:.1f} GWh/yr\n"
        f"  └─ new pipe: {s['Heat_viable_new_pipe_GWh']:.1f} GWh/yr\n"
        f"Tech. Potential: {s['Technical_potential_GWh']:.2f} GWh/yr\n"
        f"Viable: {s['Pct_viable']:.1f}% of pixels"
    )
    ax.text(0.02, 0.98, info, transform=ax.transAxes, va="top", ha="left",
            fontsize=10, fontfamily="monospace",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9))

    # Legend – solid colors, no ambiguity
    ax.legend(handles=[
        mpatches.Patch(color="#4292c6", label="Schwerin Lake"),
        mpatches.Patch(color="#9ecae1", label="Other lakes"),
        mpatches.Patch(color="#2ca02c", alpha=0.7, label="Viable – new pipe (EI ≥ 1)"),
        mpatches.Patch(color="#9467bd", alpha=0.7, label="Viable – existing DH (no pipe cost)"),
        mpatches.Patch(color="#ff9999", alpha=0.5, label="Not viable (EI < 1)"),
        mpatches.Patch(facecolor="none", edgecolor="gray", label="Heat demand (background)"),
    ], loc="lower right", fontsize=9, framealpha=0.95)

    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    ax.set_title(f"Viable Supply Area – Lake {LAKE_ID} (Schwerin)\n"
                 f"Including existing district heating networks", fontsize=13)
    ax.set_aspect("equal")
    plt.tight_layout()

    out = OUTPUT_DIR / "Viable_Area_Context.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# MAIN
# ============================================================

def main():
    results = recalculate_ei_for_lake(cfg, LAKE_ID, BUFFER_M)

    stats_df = pd.DataFrame([results["stats"]])
    stats_path = OUTPUT_DIR / "lake_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\n  Statistics saved: {stats_path}")

    print("\n[PLOT]")
    plot_viable_area_with_context(results)

    print(f"\n  Done → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
