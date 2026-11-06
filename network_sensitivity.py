"""Sensitivity of the economic potential to the network approximation (Reviewer 2, Comment 2.10).

Varies the hub size (hub_grid_m) and the maximum effective pipeline distance (ei_max_dist_m).
All cost parameters stay at the base case. Usage (after main.py):

    python network_sensitivity.py

- Variants with the base hub size and a smaller maximum distance are computed from the
  existing base EI raster: cells whose trunk pipeline to the lake (band 5, d_hub,lake)
  exceeds the limit are dropped.
- All other variants recompute the EI raster with economic_potential._calculate_ei_raster.
  Each variant is written to its own GeoTIFF in Data/Processed/network_sensitivity/,
  so the base raster (EI_results.tif) is not overwritten. Existing variant rasters are reused.
- The economic potential per variant is computed with the same function as the scenario
  analysis (EI >= threshold -> capped by delivered-heat potential -> minimum project size).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import Config
from economic_potential import _calculate_ei_raster
from scenario_analysis import _base_params, _scenario_potential


# (label, group, hub size [m], max. distance [m])
VARIANTS = [
    ("Base case (1 km hubs, 10 km)", "Base", 1000, 10000),
    ("Hub size 0.5 km", "Hub size", 500, 10000),
    ("Hub size 2 km", "Hub size", 2000, 10000),
    ("Max. distance 5 km", "Max. distance", 1000, 5000),
    ("Max. distance 15 km", "Max. distance", 1000, 15000),
]

GROUP_COLORS = {"Base": "#555555", "Hub size": "#f58220", "Max. distance": "#1976d2"}
OUT_DIR_NAME = "network_sensitivity"


class VariantConfig(Config):
    """Config with its own EI raster path, so the base raster is not overwritten."""

    def __init__(self, hub_m, max_dist_m):
        super().__init__()
        self.hub_grid_m = hub_m
        self.ei_max_dist_m = max_dist_m
        self._variant_tif = str(Path(self.processed_dir) / OUT_DIR_NAME
                                / f"EI_results_hub{hub_m}_dist{max_dist_m}.tif")

    @property
    def ei_output_tif(self):
        return self._variant_tif


def _read_raster(cfg, tif_path, with_trunk=False):
    """Read lake assignment and effective distance (and optionally the trunk length) from an
    EI raster, plus heat demand."""
    with rasterio.open(tif_path) as src:
        hylak = src.read(2).astype("int32")
        dist = src.read(3).astype("float32")
        if src.nodata is not None:
            dist[dist == src.nodata] = np.nan
        if with_trunk:
            if src.count < 5:
                raise RuntimeError(f"{tif_path} has no trunk-length band. Delete it and rerun main.py.")
            trunk = src.read(5).astype("float32")
    with rasterio.open(cfg.heat_path) as src:
        heat = src.read(1).astype("float32")
        if src.nodata is not None:
            heat[heat == src.nodata] = np.nan
    if with_trunk:
        return hylak, dist, heat, trunk
    return hylak, dist, heat


def _get_variant_data(base_cfg, hub_m, max_dist_m):
    """Return (hylak, dist, heat) for one variant, recomputing the raster only if needed."""
    if hub_m == base_cfg.hub_grid_m and max_dist_m <= base_cfg.ei_max_dist_m:
        print(f"  Using base EI raster (limit {max_dist_m / 1000:.1f} km)")
        hylak, dist, heat, trunk = _read_raster(base_cfg, base_cfg.ei_output_tif, with_trunk=True)
        dist = np.where(trunk <= max_dist_m, dist, np.nan).astype("float32")
        return hylak, dist, heat

    vcfg = VariantConfig(hub_m, max_dist_m)
    if Path(vcfg.ei_output_tif).exists():
        print(f"  [CACHE] Using {vcfg.ei_output_tif}")
    else:
        print(f"  Computing EI raster: hub {hub_m} m, limit {max_dist_m} m (this takes a while)...")
        _calculate_ei_raster(vcfg)
    return _read_raster(vcfg, vcfg.ei_output_tif)


def run_network_sensitivity(cfg):
    pot_df = pd.read_csv(cfg.technical_extractable_csv)
    pot_lake_GWh = dict(zip(pot_df["Hylak_id"].astype(int), pot_df["potential_Wh"] / 1e9))
    p = _base_params(cfg)

    rows = []
    for label, group, hub_m, max_dist_m in VARIANTS:
        print(f"\n--- {label} ---")
        hylak, dist, heat = _get_variant_data(cfg, hub_m, max_dist_m)
        mask = np.isfinite(dist) & (hylak > 0) & np.isfinite(heat) & (heat > 0)
        df = _scenario_potential(
            cfg, p,
            heat[mask].astype("float64") * 1000.0,      # MWh -> kWh
            dist[mask].astype("float64"),
            hylak[mask].astype("int64"),
            pot_lake_GWh,
        )
        above = df["Above_Min_Size"]
        rows.append({
            "label": label,
            "group": group,
            "hub_grid_m": hub_m,
            "max_dist_m": max_dist_m,
            "Econ_before_min_TWh": df["Covered_Heat_GWh"].sum() / 1000.0,
            "Econ_TWh": df.loc[above, "Covered_Heat_GWh"].sum() / 1000.0,
            "Lakes_with_potential": int(len(df)),
            "Lakes_above_min": int(above.sum()),
        })
        del hylak, dist, heat

    out = pd.DataFrame(rows)
    base_val = out.loc[out["group"] == "Base", "Econ_TWh"].iloc[0]
    out["Change_vs_base_pct"] = (out["Econ_TWh"] / base_val - 1.0) * 100 if base_val > 0 else np.nan

    print("\n" + "=" * 78)
    print("  NETWORK SENSITIVITY  (economic potential incl. minimum project size)")
    print("=" * 78)
    print(f"  {'Variant':<32}{'before min':>11}{'TWh/a':>9}{'vs. base':>11}{'lakes':>8}")
    for _, r in out.iterrows():
        print(f"  {r['label']:<32}{r['Econ_before_min_TWh']:>11.2f}{r['Econ_TWh']:>9.2f}"
              f"{r['Change_vs_base_pct']:>10.2f}%{r['Lakes_above_min']:>8d}")
    print("=" * 78 + "\n")
    return out


def plot_network_sensitivity(cfg, df, out_path):
    """Economic potential (left) and number of lakes (right) per variant."""
    plt.rcParams.update({"font.size": 14, "font.family": "serif", "mathtext.fontset": "cm"})

    d = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(d))
    colors = [GROUP_COLORS.get(g, "#999999") for g in d["group"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 0.6 * len(d) + 1.8),
                                   gridspec_kw={"wspace": 0.08})

    panels = [
        (ax1, "Econ_TWh", "Economic potential [TWh/a]", "{:.2f}"),
        (ax2, "Lakes_above_min", f"Lakes with economic potential (≥ {cfg.min_project_capacity_kW:.0f} kW)", "{:,.0f}"),
    ]
    for ax, col, xlabel, fmt in panels:
        ax.barh(y, d[col], color=colors, height=0.6)
        ax.axvline(df.loc[df["group"] == "Base", col].iloc[0], color="k", ls="--", lw=1.2)
        xmax = max(d[col].max(), 1e-6)
        for yi, v in zip(y, d[col]):
            ax.text(v + xmax * 0.01, yi, fmt.format(v), va="center", fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, xmax * 1.15)
        ax.grid(axis="x", alpha=0.3)

    ax1.set_yticks(y)
    ax1.set_yticklabels(d["label"])
    groups = list(dict.fromkeys(df["group"]))
    handles = [mpatches.Patch(color=GROUP_COLORS.get(g, "#999999"), label=g) for g in groups]
    handles += [plt.Line2D([], [], color="k", ls="--", lw=1.2, label="base case")]
    fig.legend(handles=handles, fontsize=11, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=len(handles), frameon=False)
    plt.tight_layout(rect=(0, 0.04, 1, 1))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def main():
    cfg = Config()
    if not Path(cfg.ei_output_tif).exists():
        raise FileNotFoundError(f"{cfg.ei_output_tif} not found. Run main.py first.")
    if cfg.hub_grid_m != 1000 or cfg.ei_max_dist_m != 10000:
        print("  WARNING: base config differs from 1 km hubs / 10 km; check VARIANTS.")

    df = run_network_sensitivity(cfg)
    csv_path = Path(cfg.processed_dir) / f"economic_network_sensitivity_{cfg.cache_tag}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Results saved: {csv_path}")
    plot_network_sensitivity(cfg, df, Path(cfg.results_dir) / "Economic_Network_Sensitivity.pdf")


if __name__ == "__main__":
    main()
