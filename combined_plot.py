# combined_plot.py
from pathlib import Path
import geopandas as gpd
from lake_io import read_lakes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_combined_map(cfg, lakes_theoretical, compare_df):
    """Side-by-side map: theoretical potential (left) vs economic potential (right)."""

    target_crs = cfg.target_crs
    lang = cfg.plot_language
    base_font = 11
    factor = 1 / 0.65

    plt.rcParams.update({
        "font.size": base_font * factor,
        "axes.titlesize": base_font * factor * 1.2,
        "axes.labelsize": base_font * factor,
        "legend.fontsize": base_font * factor * 0.85,
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

    # --- Load geometries ---
    germany_gdf = gpd.read_file(cfg.germany_path).to_crs(target_crs)
    states_gdf = gpd.read_file(cfg.nuts3_path).to_crs(target_crs)

    if cfg.nuts1_filter:
        nuts1 = gpd.read_file(cfg.nuts1_path).to_crs(target_crs)
        nuts1["NUTS_CODE"] = nuts1["NUTS_CODE"].astype(str)
        filtered_nuts = nuts1[nuts1["NUTS_CODE"].isin(cfg.nuts1_filter)]
        if filtered_nuts["NUTS_CODE"].duplicated().any():
            filtered_nuts = filtered_nuts[filtered_nuts["GF"] == 4]
        border_gdf = filtered_nuts
        states_gdf = states_gdf[states_gdf["NUTS_CODE"].str[:3].isin(cfg.nuts1_filter)]
        xmin, ymin, xmax, ymax = filtered_nuts.total_bounds
    else:
        border_gdf = germany_gdf
        xmin, ymin, xmax, ymax = germany_gdf.total_bounds

    width = xmax - xmin
    height = ymax - ymin

    # --- Prepare economic potential lake data ---
    lakes_gdf = read_lakes(cfg, target_crs)
    lakes_gdf["Hylak_id"] = lakes_gdf["Hylak_id"].astype(int)
    compare_df = compare_df.copy()
    compare_df["Hylak_id"] = compare_df["Hylak_id"].astype(int)
    lakes_economic = lakes_gdf.merge(
        compare_df[["Hylak_id", "Covered_Heat_GWh"]], on="Hylak_id", how="left"
    )
    lakes_economic["Covered_Heat_GWh"] = lakes_economic["Covered_Heat_GWh"].fillna(0)

    if cfg.nuts1_filter:
        lakes_economic = gpd.sjoin(
            lakes_economic, filtered_nuts[["NUTS_CODE", "geometry"]],
            how="inner", predicate="intersects"
        )
        if "index_right" in lakes_economic.columns:
            lakes_economic = lakes_economic.drop(columns="index_right")

    # --- Compute district highlighting for economic panel ---
    states_econ = states_gdf.copy()
    relevant = lakes_economic[lakes_economic["Covered_Heat_GWh"] > 0.0001]
    name_col = next((c for c in ["NUTS_NAME", "GEN", "NAME_3", "NAME"]
                     if c in states_econ.columns), None)

    if name_col and not relevant.empty:
        rel_clean = relevant.copy()
        if "index_right" in rel_clean.columns:
            rel_clean = rel_clean.drop(columns="index_right")
        joined = gpd.sjoin(rel_clean, states_econ[[name_col, "geometry"]],
                           how="inner", predicate="intersects")
        agg = joined.groupby(name_col)["Covered_Heat_GWh"].sum().reset_index()
        agg.columns = [name_col, "Total_GWh"]
        states_econ = states_econ.merge(agg, on=name_col, how="left")
        states_econ["Total_GWh"] = states_econ["Total_GWh"].fillna(0)
        states_econ["highlight"] = np.select(
            [states_econ["Total_GWh"] > 100, states_econ["Total_GWh"] > 0.0001],
            [2, 1], default=0
        )
    else:
        states_econ["highlight"] = 0

    # --- Reference circle values ---
    example_values = [50, 100, 1000]  # GWh/yr
    scale_factor_theo = cfg.scale_factor
    scale_factor_econ = cfg.scale_factor

    # --- Create figure ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 12))

    for idx, ax in enumerate(axes):
        # Draw borders
        border_gdf.boundary.plot(ax=ax, color="grey", linewidth=0.8, zorder=0)

        if idx == 0:
            # --- LEFT: Theoretical Potential ---
            states_gdf.boundary.plot(ax=ax, color="lightgrey", linewidth=0.4, zorder=1)

            title = "Theoretical Potential" if lang == "en" else "Theoretisches Potenzial"
            ax.set_title(title, fontweight="bold")
            bubble_color = "#91bfdb"
            value_col = "thermal_potential_GWh"
            data = lakes_theoretical
            sf = scale_factor_theo

            for _, lake in data.iterrows():
                val = lake[value_col]
                if np.isnan(val) or val <= 0:
                    continue
                x = lake.geometry.centroid.x
                y = lake.geometry.centroid.y
                radius = np.sqrt(val) * sf
                ax.add_patch(plt.Circle((x, y), radius, color=bubble_color,
                                        alpha=0.4, ec="none", zorder=2))

            data.plot(ax=ax, color="#08306b", edgecolor="none", markersize=3, zorder=3)

            # Legend
            pot_label = "Theoretical thermal potential" if lang == "en" else "Theoretisches thermisches Potenzial"
            lake_label = "Lakes" if lang == "en" else "Seen"
            ax.legend(handles=[
                mpatches.Patch(color=bubble_color, alpha=0.6, label=pot_label),
                mpatches.Patch(color="#08306b", label=lake_label),
            ], loc="upper left", frameon=True)

        else:
            # --- RIGHT: Economic Potential with district highlighting ---
            # Plot highlighted districts BEFORE boundaries
            states_econ[states_econ["highlight"] == 2].plot(
                ax=ax, color="#41ab5d", alpha=0.5, zorder=0)
            states_econ[states_econ["highlight"] == 1].plot(
                ax=ax, color="#c7e9c0", alpha=0.7, zorder=0)
            states_econ.boundary.plot(ax=ax, color="lightgrey", linewidth=0.3, zorder=1)

            title = "Economic Potential" if lang == "en" else "Wirtschaftliches Potenzial"
            ax.set_title(title, fontweight="bold")
            bubble_color = "#f58223"
            value_col = "Covered_Heat_GWh"
            data = lakes_economic
            sf = scale_factor_econ

            for _, lake in data.iterrows():
                val = lake[value_col]
                if np.isnan(val) or val <= 0:
                    continue
                x = lake.geometry.centroid.x
                y = lake.geometry.centroid.y

                if cfg.label_bodensee and int(lake["Hylak_id"]) == cfg.bodensee_id:
                    label = f"{val:.0f} GWh/yr" if lang == "en" else f"{val:.0f} GWh/a"
                    ax.text(x, y, label, ha="center", va="center",
                            fontsize=base_font * factor * 0.85,
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
                            zorder=7)
                    continue

                radius = np.sqrt(val) * sf
                ax.add_patch(plt.Circle((x, y), radius, color=bubble_color,
                                        alpha=0.4, ec=bubble_color, lw=0.2, zorder=2))

            data.plot(ax=ax, color="#08306b", edgecolor="none", markersize=3, zorder=3)

            # Redraw outer border on top
            border_gdf.boundary.plot(ax=ax, color="grey", linewidth=0.8, zorder=10)

            # Legend with district highlighting
            pot_label = "Economic potential" if lang == "en" else "Wirtschaftliches Potenzial"
            lake_label = "Lakes" if lang == "en" else "Seen"
            dist_lo = ("Districts with potential (>100 kWh)" if lang == "en"
                       else "Kreise mit Potenzial (>100 kWh)")
            dist_hi = ("Districts with potential (>100 GWh)" if lang == "en"
                       else "Kreise mit Potenzial (>100 GWh)")
            ax.legend(handles=[
                mpatches.Patch(color=bubble_color, alpha=0.6, label=pot_label),
                mpatches.Patch(color="#08306b", label=lake_label),
                mpatches.Patch(color="#c7e9c0", label=dist_lo),
                mpatches.Patch(color="#41ab5d", label=dist_hi),
            ], loc="upper left", frameon=True)

        # --- Reference circles (both panels) ---
        x_ref = xmax + 0.1 * width
        max_radius = max(np.sqrt(v) * sf for v in example_values)
        text_x = x_ref + max_radius * 1.2
        y_ref = ymin + 0.2 * height

        for j, val in enumerate(example_values):
            radius = np.sqrt(val) * sf
            if j == 0:
                y_ref += radius
            else:
                prev_radius = np.sqrt(example_values[j - 1]) * sf
                y_ref += prev_radius + radius + 0.04 * height
            ax.add_patch(plt.Circle((x_ref, y_ref), radius, color=bubble_color,
                                    alpha=0.4, ec="none", zorder=4))
            ax.text(text_x, y_ref, f"{val:.0f} GWh/yr", va="center",
                    fontsize=base_font * factor * 0.85)

        ax.set_xlim(xmin - 0.05 * width, xmax + 0.4 * width)
        ax.set_ylim(ymin - 0.05 * height, ymax + 0.1 * height)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    plt.tight_layout()

    out_path = str(Path(cfg.theoretical_plot_path).parent / "combined_potential_map.pdf")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Combined map saved: {out_path}")
