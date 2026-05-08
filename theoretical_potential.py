from pathlib import Path
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def calculate(cfg):
    """Calculate theoretical thermal potential for all lakes."""
    lakes = gpd.read_file(cfg.lakes_path).to_crs(cfg.target_crs)

    if cfg.nuts1_filter:
        nuts1 = gpd.read_file(cfg.nuts1_path).to_crs(cfg.target_crs)
        nuts1["NUTS_CODE"] = nuts1["NUTS_CODE"].astype(str)
        filtered_nuts = nuts1[nuts1["NUTS_CODE"].isin(cfg.nuts1_filter)]
        if filtered_nuts["NUTS_CODE"].duplicated().any():
            filtered_nuts = filtered_nuts[filtered_nuts["GF"] == 4]
        lakes = gpd.sjoin(lakes, filtered_nuts[["NUTS_CODE", "geometry"]],
                          how="inner", predicate="intersects")

    lakes["Hylak_id"] = lakes["Hylak_id"].astype(int)
    lakes["thermal_potential_kWh"] = (
        cfg.usable_factor * lakes["Vol_total"] * 1e6
        * cfg.density_water * cfg.cp_water * cfg.delta_T / 3_600_000
    )
    lakes["thermal_potential_GWh"] = lakes["thermal_potential_kWh"] / 1e6

    Path(cfg.theoretical_csv).parent.mkdir(parents=True, exist_ok=True)
    lakes[["Hylak_id", "thermal_potential_GWh"]].to_csv(cfg.theoretical_csv, index=False)

    total_TWh = lakes["thermal_potential_GWh"].sum() / 1000
    region = cfg.nuts1_filter if cfg.nuts1_filter else "all lakes"
    print(f"  Theoretical Potential ({region}): {total_TWh:,.2f} TWh/year")

    return lakes


def plot(cfg, lakes):
    """Plot lakes with proportional circles for thermal potential."""
    germany_gdf = gpd.read_file(cfg.germany_path).to_crs(cfg.target_crs)
    states_gdf = gpd.read_file(cfg.nuts3_path).to_crs(cfg.target_crs)

    lang = cfg.plot_language
    scale_factor = cfg.scale_factor
    base_font = 11
    factor = 1 / 0.75

    plt.rcParams.update({
        "font.size": base_font * factor,
        "axes.titlesize": base_font * factor * 1.2,
        "axes.labelsize": base_font * factor,
        "xtick.labelsize": base_font * factor * 0.9,
        "ytick.labelsize": base_font * factor * 0.9,
        "legend.fontsize": base_font * factor * 0.9,
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

    if cfg.nuts1_filter:
        nuts1_gdf = gpd.read_file(cfg.nuts1_path).to_crs(cfg.target_crs)
        nuts1_gdf["NUTS_CODE"] = nuts1_gdf["NUTS_CODE"].astype(str)
        filtered_nuts = nuts1_gdf[nuts1_gdf["NUTS_CODE"].isin(cfg.nuts1_filter)]
        states_gdf = states_gdf[states_gdf["NUTS_CODE"].str[:3].isin(cfg.nuts1_filter)]
        xmin, ymin, xmax, ymax = filtered_nuts.total_bounds
    else:
        xmin, ymin, xmax, ymax = germany_gdf.total_bounds

    fig, ax = plt.subplots(figsize=(10, 12))

    if cfg.nuts1_filter:
        filtered_nuts.boundary.plot(ax=ax, color="grey", linewidth=0.8)
    else:
        germany_gdf.boundary.plot(ax=ax, color="grey", linewidth=0.8)
    states_gdf.boundary.plot(ax=ax, color="lightgrey", linewidth=0.4, zorder=1)

    width = xmax - xmin
    height = ymax - ymin
    ax.set_xlim(xmin - 0.05 * width, xmax + 0.4 * width)
    ax.set_ylim(ymin - 0.05 * height, ymax + 0.1 * height)

    for _, lake in lakes.iterrows():
        val = lake["thermal_potential_GWh"]
        if np.isnan(val) or val <= 0:
            continue
        x = lake.geometry.centroid.x
        y = lake.geometry.centroid.y

        if cfg.label_bodensee and int(lake["Hylak_id"]) == cfg.bodensee_id:
            val_twh = val / 1000
            label = f"{val_twh:.1f} TWh/yr" if lang == "en" else f"{val_twh:.1f} TWh/a"
            ax.text(x, y, label, ha="center", va="center",
                    fontsize=base_font * factor * 0.9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7), zorder=5)
            continue

        radius = np.sqrt(val) * scale_factor
        ax.add_patch(plt.Circle((x, y), radius, color="#91bfdb", alpha=0.4, ec="none", zorder=2))

    lakes.plot(ax=ax, color="#08306b", edgecolor="none", markersize=3, zorder=3)

    # Reference circles
    x_ref = xmax + 0.1 * width
    max_radius = max(np.sqrt(v) * scale_factor for v in cfg.example_values)
    text_x = x_ref + max_radius * 1.1
    y_ref = ymin + 0.25 * height

    for j, val in enumerate(cfg.example_values):
        radius = np.sqrt(val) * scale_factor
        if j == 0:
            y_ref += radius
        else:
            prev_radius = np.sqrt(cfg.example_values[j - 1]) * scale_factor
            y_ref += prev_radius + radius + 0.05 * height
        ax.add_patch(plt.Circle((x_ref, y_ref), radius, color="#91bfdb", alpha=0.4, ec="none", zorder=4))
        ax.text(text_x, y_ref, f"{val:.0f} GWh/yr", va="center")

    pot_label = "Theoretical thermal potential" if lang == "en" else "Theoretisches Thermisches Potenzial"
    lake_label = "Lakes" if lang == "en" else "Seen"
    ax.legend(handles=[
        mpatches.Patch(color="#91bfdb", label=pot_label),
        mpatches.Patch(color="#08306b", label=lake_label),
    ], loc="upper left", frameon=True)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    plt.tight_layout()

    Path(cfg.theoretical_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cfg.theoretical_plot_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(cfg.theoretical_plot_path.replace(".pdf", ".svg"), bbox_inches="tight")
    plt.close()