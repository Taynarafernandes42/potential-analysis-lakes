from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def calculate_technical_extractable_potential(cfg):
    """Calculate technical-extractable (technical_extractable) thermal potential."""
    lakes = gpd.read_file(cfg.lakes_path).to_crs(cfg.target_crs)

    # NUTS1 filter
    if cfg.nuts1_filter:
        nuts1 = gpd.read_file(cfg.nuts1_path).to_crs(cfg.target_crs)
        nuts1["NUTS_CODE"] = nuts1["NUTS_CODE"].astype(str)
        filtered_nuts = nuts1[nuts1["NUTS_CODE"].isin(cfg.nuts1_filter)]
        if filtered_nuts["NUTS_CODE"].duplicated().any():
            filtered_nuts = filtered_nuts[filtered_nuts["GF"] == 4]
        lakes = gpd.sjoin(lakes, filtered_nuts[["NUTS_CODE", "geometry"]],
                          how="inner", predicate="intersects")
        if "index_right" in lakes.columns:
            lakes = lakes.drop(columns="index_right")

    lakes["Hylak_id"] = lakes["Hylak_id"].astype(int)

    # Flow rate: volume per year → m³/s
    seconds_per_year = 365 * 24 * 3600
    lakes["flow_rate_m3s"] = lakes["Vol_total"] * 1e6 / seconds_per_year

    # technical_extractable withdrawal rate
    lakes["withdrawal_m3s"] = lakes["flow_rate_m3s"] * (cfg.deltaT_technical_extractable / cfg.deltaT_hex)

    # Thermal capacity [W] and potential [Wh]
    lakes["capacity_W"] = lakes["withdrawal_m3s"] * cfg.density_water * cfg.cp_water * cfg.deltaT_hex
    lakes["potential_Wh"] = lakes["capacity_W"] * cfg.hours_per_year
    lakes["potential_TWh"] = lakes["potential_Wh"] / 1e12

    total_TWh = lakes["potential_TWh"].sum()
    region = cfg.nuts1_filter if cfg.nuts1_filter else "all lakes"
    print(f"Total Technical-Extractable Potential ({region}): {total_TWh:.2f} TWh/year")

    # Save CSV
    out_path = Path(cfg.technical_extractable_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lakes[["Hylak_id", "potential_Wh"]].to_csv(out_path, index=False)
    print(f"Written: {out_path} ({len(lakes)} lakes)")

    return lakes, total_TWh


def sensitivity_analysis(cfg):
    """Run potential calculation for multiple full-load-hour values."""
    lakes = gpd.read_file(cfg.lakes_path).to_crs(cfg.target_crs)

    if cfg.nuts1_filter:
        nuts1 = gpd.read_file(cfg.nuts1_path).to_crs(cfg.target_crs)
        nuts1["NUTS_CODE"] = nuts1["NUTS_CODE"].astype(str)
        filtered_nuts = nuts1[nuts1["NUTS_CODE"].isin(cfg.nuts1_filter)]
        if filtered_nuts["NUTS_CODE"].duplicated().any():
            filtered_nuts = filtered_nuts[filtered_nuts["GF"] == 4]
        lakes = gpd.sjoin(lakes, filtered_nuts[["NUTS_CODE", "geometry"]],
                          how="inner", predicate="intersects")
        if "index_right" in lakes.columns:
            lakes = lakes.drop(columns="index_right")

    seconds_per_year = 365 * 24 * 3600
    lakes["flow_rate_m3s"] = lakes["Vol_total"] * 1e6 / seconds_per_year
    lakes["withdrawal_m3s"] = lakes["flow_rate_m3s"] * (cfg.deltaT_technical_extractable / cfg.deltaT_hex)
    capacity_total_W = (lakes["withdrawal_m3s"] * cfg.density_water * cfg.cp_water * cfg.deltaT_hex).sum()

    results = []
    for h in cfg.sensitivity_h_values:
        total_TWh = capacity_total_W * h / 1e12
        results.append({"h": h, "Total_Potential_TWh": total_TWh})

    df = pd.DataFrame(results)

    # Relative change vs. base
    base_val = df.loc[df["h"] == cfg.hours_per_year, "Total_Potential_TWh"]
    if base_val.empty:
        base_val = df.iloc[(np.abs(df["h"] - cfg.hours_per_year)).argmin()]["Total_Potential_TWh"]
    else:
        base_val = base_val.values[0]
    df["RelChange_pct"] = (df["Total_Potential_TWh"] - base_val) / base_val * 100.0

    return df


def plot_sensitivity(cfg, df_sens):
    """Bar chart: total technical_extractable potential vs. full-load hours."""
    base_font = 11
    factor = 1 / 0.75
    plt.rcParams.update({
        "font.size": base_font * factor,
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

    scenario_color = "#179c7d"
    base_color = "#f58220"

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(df_sens["h"].astype(str), df_sens["Total_Potential_TWh"],
                  color=scenario_color, width=0.6)

    # Highlight base scenario
    base_indices = df_sens.index[df_sens["h"] == cfg.hours_per_year].tolist()
    if base_indices:
        bars[base_indices[0]].set_color(base_color)

    # Value labels
    for rect, val in zip(bars, df_sens["Total_Potential_TWh"]):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.5,
                f"{val:,.2f}", ha="center", va="bottom", fontsize=13)

    ax.set_xlabel("Full Load Hours (h)")
    ax.set_ylabel("Total Technical-Extractable Thermal Potential [TWh]")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, df_sens["Total_Potential_TWh"].max() * 1.15)
    ax.legend(handles=[
        mpatches.Patch(color=scenario_color, label="Scenarios"),
        mpatches.Patch(color=base_color, label="Base scenario"),
    ], loc="upper left")

    plt.tight_layout()
    Path(cfg.sensitivity_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cfg.sensitivity_plot_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(cfg.sensitivity_plot_path.replace(".pdf", ".svg"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    from config import Config

    cfg = Config()
    lakes, total = calculate_technical_extractable_potential(cfg)

    if cfg.run_sensitivity:
        df_sens = sensitivity_analysis(cfg)
        print(df_sens)
        plot_sensitivity(cfg, df_sens)