"""Combined scenario analysis for the economic potential (standalone script).

Run after main.py, which creates the EI raster (Data/Processed/EI_results.tif) and the
technically extractable potential per lake. Usage:

    python scenario_analysis.py

Each scenario in Config.scenarios changes several parameters at once relative to the base
case (scenario A) and recomputes the economic potential with the same method as the OAT
sensitivity analysis in economic_potential.py:
EI per demand cell -> eligible demand per lake (EI >= threshold) -> capped by the technically
extractable potential converted to delivered heat -> minimum project size.

Scenario parameters: all EI parameters of the OAT analysis (c_el_L, c_el_W, c_pipe, COP_Water,
COP_Air, q_loss, CAPEX_Air, CAPEX_Water, hours_Air, hours_Water, wacc_decentral, wacc_central)
plus
  demand_factor   scales the heat demand of all cells (e.g. 0.7 = -30 % after renovation)
  deltaT_sus      permissible lake cooling [K]; the technically extractable potential
                  scales linearly with it (base: cfg.deltaT_technical_extractable)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import Config, lake_to_useful
from economic_potential import _annuity_factor, _load_ei_results


EI_PARAMS = [
    "c_el_L", "c_el_W", "c_pipe", "COP_Water", "COP_Air", "q_loss",
    "CAPEX_Air", "CAPEX_Water", "hours_Air", "hours_Water",
    "wacc_decentral", "wacc_central",
]

GROUP_COLORS = {"Base": "#555555", "Techno-economic": "#f58220", "Lake resource": "#1976d2"}


def _base_params(cfg):
    p = {k: getattr(cfg, k) for k in EI_PARAMS}
    p["demand_factor"] = 1.0
    p["deltaT_sus"] = cfg.deltaT_technical_extractable
    return p


def _scenario_potential(cfg, p, E_h_base, D_base, HI_base, pot_lake_GWh):
    """Economic potential per lake for one parameter set (same EI as the OAT analysis)."""
    af_decentral = _annuity_factor(p["wacc_decentral"], cfg.lifetime)
    af_central = _annuity_factor(p["wacc_central"], cfg.lifetime)

    E_h = E_h_base * p["demand_factor"]            # kWh, consumer side
    E_h_W = E_h * (1 + p["q_loss"])                # kWh, LSWSHP output

    annual_Air = (p["CAPEX_Air"] * (E_h / p["hours_Air"]) * af_decentral
                  + p["c_el_L"] * E_h / p["COP_Air"])
    annual_Water = (p["CAPEX_Water"] * (E_h_W / p["hours_Water"]) * af_central
                    + p["c_el_W"] * E_h_W / p["COP_Water"])
    annual_pipe = p["c_pipe"] * D_base * af_central
    annual_pipe = np.where(annual_pipe == 0, np.nan, annual_pipe)

    EI = (annual_Air - annual_Water) / annual_pipe
    mask = EI >= cfg.ei_threshold
    if not np.any(mask):
        return pd.DataFrame(columns=["Hylak_id", "Covered_Heat_GWh", "HP_Capacity_kW", "Above_Min_Size"])

    uids, inv = np.unique(HI_base[mask], return_inverse=True)
    eligible_GWh = np.bincount(inv, weights=E_h[mask]).astype(float) / 1e6

    dT_factor = p["deltaT_sus"] / cfg.deltaT_technical_extractable
    pot_lake = np.array([pot_lake_GWh.get(int(u), 0.0) for u in uids]) * dT_factor
    pot_del = lake_to_useful(pot_lake, p["COP_Water"], p["q_loss"])

    covered = np.minimum(eligible_GWh, pot_del)
    df = pd.DataFrame({
        "Hylak_id": uids.astype(int),
        "Covered_Heat_GWh": covered,
        "HP_Capacity_kW": covered * 1e6 * (1 + p["q_loss"]) / p["hours_Water"],
    })
    df["Above_Min_Size"] = df["HP_Capacity_kW"] >= cfg.min_project_capacity_kW
    return df[df["Covered_Heat_GWh"] > 0]


def run_scenarios(cfg, hylak_data, dist_data, heat_data):
    """Compute all scenarios in cfg.scenarios. Returns a summary DataFrame."""
    mask_base = np.isfinite(dist_data) & (hylak_data > 0) & np.isfinite(heat_data) & (heat_data > 0)
    E_h_base = heat_data[mask_base].astype("float64") * 1000.0   # MWh -> kWh
    D_base = dist_data[mask_base].astype("float64")
    HI_base = hylak_data[mask_base].astype("int64")

    pot_df = pd.read_csv(cfg.technical_extractable_csv)
    pot_lake_GWh = dict(zip(pot_df["Hylak_id"].astype(int), pot_df["potential_Wh"] / 1e9))

    base = _base_params(cfg)
    rows = []
    for key, sc in cfg.scenarios.items():
        unknown = set(sc["params"]) - set(base)
        if unknown:
            raise KeyError(f"Scenario {key}: unknown parameters {sorted(unknown)}")
        p = {**base, **sc["params"]}
        df = _scenario_potential(cfg, p, E_h_base, D_base, HI_base, pot_lake_GWh)
        above = df["Above_Min_Size"]
        rows.append({
            "scenario": key,
            "label": sc["label"],
            "group": sc["group"],
            "changes": "; ".join(f"{k}={v}" for k, v in sc["params"].items()) or "-",
            "Econ_before_min_TWh": df["Covered_Heat_GWh"].sum() / 1000.0,
            "Econ_TWh": df.loc[above, "Covered_Heat_GWh"].sum() / 1000.0,
            "Lakes_with_potential": int(len(df)),
            "Lakes_above_min": int(above.sum()),
            "Lake_heat_extracted_TWh": float(
                (df.loc[above, "Covered_Heat_GWh"] * (1 + p["q_loss"]) * (1 - 1 / p["COP_Water"])).sum() / 1000.0
            ),
        })

    out = pd.DataFrame(rows)
    base_key = next(k for k, sc in cfg.scenarios.items() if not sc["params"])
    base_val = out.loc[out["scenario"] == base_key, "Econ_TWh"].iloc[0]
    out["Change_vs_base_pct"] = (out["Econ_TWh"] / base_val - 1.0) * 100 if base_val > 0 else np.nan

    print("\n" + "=" * 78)
    print("  SCENARIO ANALYSIS  (economic potential incl. minimum project size)")
    print("=" * 78)
    print(f"  {'Scenario':<38}{'before min':>11}{'TWh/a':>9}{'vs. A':>10}{'lakes':>8}")
    for _, r in out.iterrows():
        print(f"  {r['scenario'] + ': ' + r['label']:<38}{r['Econ_before_min_TWh']:>11.2f}{r['Econ_TWh']:>9.2f}"
              f"{r['Change_vs_base_pct']:>9.2f}%{r['Lakes_above_min']:>8d}")
    print("=" * 78 + "\n")
    return out


def plot_scenarios(cfg, df):
    """Economic potential (left) and number of lakes with economic potential (right) per scenario."""
    plt.rcParams.update({"font.size": 14, "font.family": "serif", "mathtext.fontset": "cm"})

    d = df.iloc[::-1].reset_index(drop=True)            # scenario A on top
    y = np.arange(len(d))
    colors = [GROUP_COLORS.get(g, "#999999") for g in d["group"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 0.55 * len(d) + 1.8),
                                   gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08})

    panels = [
        (ax1, "Econ_TWh", "Economic potential [TWh/a]", "{:.2f}"),
        (ax2, "Lakes_above_min", f"Lakes with economic potential", "{:,.0f}"),
    ]
    for ax, col, xlabel, fmt in panels:
        ax.barh(y, d[col], color=colors, height=0.6)
        base_val = df.loc[df["group"] == "Base", col].iloc[0]
        ax.axvline(base_val, color="k", ls="--", lw=1.2)
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
    fig.legend(handles=handles, fontsize=11, loc="lower center", bbox_to_anchor=(0.5, -0.09),
               ncol=len(handles), frameon=False)
    plt.tight_layout(rect=(0, 0.04, 1, 1))

    out = Path(cfg.scenario_plot_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Scenario plot saved: {out}")


def main():
    cfg = Config()
    if not Path(cfg.ei_output_tif).exists():
        raise FileNotFoundError(f"{cfg.ei_output_tif} not found. Run main.py first.")

    print("  Loading EI raster and heat demand...")
    _, hylak_data, dist_data, _, heat_data, _, _, _ = _load_ei_results(cfg)

    df = run_scenarios(cfg, hylak_data, dist_data, heat_data)
    Path(cfg.scenario_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.scenario_csv, index=False)
    print(f"  Scenario results saved: {cfg.scenario_csv}")
    plot_scenarios(cfg, df)


if __name__ == "__main__":
    main()