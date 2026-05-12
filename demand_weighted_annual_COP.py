import pandas as pd
import matplotlib.pyplot as plt
from demandlib import bdew
import holidays as hol
from pathlib import Path

# =============================================================================
# PARAMETERS
# =============================================================================
ANNUAL_DEMAND_TOTAL_KWH = 6_000_000
ANNUAL_HEAT_DEMAND_KWH = ANNUAL_DEMAND_TOTAL_KWH * 0.85
ANNUAL_HOT_WATER_DEMAND_KWH = ANNUAL_DEMAND_TOTAL_KWH * 0.15
BUILDING_TYPE = "MFH"
BUILDING_CLASS = 2
WIND_CLASS = 0
STATE = "MV"

T_SUPPLY_C = 90
COP_C1 = 6.7
COP_C2 = 0.03697
COP_C3 = -0.03697
COP_C4 = 1.605e-7

# =============================================================================
# COP FORMULA
# =============================================================================
def calc_cop(temp_lake: float, t_supply: float) -> float:
    return COP_C1 + COP_C2 * temp_lake + COP_C3 * t_supply + COP_C4 * temp_lake * t_supply


# =============================================================================
# LOAD DATA
# =============================================================================
df_air = pd.read_csv(
    r"Data/Raw/schwerin_airtemp_1994_2020.csv",
    parse_dates=["datetime"],
    index_col="datetime",
)

df_lake = pd.read_csv(
    r"Data/Raw/schwerin_temperature_13.5m_hourly.csv",
    parse_dates=["datetime"],
    index_col="datetime",
)

# =============================================================================
# FILTER TO COMPLETE YEARS (present in BOTH datasets)
# =============================================================================
hours_air = df_air.groupby(df_air.index.year).size()
complete_air = set(hours_air[hours_air >= 8760].index)

hours_lake = df_lake.groupby(df_lake.index.year).size()
complete_lake = set(hours_lake[hours_lake >= 8760].index)

complete_years = sorted(complete_air & complete_lake)
print(f"Complete years in both datasets: {complete_years[0]}–{complete_years[-1]} ({len(complete_years)} years)")

# =============================================================================
# GENERATE DEMAND + COP PER YEAR → WEIGHTED ANNUAL COP
# =============================================================================
annual_results = []

for year in complete_years:
    # --- Demand profile ---
    year_air = df_air.loc[df_air.index.year == year]
    holidays_dict = dict(hol.Germany(state=STATE, years=year))

    heating = bdew.HeatBuilding(
        year_air.index,
        holidays=holidays_dict,
        temperature=year_air["airtemp_C"],
        shlp_type=BUILDING_TYPE,
        building_class=BUILDING_CLASS,
        wind_class=WIND_CLASS,
        annual_heat_demand=ANNUAL_HEAT_DEMAND_KWH,
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
        annual_heat_demand=ANNUAL_HOT_WATER_DEMAND_KWH,
        name="hotwater",
        ww_incl=True,
    ).get_bdew_profile()

    demand = (heating + hotwater).to_frame(name="demand_kW")

    # --- Lake temperature & COP ---
    year_lake = df_lake.loc[df_lake.index.year == year, ["temp_C"]].copy()
    year_lake["COP"] = year_lake["temp_C"].apply(lambda t: calc_cop(t, T_SUPPLY_C))

    # --- Merge on datetime index ---
    merged = demand.join(year_lake[["COP"]], how="inner")

    # --- Demand-weighted annual COP ---
    # COP_annual = ΣQ_demand / Σ(Q_demand / COP)
    total_demand = merged["demand_kW"].sum()
    total_elec = (merged["demand_kW"] / merged["COP"]).sum()
    cop_annual = total_demand / total_elec

    annual_results.append({
        "year": year,
        "total_demand_kWh": total_demand,
        "total_elec_kWh": total_elec,
        "COP_weighted": cop_annual,
        "COP_mean_unweighted": merged["COP"].mean(),
    })

    print(f"  {year}: weighted COP = {cop_annual:.3f}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
df_results = pd.DataFrame(annual_results)
print("\n" + "=" * 60)
print(df_results.to_string(index=False, float_format="%.3f"))
print("=" * 60)
print(f"\nOverall demand-weighted COP (all years): "
      f"{df_results['total_demand_kWh'].sum() / df_results['total_elec_kWh'].sum():.3f}")

# =============================================================================
# PLOT
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(df_results["year"], df_results["COP_weighted"], color="teal", edgecolor="k", linewidth=0.3)
ax.axhline(df_results["COP_weighted"].mean(), ls="--", color="grey", label="Mean across years")
ax.set_xlabel("Year")
ax.set_ylabel("Demand-weighted COP")
ax.set_title(f"Annual Demand-Weighted COP (T_supply = {T_SUPPLY_C} °C)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("annual_cop_weighted.png", dpi=150)
plt.show()