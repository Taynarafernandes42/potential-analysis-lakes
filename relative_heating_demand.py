import pandas as pd
import matplotlib.pyplot as plt
from demandlib import bdew
import holidays as hol

# =============================================================================
# PARAMETERS – adjust as needed
# =============================================================================
ANNUAL_DEMAND_TOTAL_KWH = 6_000_000  # Total annual demand (space heating + hot water) [kWh]
ANNUAL_HEAT_DEMAND_KWH = ANNUAL_DEMAND_TOTAL_KWH * 0.85     # Annual space heating demand [kWh]
ANNUAL_HOT_WATER_DEMAND_KWH = ANNUAL_DEMAND_TOTAL_KWH * 0.15  # Annual hot water demand [kWh]
BUILDING_TYPE = "MFH"
BUILDING_CLASS = 2
WIND_CLASS = 0
STATE = "MV"

# =============================================================================
# LOAD DATA
# =============================================================================
df = pd.read_csv(
    r"D:\Users\pau07368\PycharmProjects\potential-analysis-lakes-1"
    r"\Data\Raw\schwerin_airtemp_1994_2020.csv",
    parse_dates=["datetime"],
    index_col="datetime",
)

# =============================================================================
# FILTER TO COMPLETE YEARS ONLY
# =============================================================================
hours_per_year = df.groupby(df.index.year).size()
complete_years = hours_per_year[hours_per_year >= 8760].index.tolist()
df = df[df.index.year.isin(complete_years)]
print(f"Complete years: {complete_years[0]}–{complete_years[-1]} ({len(complete_years)} years)")

# =============================================================================
# GENERATE DEMAND PROFILES
# =============================================================================
profiles = []

for year in complete_years:
    year_data = df.loc[df.index.year == year]
    holidays_dict = dict(hol.Germany(state=STATE, years=year))

    heating = bdew.HeatBuilding(
        year_data.index,
        holidays=holidays_dict,
        temperature=year_data["airtemp_C"],
        shlp_type=BUILDING_TYPE,
        building_class=BUILDING_CLASS,
        wind_class=WIND_CLASS,
        annual_heat_demand=ANNUAL_HEAT_DEMAND_KWH,
        name="heating",
        ww_incl=False,
    ).get_bdew_profile()

    hotwater = bdew.HeatBuilding(
        year_data.index,
        holidays=holidays_dict,
        temperature=year_data["airtemp_C"],
        shlp_type=BUILDING_TYPE,
        building_class=BUILDING_CLASS,
        wind_class=WIND_CLASS,
        annual_heat_demand=ANNUAL_HOT_WATER_DEMAND_KWH,
        name="hotwater",
        ww_incl=True,
    ).get_bdew_profile()

    profiles.append(heating + hotwater)
    print(f"  {year} done")

result = pd.concat(profiles)
result.name = "total_demand_kW"

# =============================================================================
# PLOT
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(result.index, result.values, linewidth=0.2, color="firebrick")
ax.set_ylabel("Demand [kW]")
ax.set_xlabel("Time")
ax.set_title(f"District Heating + Hot Water Demand – Schwerin ({complete_years[0]}–{complete_years[-1]})")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("district_heating_demand.png", dpi=150)
plt.show()