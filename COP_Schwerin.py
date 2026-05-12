import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config import Config

# ============================================================
# Configuration
# ============================================================

cfg = Config()

OUTPUT_DIR = Path(cfg.results_dir) / "CaseStudy_Schwerin"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_INPUT = "Data/Raw/schwerin_temperature_13.5m_hourly.csv"
CSV_OUTPUT = OUTPUT_DIR / "schwerin_cop.csv"
PLOT_OUTPUT = OUTPUT_DIR / "schwerin_correlation_plot.png"

# Heat pump supply temperature [°C]
T_SUPPLY_C = 90  # e.g. 90 °C

# COP formula coefficients
COP_C1 = 6.7
COP_C2 = 0.03697
COP_C3 = -0.03697
COP_C4 = 1.605e-7

# ============================================================
# COP Calculation
# ============================================================

def calc_cop(temp_lake: float, t_supply: float) -> float:
    """COP = C1 + C2*T_lake + C3*T_supply + C4*T_lake*T_supply"""
    return (
        COP_C1
        + COP_C2 * temp_lake
        + COP_C3 * t_supply
        + COP_C4 * temp_lake * t_supply
    )

def main():
    # Read CSV
    df = pd.read_csv(CSV_INPUT, parse_dates=["datetime"])

    # Calculate COP for each time step
    df["T_supply_C"] = T_SUPPLY_C
    df["COP"] = df["temp_C"].apply(lambda t_lake: calc_cop(t_lake, T_SUPPLY_C))

    # Print mean and median COP
    mean_cop = df["COP"].mean()
    median_cop = df["COP"].median()
    print(f"Mean COP: {mean_cop:.2f}")
    print(f"Median COP: {median_cop:.2f}")

    # Save results
    df.to_csv(CSV_OUTPUT, index=False)
    print(f"\nResults saved to: {CSV_OUTPUT}")


if __name__ == "__main__":
    main()