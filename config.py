from pathlib import Path

BASE = Path(__file__).resolve().parent


class Config:
    # Language
    plot_language = "en"

    # Filter (None = all of Germany)
    nuts1_filter = None

    # Input paths
    lakes_path = str(BASE / "Data/Raw/HydroLAKES_germany.shp")
    nuts1_path = str(BASE / "Data/Raw/NUTS250_N1.shp")
    nuts3_path = str(BASE / "Data/Raw/NUTS250_N3.shp")
    germany_path = str(BASE / "Data/Raw/Germany.shp")

    # Output directory
    results_dir = str(BASE / "Data/Results")
    processed_dir = str(BASE / "Data/Processed")

    # CRS
    target_crs = "EPSG:3035"

    # Physical constants
    cp_water = 4184         # J/(kg·K)
    density_water = 998     # kg/m³

    # --- Theoretical potential parameters ---
    delta_T = 1.0           # K
    usable_factor = 1.0

    # Plot: proportional circles
    scale_factor = 1000
    example_values = [50, 100, 1000]  # GWh reference circles
    label_bodensee = False
    bodensee_id = 1243

    # --- Technical-extractable potential parameters ---
    deltaT_hex = 5.0            # K, heat exchanger side
    deltaT_technical_extractable = 1.0    # K, allowed lake temperature change
    hours_per_year = 3500       # full load hours

    # Sensitivity analysis
    run_sensitivity = True
    sensitivity_h_values = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]

    # --- Derived paths ---
    @property
    def theoretical_csv(self):
        suffix = f"_{self.nuts1_filter[0]}" if self.nuts1_filter else ""
        return str(Path(self.processed_dir) / f"theoretical_potential_by_lake{suffix}.csv")

    @property
    def theoretical_plot_path(self):
        suffix = f"_{self.nuts1_filter[0]}" if self.nuts1_filter else ""
        return str(Path(self.results_dir) / f"Theoretical_Potential{suffix}.pdf")

    @property
    def technical_extractable_csv(self):
        suffix = f"_{self.nuts1_filter[0]}" if self.nuts1_filter else ""
        return str(Path(self.processed_dir) / f"technical_extractable_potential_by_lake{suffix}.csv")

    @property
    def sensitivity_plot_path(self):
        return str(Path(self.results_dir) / "Sensitivity_Full_Load_Hours.pdf")