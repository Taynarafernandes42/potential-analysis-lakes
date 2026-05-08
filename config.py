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

    # CRS
    target_crs = "EPSG:3035"

    # Thermal potential parameters
    delta_T = 1.0           # K
    usable_factor = 1.0
    cp_water = 4.18         # kJ/(kg·K)
    density_water = 1000    # kg/m³

    # Plot parameters
    scale_factor = 1000
    example_values = [50, 100, 1000]  # GWh reference circles
    label_bodensee = False
    bodensee_id = 1243

    @property
    def theoretical_csv(self):
        suffix = f"_{self.nuts1_filter[0]}" if self.nuts1_filter else ""
        return str(BASE / f"Data/Processed/theoretical_potential_by_lake{suffix}.csv")

    @property
    def theoretical_plot_path(self):
        suffix = f"_{self.nuts1_filter[0]}" if self.nuts1_filter else ""
        return str(Path(self.results_dir) / f"Theoretical_Potential{suffix}.pdf")