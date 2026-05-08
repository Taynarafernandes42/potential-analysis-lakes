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
    heat_path = str(BASE / "Data/Raw/heat_tot_curr_density_germany.tif")

    # Output directories
    results_dir = str(BASE / "Data/Results")
    processed_dir = str(BASE / "Data/Processed")

    # CRS
    target_crs = "EPSG:3035"

    # Physical constants
    cp_water = 4184         # J/(kg·K)
    density_water = 998     # kg/m³

    # --- Theoretical potential ---
    delta_T = 1.0
    usable_factor = 1.0
    scale_factor = 1000
    example_values = [50, 100, 1000]
    label_bodensee = False
    bodensee_id = 1243

    # --- Technical-extractable (technical_extractable) potential ---
    deltaT_hex = 5.0
    deltaT_technical_extractable = 1.0
    hours_per_year = 3500

    # Technical sensitivity
    run_technical_sensitivity = True
    technical_sensitivity_h_values = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]

    # --- Economic potential (EI) ---
    c_el_L = 0.37           # €/kWh electricity (air-source HP, decentralized)
    c_el_W = 0.16           # €/kWh electricity (water-source HP, centralized)
    lifetime = 20           # years
    c_pipe = 2500.0         # €/m pipe cost
    COP_Water = 3.3
    COP_Air = 2.9
    q_loss = 0.12           # heat loss fraction in pipe network
    inflation_rate = 0.02

    CAPEX_Air = 1872.0      # €/kWth
    CAPEX_Water = 1064.0    # €/kWth
    hours_Air = 2200        # h/a full load hours air HP
    hours_Water = 3500      # h/a full load hours water HP

    # Distance / Hub parameters
    ei_max_dist_m = 10000
    min_pipe_length_m = 1.0
    distance_mode = "hub"       # "pixel" or "hub"
    hub_grid_m = 1000           # grid cell size for hub aggregation
    hub_share_mode = "weighted" # "weighted" or "equal"

    # EI threshold for "economically viable"
    ei_threshold = 1.0

    # Computation
    n_jobs = 4

    # Economic sensitivity
    run_economic_sensitivity = True
    sensitivity_param_ranges = {
        "c_el_L": (0.25, 0.45),
        "c_el_W": (0.12, 0.20),
        "c_pipe": (2000, 3500),
        "COP_Water": (3.0, 4.5),
        "COP_Air": (2.5, 3.5),
        "q_loss": (0.10, 0.20),
        "CAPEX_Air": (1500, 2500),
        "CAPEX_Water": (750, 1750),
        "hours_Air": (1800, 2400),
        "hours_Water": (2000, 5000),
    }
    sensitivity_n_steps = 10
    sensitivity_plot_params = [
        "CAPEX_Air", "hours_Air", "c_el_L", "COP_Air",
        "CAPEX_Water", "hours_Water", "c_el_W", "COP_Water", "q_loss", "c_pipe",
    ]

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
    def technical_sensitivity_plot_path(self):
        return str(Path(self.results_dir) / "Sensitivity_Full_Load_Hours.pdf")

    @property
    def ei_output_tif(self):
        return str(Path(self.processed_dir) / "EI_results.tif")

    @property
    def economic_compare_csv(self):
        suffix = f"_{self.nuts1_filter[0]}" if self.nuts1_filter else ""
        return str(Path(self.processed_dir) / f"economic_potential_comparison{suffix}.csv")

    @property
    def economic_scatter_plot_path(self):
        return str(Path(self.results_dir) / "EI_Scatter_Plot.png")

    @property
    def economic_map_plot_path(self):
        return str(Path(self.results_dir) / "Economic_Potential_Map.pdf")

    @property
    def economic_sensitivity_plot_path(self):
        return str(Path(self.results_dir) / "Economic_Sensitivity.pdf")