from pathlib import Path

BASE = Path(__file__).resolve().parent

# Lake-side heat extraction vs. consumer-side useful heat:
# Q_lake = Q_useful * (1 + q_loss) * (1 - 1/COP)
# Q_useful = Q_lake / ((1 + q_loss) * (1 - 1/COP))
# Q_lake = heat extracted from the lake, Q_useful = useful heat delivered to the consumer,
# q_loss = network heat loss fraction, COP = heat-pump COP (scalar or hourly array).

def useful_to_lake(q_useful, cop, q_loss):
    """Consumer-side useful heat -> heat that must be extracted from the lake."""
    return q_useful * (1.0 + q_loss) * (1.0 - 1.0 / cop)


def lake_to_useful(q_lake, cop, q_loss):
    """Heat extracted from the lake -> consumer-side useful heat it can supply."""
    return q_lake / ((1.0 + q_loss) * (1.0 - 1.0 / cop))


class Config:
    # Language
    plot_language = "en"

    use_cache = True
    cache_tag = "useful_heat_v1"

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
    cp_water = 4180         # J/(kg·K)
    density_water = 1000     # kg/m³

    # --- Theoretical potential ---
    delta_T = 1.0
    usable_factor = 1.0
    scale_factor = 1000
    example_values = [50, 100, 1000]
    label_bodensee = False
    bodensee_id = 1243

    # --- Technical-extractable potential ---
    deltaT_hex = 5.0
    deltaT_technical_extractable = 1.0
    hours_per_year = 3500

    # Technical sensitivity
    run_technical_sensitivity = True
    technical_sensitivity_h_values = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]

    # --- Economic potential (EI) ---
    c_el_L = 0.37           # €/kWh electricity (air-source HP, decentralized)
    c_el_W = 0.17           # €/kWh electricity (water-source HP, centralized)
    lifetime = 20           # years
    c_pipe = 2500.0         # €/m pipe cost
    COP_Water = 3.3
    COP_Air = 2.8
    q_loss = 0.12           # heat loss fraction in pipe network
    wacc_decentral = 0.02   # 4% nominal WACC - 2% inflation = 2% realWACC 
    wacc_central = 0.06     # 8% nominal WACC - 2% inflation = 2% realWACC 

    CAPEX_Air = 1901.0      # €/kWth
    CAPEX_Water = 1080.0    # €/kWth
    hours_Air = 2150        # h/a full load hours air HP
    hours_Water = 3500      # h/a full load hours water HP

    # Distance / Hub parameters
    ei_max_dist_m = 10000
    min_pipe_length_m = 1.0
    distance_mode = "hub"       # "pixel" or "hub"
    hub_grid_m = 1000           # grid cell size for hub aggregation
    hub_share_mode = "weighted" # "weighted" or "equal"

    # EI threshold for "economically viable"
    ei_threshold = 1.0

    # Minimum project size, applied at the very end of the economic assessment.
    # Below this heat-pump capacity the cost assumptions (large-scale LSWSHP,
    # central financing, district heating) are no longer valid.
    # Capacity = Covered_Heat * (1 + q_loss) / hours_Water  (HP thermal output)
    min_project_capacity_kW = 100.0

    # Computation
    n_jobs = 4

    # Economic sensitivity
    run_economic_sensitivity = True
    sensitivity_param_ranges = {
        "c_el_L": (0.25, 0.47),
        "c_el_W": (0.15, 0.25),
        "c_pipe": (2000, 3500),
        "COP_Water": (3.0, 4.0),
        "COP_Air": (2.6, 3.75),
        "q_loss": (0.07, 0.20),
        "CAPEX_Air": (1380, 2240),
        "CAPEX_Water": (520, 1500),
        "hours_Air": (2000, 2300),
        "hours_Water": (2000, 5000),
        "wacc_decentral": (0.01, 0.03),
        "wacc_central": (0.03, 0.09), 
    }
    sensitivity_n_steps = 10
    sensitivity_plot_params = [
        "CAPEX_Air", "hours_Air", "c_el_L", "COP_Air",
        "CAPEX_Water", "hours_Water", "c_el_W", "COP_Water", "wacc_central", "q_loss", "c_pipe",  
    ]

    # Combined scenarios (scenario_analysis.py, run after the OAT sensitivity analysis)
    # Scenario A is the base case. Values not listed keep the base case. The plot follows this order.
    # Groups: "Base", "Techno-economic" (prices, costs, COP), "Lake resource" (permissible lake cooling).
    # WACC values are real (nominal - 2 % inflation), as in sensitivity_param_ranges.
    # Additional keys: demand_factor (scales all heat demand), deltaT_sus (permissible lake cooling in K).
    scenarios = {
        "A": {"label": "Base case", "group": "Base",
              "params": {}},
        "B": {"label": "$c_{el}$ = 0.25 €/kWh (ASHP and LSWSHP)", "group": "Techno-economic",
              "params": {"c_el_L": 0.25, "c_el_W": 0.25}},
        "C": {"label": "LSWSHP: $C_{inv}$ −10 %, COP 4.0", "group": "Techno-economic",
              "params": {"CAPEX_Water": 1080.0 * 0.9, "COP_Water": 4.0}},
        "D": {"label": "ASHP: $C_{inv}$ −10 %, COP 3.75", "group": "Techno-economic",
              "params": {"CAPEX_Air": 1901.0 * 0.9, "COP_Air": 3.75}},
        "E": {"label": "$\\Delta T_{sus}$ = 0.5 K", "group": "Lake resource",
              "params": {"deltaT_sus": 0.5}},
        "F": {"label": "$\\Delta T_{sus}$ = 2 K", "group": "Lake resource",
              "params": {"deltaT_sus": 2.0}},
        "G": {"label": "$\\Delta T_{sus}$ = 3 K", "group": "Lake resource",
              "params": {"deltaT_sus": 3.0}},
    }

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
    def project_size_plot_path(self):
        return str(Path(self.results_dir) / "Project_Size_Distribution.pdf")

    @property
    def economic_sensitivity_plot_path(self):
        return str(Path(self.results_dir) / "Economic_Sensitivity.pdf")
    
    @property
    def utilized_mask_cache_path(self):
        suffix = f"_{self.nuts1_filter[0]}" if self.nuts1_filter else ""
        return str(Path(self.processed_dir) / f"utilized_mask_{self.cache_tag}{suffix}.npy")

    @property
    def scenario_csv(self):
        return str(Path(self.processed_dir) / f"economic_scenarios_{self.cache_tag}.csv")

    @property
    def scenario_plot_path(self):
        return str(Path(self.results_dir) / "Economic_Scenarios.pdf")

    @property
    def economic_sensitivity_csv(self):
        return str(Path(self.processed_dir) / f"economic_sensitivity_{self.cache_tag}.csv")