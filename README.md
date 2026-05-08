### Lake Thermal Energy Potential Assessment

A Python pipeline to estimate the **thermal energy potential of lakes** in Germany across three levels:


1. **Theoretical Potential** – Maximum thermal energy stored in lake volumes.
2. **Technical-Extractable Potential** – Achievable extraction considering heat exchanger design and operational hours.

3. **Economic Potential** – Viability of lake-source heat pumps vs. air-source heat pumps based on an Economic Index (EI).

---

#### Project Structure

```
.
├── main.py                             # Entry point – runs all three steps
├── config.py                           # Central configuration (paths, parameters)
├── theoretical_potential.py            # Step 1: Theoretical potential calculation & plot
├── technical_extractable_potential.py  # Step 2: Technical-extractable potential & sensitivity
├── economic_potential.py               # Step 3: EI raster, per-lake aggregation, maps, sensitivity
├── Data/
│   ├── Raw/                            # Input data (shapefiles, raster)
│   │   ├── HydroLAKES_germany.shp     # Lake polygons (HydroLAKES subset)
│   │   ├── NUTS250_N1.shp             # NUTS-1 boundaries (federal states)
│   │   ├── NUTS250_N3.shp             # NUTS-3 boundaries (districts)
│   │   ├── Germany.shp                # Germany outline
│   │   └── heat_tot_curr_density_germany.tif  # Heat demand raster (MWh/ha)
│   ├── Processed/                      # Intermediate outputs (CSV, NPY, TIF)
│   └── Results/                        # Final plots and figures (PDF, SVG, PNG)
├── requirements.txt
└── README.md
```

---

#### Installation

```bash
# Clone the repository
git clone <repo-url>
cd <repo-folder>

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

#### Input Data

| File | Description | Source |
|------|-------------|--------|
| `HydroLAKES_germany.shp` | Lake polygons with volume (`Vol_total` in km³) | [HydroLAKES](https://www.hydrosheds.org/products/hydrolakes) |
| `NUTS250_N1.shp` | NUTS-1 regions (federal states) | [BKG](https://gdz.bkg.bund.de/) |
| `NUTS250_N3.shp` | NUTS-3 regions (districts) | [BKG](https://gdz.bkg.bund.de/) |
| `Germany.shp` | Germany national boundary | [BKG](https://gdz.bkg.bund.de/) |
| `heat_tot_curr_density_germany.tif` | Heat demand density raster (MWh/ha) | [Hotmaps / Peta](https://www.hotmaps-project.eu/) |

Place all raw data into `Data/Raw/` before running.

---

#### Usage

```bash
python main.py
```

This executes all three assessment steps sequentially. Results are written to `Data/Results/` (plots) and `Data/Processed/` (intermediate data).

##### Caching

Set `use_cache = True` in `config.py` (default) to skip expensive computations on re-runs. Delete files in `Data/Processed/` to force recomputation.

##### Regional Filtering

To restrict analysis to specific federal states, set the NUTS-1 codes in `config.py`:

```python
nuts1_filter = ["DE1"]       # e.g., Baden-Württemberg
nuts1_filter = ["DE1", "DE2"] # multiple states
nuts1_filter = None           # all of Germany (default)
```

---

#### Configuration

All parameters are centralized in `config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `delta_T` | 1.0 K | Temperature difference (theoretical) |
| `deltaT_hex` | 5.0 K | Heat exchanger delta-T (technical) |
| `hours_per_year` | 3500 h | Full-load hours (technical) |
| `COP_Water` | 3.3 | COP of water-source heat pump |
| `COP_Air` | 2.9 | COP of air-source heat pump |
| `c_pipe` | 2500 EUR/m | Pipe network cost |
| `ei_max_dist_m` | 10000 m | Maximum lake-to-demand distance |
| `ei_threshold` | 1.0 | Minimum EI for economic viability |
| `distance_mode` | `"hub"` | Distance aggregation (`"hub"` or `"pixel"`) |
| `hub_grid_m` | 1000 m | Hub grid cell size |
| `n_jobs` | 4 | Parallel workers for per-lake computation |

---

#### Outputs

##### Step 1 – Theoretical Potential

- `Data/Processed/theoretical_potential_by_lake.csv`
- `Data/Results/Theoretical_Potential.pdf` / `.svg`

##### Step 2 – Technical-Extractable Potential

- `Data/Processed/technical_extractable_potential_by_lake.csv`
- `Data/Results/Sensitivity_Full_Load_Hours.pdf` / `.svg`

##### Step 3 – Economic Potential

- `Data/Processed/EI_results.tif` – 4-band GeoTIFF (EI, Lake ID, Distance, Hub ID)
- `Data/Processed/economic_potential_comparison.csv`

- `Data/Results/EI_Scatter_Plot.png`
- `Data/Results/Economic_Potential_Map.pdf` / `.svg`

- `Data/Results/Economic_Sensitivity.pdf` / `.svg`

---

#### Methodology

##### Economic Index (EI)

The EI compares the levelized cost of a decentralized air-source heat pump to a centralized lake-source heat pump connected via district heating:

```
EI = (CAPEX_Air * E_h/h_Air + OpEx_Air - CAPEX_Water * E_h/h_Water - OpEx_Water) / (c_pipe * d)
```


- **EI >= 1**: Lake-source heat pump is economically preferable.
- **EI < 1**: Air-source heat pump is cheaper (pipe cost too high).

---

#### License

Please add your license here.

---

#### Citation

If you use this tool in academic work, please cite accordingly.
```