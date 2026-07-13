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

#### License

Please add your license here.

---

#### Citation

If you use this tool in academic work, please cite accordingly.
```
