"""Re-create the project size distribution figure from the cached results
(Data/Processed/economic_potential_comparison.csv) without rerunning the full assessment."""
import pandas as pd

from config import Config
from economic_potential import plot_project_size_distribution

if __name__ == "__main__":
    cfg = Config()
    compare_df = pd.read_csv(cfg.economic_compare_csv)
    plot_project_size_distribution(cfg, compare_df)