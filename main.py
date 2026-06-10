from config import Config
import theoretical_potential
import technical_extractable_potential
import economic_potential
from combined_plot import plot_combined_map


def main():
    cfg = Config()

    # --- Step 1: Theoretical Potential ---
    print("=" * 60)
    print("STEP 1: Theoretical Potential")
    print("=" * 60)
    lakes = theoretical_potential.calculate(cfg)
    theoretical_potential.plot(cfg, lakes)
    plot_combined_map(cfg, lakes, economic_potential.load_comparison_data(cfg))


    # --- Step 2: Technical-Extractable Potential ---
    print("=" * 60)
    print("STEP 2: Technical-Extractable Potential")
    print("=" * 60)
    technical_extractable_potential.calculate(cfg)

    if cfg.run_technical_sensitivity:
        df_sens = technical_extractable_potential.sensitivity_analysis(cfg)
        print(df_sens.to_string(index=False))
        technical_extractable_potential.plot_sensitivity(cfg, df_sens)

    # --- Step 3: Economic Potential ---
    print("=" * 60)
    print("STEP 3: Economic Potential")
    print("=" * 60)
    economic_potential.calculate(cfg)

    plot_combined_map(cfg, lakes, economic_potential.load_comparison_data(cfg))

    print("=" * 60)
    print("Done.")




if __name__ == "__main__":
    main()