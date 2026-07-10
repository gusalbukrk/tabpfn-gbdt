"""
This script performs a fair share analysis on the feature reliance of GBDTs.
It calculates the Expected Share based on the total physical column count,
and compares it to the Actual Predictive Share using two metrics:
1. Fair Share Gap (Actual % - Expected %)
2. Fair Share Multiplier (Actual % / Expected %)
Outputs include two box plots with neutral baseline anchors and a summary CSV.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# ==============================================================================
# CONFIGURATION: DECLARE FILE PATHS HERE (Relative to script location)
# ==============================================================================
INPUT_CSV_RELATIVE = "summary.csv"
OUTPUT_GAP_RELATIVE = "outputs/fair_share_gap.png"
OUTPUT_MULTIPLIER_RELATIVE = "outputs/fair_share_multiplier.png"
OUTPUT_CSV_RELATIVE = "outputs/fair_share_summary.csv"
# ==============================================================================

# Resolve absolute paths
SCRIPT_DIR = Path(__file__).parent.resolve()
INPUT_PATH = SCRIPT_DIR / INPUT_CSV_RELATIVE
OUTPUT_GAP = SCRIPT_DIR / OUTPUT_GAP_RELATIVE
OUTPUT_MULTIPLIER = SCRIPT_DIR / OUTPUT_MULTIPLIER_RELATIVE
OUTPUT_CSV = SCRIPT_DIR / OUTPUT_CSV_RELATIVE

# Ensure the outputs directory exists
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Map raw algorithm strings to presentation names
ALGO_MAP = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost"}


def prepare_fair_share_data(df):
    """Calculates Expected Share, Actual Share, Gap, and Multiplier for the combined mode."""
    combined_df = df[df["mode"].str.lower() == "combined"].copy()

    # Filter for the main GBDT algorithms and apply naming
    combined_df = combined_df[
        combined_df["algorithm"].str.lower().isin(ALGO_MAP.keys())
    ].copy()
    combined_df["Algorithm"] = combined_df["algorithm"].str.lower().map(ALGO_MAP)

    # 1. Calculate Expected Share (%) based on physical column pools (Fair Share)
    safe_total = combined_df["feat_total_count"].replace(0, np.nan)
    combined_df["Expected Share %"] = (
        combined_df["pca_n_components"] / safe_total
    ) * 100

    # 2. Calculate Actual Share (%) based strictly on the recorded metric
    combined_df["Actual Share %"] = combined_df["feat_share_embedded_pct"]

    # 3. Calculate Fair Share Metrics
    # Approach A: Gap (+/- percentage points)
    combined_df["Fair Share Gap"] = (
        combined_df["Actual Share %"] - combined_df["Expected Share %"]
    )

    # Approach B: Multiplier (Factor)
    safe_expected = combined_df["Expected Share %"].replace(0, np.nan)
    combined_df["Fair Share Multiplier"] = combined_df["Actual Share %"] / safe_expected

    # 4. Generate the 'Overall' stratification group
    overall_df = (
        combined_df.groupby("dataset")
        .agg({"Fair Share Gap": "mean", "Fair Share Multiplier": "mean"})
        .reset_index()
    )
    overall_df["Algorithm"] = "Overall"

    # Combine Overall with the specific algorithms for unified plotting
    plot_cols = ["dataset", "Algorithm", "Fair Share Gap", "Fair Share Multiplier"]
    final_df = pd.concat(
        [overall_df[plot_cols], combined_df[plot_cols]], ignore_index=True
    )

    return final_df


def generate_gap_plot(df, save_path):
    """Generates a box plot for Fair Share Gap with a neutral anchor at 0."""
    plt.figure(figsize=(10, 6))

    order = ["Overall", "CatBoost", "LightGBM", "XGBoost"]
    grouped = df.groupby("Algorithm")["Fair Share Gap"].median()

    ax = sns.boxplot(
        data=df,
        x="Algorithm",
        y="Fair Share Gap",
        order=order,
        hue="Algorithm",
        legend=False,
        palette="muted",
        width=0.5,
    )

    # Add the neutral baseline (0 Gap = perfectly proportional weighting)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, zorder=0)

    # Inject calculated medians
    for i, algo in enumerate(order):
        median_val = grouped[algo]
        sign = "+" if median_val > 0 else ""
        ax.text(
            i,
            median_val,
            f"{sign}{median_val:.1f}%",
            ha="center",
            va="center",
            color="black",
            weight="bold",
            fontsize=10,
            bbox=dict(
                facecolor="white",
                alpha=0.85,
                edgecolor="none",
                boxstyle="round,pad=0.2",
            ),
        )

    plt.title("")
    plt.ylabel("Fair Share Gap (% Points)\n[> 0 indicates Over-weighting]", fontsize=11)
    plt.xlabel("Framework", fontsize=11)

    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Generated Fair Share Gap plot: {save_path}")


def generate_multiplier_plot(df, save_path):
    """Generates a log-scale box plot for Fair Share Multiplier with a neutral anchor at 1x."""
    plt.figure(figsize=(10, 6))

    order = ["Overall", "CatBoost", "LightGBM", "XGBoost"]
    grouped = df.groupby("Algorithm")["Fair Share Multiplier"].median()

    # Clip absolute zero multipliers to a small positive number to prevent log-scale collapse
    df["Plot Multiplier"] = df["Fair Share Multiplier"].clip(lower=0.01)

    ax = sns.boxplot(
        data=df,
        x="Algorithm",
        y="Plot Multiplier",
        order=order,
        hue="Algorithm",
        legend=False,
        palette="muted",
        width=0.5,
    )

    # Add the neutral baseline (1.0x = perfectly proportional weighting)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, zorder=0)
    ax.set_yscale("log")

    # Inject calculated medians
    for i, algo in enumerate(order):
        median_val = grouped[algo]
        ax.text(
            i,
            median_val,
            f"{median_val:.2f}x",
            ha="center",
            va="center",
            color="black",
            weight="bold",
            fontsize=10,
            bbox=dict(
                facecolor="white",
                alpha=0.85,
                edgecolor="none",
                boxstyle="round,pad=0.2",
            ),
        )

    plt.title("")
    plt.ylabel(
        "Fair Share Multiplier (Log Scale)\n[> 1.0x indicates Over-weighting]",
        fontsize=11,
    )
    plt.xlabel("Framework", fontsize=11)

    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Generated Fair Share Multiplier plot: {save_path}")


def generate_summary_csv(df, save_path):
    """Calculates Mean ± SD and Median for both fair share metrics."""
    summary_data = []
    order = ["Overall", "CatBoost", "LightGBM", "XGBoost"]

    for algo in order:
        algo_df = df[df["Algorithm"] == algo]
        if algo_df.empty:
            continue

        gap_mean = algo_df["Fair Share Gap"].mean()
        gap_std = algo_df["Fair Share Gap"].std()
        gap_med = algo_df["Fair Share Gap"].median()

        mult_mean = algo_df["Fair Share Multiplier"].mean()
        mult_std = algo_df["Fair Share Multiplier"].std()
        mult_med = algo_df["Fair Share Multiplier"].median()

        summary_data.append(
            {
                "Algorithm": algo,
                "Gap (Median)": f"{gap_med:+.2f}%",
                "Gap (Mean ± SD)": f"{gap_mean:+.2f} ± {gap_std:.2f}",
                "Multiplier (Median)": f"{mult_med:.2f}x",
                "Multiplier (Mean ± SD)": f"{mult_mean:.2f} ± {mult_std:.2f}",
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(save_path, index=False)
    print(f"Generated statistical summary table: {save_path}")


def main():
    if not INPUT_PATH.exists():
        print(f"Error: Target file not found at {INPUT_PATH}.")
        return

    df = pd.read_csv(INPUT_PATH)
    processed_df = prepare_fair_share_data(df)

    if processed_df.empty:
        print("Warning: No matching records found after filtering.")
        return

    generate_gap_plot(processed_df, OUTPUT_GAP)
    generate_multiplier_plot(processed_df, OUTPUT_MULTIPLIER)
    generate_summary_csv(processed_df, OUTPUT_CSV)


if __name__ == "__main__":
    main()
