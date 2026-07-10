"""
This script processes the project summary matrix to compare the global feature reliance
of XGBoost, LightGBM, and CatBoost in 'combined' mode. It fixes deprecation warnings
by explicitly mapping the categorical variable to both x and hue while silencing the
redundant legend. It outputs a violin plot, a box plot with explicit text labels displaying
the exact median values inside each box, and a statistical summary CSV table.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==============================================================================
# CONFIGURATION: DECLARE FILE PATHS HERE (Relative to script location)
# ==============================================================================
INPUT_CSV_RELATIVE = "summary.csv"
OUTPUT_VIOLIN_RELATIVE = "outputs/algo_comparison_violin.png"
OUTPUT_BOX_RELATIVE = "outputs/algo_comparison_box.png"
OUTPUT_CSV_RELATIVE = "outputs/algo_comparison_summary.csv"
# ==============================================================================

# Resolve absolute paths relative to the directory containing this script
SCRIPT_DIR = Path(__file__).parent.resolve()
INPUT_PATH = SCRIPT_DIR / INPUT_CSV_RELATIVE
OUTPUT_VIOLIN = SCRIPT_DIR / OUTPUT_VIOLIN_RELATIVE
OUTPUT_BOX = SCRIPT_DIR / OUTPUT_BOX_RELATIVE
OUTPUT_CSV = SCRIPT_DIR / OUTPUT_CSV_RELATIVE

# Ensure the outputs directory exists
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Map raw algorithm strings to presentation names
ALGO_MAP = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost"}


def prepare_data(df):
    """Filters data and calculates the normalized Embedded Share %."""
    combined_df = df[df["mode"].str.lower() == "combined"].copy()

    # Filter only for the main GBDT algorithms
    combined_df = combined_df[
        combined_df["algorithm"].str.lower().isin(ALGO_MAP.keys())
    ]

    # Standardize algorithm names
    combined_df["Algorithm"] = combined_df["algorithm"].str.lower().map(ALGO_MAP)

    # Calculate shares
    total_share = (
        combined_df["feat_share_embedded_pct"] + combined_df["feat_share_raw_pct"]
    )
    total_share = total_share.replace(0, 1)  # Safeguard against division by zero

    combined_df["Embedded Share %"] = (
        combined_df["feat_share_embedded_pct"] / total_share
    ) * 100

    return combined_df


def generate_violin_plot(df, save_path):
    """Generates a standalone clean violin plot to show global distribution."""
    plt.figure(figsize=(10, 6))

    # Order algorithms by median embedded share for a cleaner visual flow
    order = (
        df.groupby("Algorithm")["Embedded Share %"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    # Explicitly assigning 'hue' to match 'x' to prevent future deprecation warnings
    sns.violinplot(
        data=df,
        x="Algorithm",
        y="Embedded Share %",
        order=order,
        hue="Algorithm",
        legend=False,
        palette="muted",
        inner=None,
        cut=0,
    )

    plt.title("")
    plt.ylabel("TabPFN-3 Embedded Share (%)", fontsize=11)
    plt.xlabel("Framework", fontsize=11)
    plt.ylim(0, 100)

    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Generated global distribution violin plot: {save_path}")


def generate_box_plot(df, save_path):
    """Generates a standalone clean box plot displaying localized numerical medians inside the boxes."""
    plt.figure(figsize=(10, 6))

    # Order algorithms by median embedded share consistently
    grouped = df.groupby("Algorithm")["Embedded Share %"].median()
    order = grouped.sort_values(ascending=False).index

    # Explicitly assigning 'hue' to match 'x' to prevent future deprecation warnings
    ax = sns.boxplot(
        data=df,
        x="Algorithm",
        y="Embedded Share %",
        order=order,
        hue="Algorithm",
        legend=False,
        palette="muted",
        width=0.5,
    )

    # Inject calculated medians directly onto the canvas chart overlay
    for i, algo in enumerate(order):
        median_val = grouped[algo]
        ax.text(
            i,
            median_val,
            f"{median_val:.1f}%",
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
    plt.ylabel("TabPFN-3 Embedded Share (%)", fontsize=11)
    plt.xlabel("Framework", fontsize=11)
    plt.ylim(0, 100)

    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Generated global distribution box plot with median labels: {save_path}")


def generate_summary_csv(df, save_path):
    """Calculates Mean ± SD and Median, formatting it for paper citations."""
    summary_data = []

    for algo in ALGO_MAP.values():
        algo_df = df[df["Algorithm"] == algo]
        if algo_df.empty:
            continue

        mean_val = algo_df["Embedded Share %"].mean()
        std_val = algo_df["Embedded Share %"].std()
        median_val = algo_df["Embedded Share %"].median()

        summary_data.append(
            {
                "Algorithm": algo,
                "Mean ± SD": f"{mean_val:.2f} ± {std_val:.2f}",
                "Median": f"{median_val:.2f}",
            }
        )

    summary_df = pd.DataFrame(summary_data)

    # Sort by Median descending to match the plots
    summary_df = summary_df.sort_values(by="Median", ascending=False).reset_index(
        drop=True
    )

    summary_df.to_csv(save_path, index=False)
    print(f"Generated statistical summary table: {save_path}")


def main():
    if not INPUT_PATH.exists():
        print(f"Error: Target file not found at {INPUT_PATH}.")
        return

    df = pd.read_csv(INPUT_PATH)
    processed_df = prepare_data(df)

    if processed_df.empty:
        print("Warning: No matching records found after filtering.")
        return

    generate_violin_plot(processed_df, OUTPUT_VIOLIN)
    generate_box_plot(processed_df, OUTPUT_BOX)
    generate_summary_csv(processed_df, OUTPUT_CSV)


if __name__ == "__main__":
    main()
