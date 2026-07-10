"""
This script processes the previously extracted summary matrix to analyze feature
reliance in 'combined' mode. It calculates the relative weight placed on TabPFN-3
embeddings versus raw features using total share percentages, generating 100%
stacked horizontal bar charts with an empty title and an external legend.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==============================================================================
# CONFIGURATION: DECLARE FILE PATHS HERE (Relative to script location)
# ==============================================================================
INPUT_CSV_RELATIVE = "summary.csv"
OUTPUT_OVERALL_RELATIVE = "outputs/graph_c_overall.png"
OUTPUT_XGBOOST_RELATIVE = "outputs/graph_c_xgboost.png"
OUTPUT_LIGHTGBM_RELATIVE = "outputs/graph_c_lightgbm.png"
OUTPUT_CATBOOST_RELATIVE = "outputs/graph_c_catboost.png"
# ==============================================================================

# Resolve absolute paths relative to the directory containing this script
SCRIPT_DIR = Path(__file__).parent.resolve()
INPUT_PATH = SCRIPT_DIR / INPUT_CSV_RELATIVE

OUTPUT_PATHS = {
    "overall": SCRIPT_DIR / OUTPUT_OVERALL_RELATIVE,
    "xgboost": SCRIPT_DIR / OUTPUT_XGBOOST_RELATIVE,
    "lightgbm": SCRIPT_DIR / OUTPUT_LIGHTGBM_RELATIVE,
    "catboost": SCRIPT_DIR / OUTPUT_CATBOOST_RELATIVE,
}


def prepare_stacked_data(df):
    """Computes relative proportions for a true 100% stacked bar chart using total shares."""
    total_share = df["feat_share_embedded_pct"] + df["feat_share_raw_pct"]
    total_share = total_share.replace(0, 1)  # Safeguard against division by zero

    df["Embedded Share %"] = (df["feat_share_embedded_pct"] / total_share) * 100
    df["Raw Share %"] = (df["feat_share_raw_pct"] / total_share) * 100

    # Sort ascending so the highest embedding reliance appears at the top of the horizontal plot
    return df.sort_values(by="Embedded Share %", ascending=True)


def plot_stacked_reliance(df, save_path):
    """Generates and saves a clean 100% stacked horizontal bar chart with external legend and no title."""
    # Ensure parent output directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Adjust plot height dynamically based on dataset volume to maintain label legibility
    fig_height = max(6, len(df) * 0.25)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # Plot 100% stacked horizontal bars
    df.plot(
        x="dataset",
        y=["Embedded Share %", "Raw Share %"],
        kind="barh",
        stacked=True,
        ax=ax,
        color=["#2b5c8f", "#d95f02"],
        width=0.8,
    )

    # Layout tuning and aesthetics
    ax.set_title("", fontsize=14, pad=15)
    ax.set_xlabel("Relative Predictive Share Contribution (%)", fontsize=11)
    ax.set_ylabel("Dataset", fontsize=11)
    ax.set_xlim(0, 100)

    # Position legend cleanly below the X-axis to completely eliminate data obfuscation
    ax.legend(
        ["TabPFN-3 Embedded Share", "Original Raw Feature Share"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=2,
        frameon=True,
    )

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    # bbox_inches='tight' guarantees the external legend isn't clipped in the saved image
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated footprint chart: {save_path}")


def main():
    # Verify input existence
    if not INPUT_PATH.exists():
        print(
            f"Error: Target summary file not found at {INPUT_PATH}. Run the extraction script first."
        )
        return

    # Load file data
    df = pd.read_csv(INPUT_PATH)

    # Filter strictly for rows executing in combined mode
    combined_df = df[df["mode"].str.lower() == "combined"].copy()

    if combined_df.empty:
        print("Warning: No matching records found with 'mode == combined'.")
        return

    # 1. Macro Profile View (Aggregating framework variations per dataset)
    overall_grouped = (
        combined_df.groupby("dataset")[
            ["feat_share_embedded_pct", "feat_share_raw_pct"]
        ]
        .mean()
        .reset_index()
    )
    overall_ready = prepare_stacked_data(overall_grouped)
    plot_stacked_reliance(overall_ready, OUTPUT_PATHS["overall"])

    # 2. Individual Framework Breakdown Processing
    framework_map = {
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
    }

    for algo_key, algo_name in framework_map.items():
        algo_df = combined_df[combined_df["algorithm"].str.lower() == algo_key].copy()

        if algo_df.empty:
            print(f"Skipping {algo_name}: No corresponding rows matched in the matrix.")
            continue

        algo_ready = prepare_stacked_data(algo_df)
        plot_stacked_reliance(algo_ready, OUTPUT_PATHS[algo_key])


if __name__ == "__main__":
    main()
