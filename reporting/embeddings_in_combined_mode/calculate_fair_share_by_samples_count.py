"""
This script analyzes the relationship between dataset sample scale (row count)
and GBDT feature reliance (Fair Share Gap and Multiplier).
It generates logarithmic scatter plots and calculates the Spearman correlation
coefficients to statistically prove if sample size dictates embedding reliance.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr

# ==============================================================================
# CONFIGURATION: DECLARE FILE PATHS HERE (Relative to script location)
# ==============================================================================
INPUT_CSV_RELATIVE = "summary.csv"
OUTPUT_GAP_SCATTER_RELATIVE = "outputs/sample_scale_gap_scatter.png"
OUTPUT_MULTIPLIER_SCATTER_RELATIVE = "outputs/sample_scale_multiplier_scatter.png"
OUTPUT_CSV_RELATIVE = "outputs/sample_scale_correlation_summary.csv"
# ==============================================================================

# Resolve absolute paths
SCRIPT_DIR = Path(__file__).parent.resolve()
INPUT_PATH = SCRIPT_DIR / INPUT_CSV_RELATIVE
OUTPUT_GAP_SCATTER = SCRIPT_DIR / OUTPUT_GAP_SCATTER_RELATIVE
OUTPUT_MULTIPLIER_SCATTER = SCRIPT_DIR / OUTPUT_MULTIPLIER_SCATTER_RELATIVE
OUTPUT_CSV = SCRIPT_DIR / OUTPUT_CSV_RELATIVE

# Ensure the outputs directory exists
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Map raw algorithm strings to presentation names
ALGO_MAP = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost"}


def prepare_sample_scale_data(df):
    """Calculates Fair Share metrics and extracts the sample count."""
    combined_df = df[df["mode"].str.lower() == "combined"].copy()

    # Filter for the main GBDT algorithms
    combined_df = combined_df[
        combined_df["algorithm"].str.lower().isin(ALGO_MAP.keys())
    ].copy()
    combined_df["Algorithm"] = combined_df["algorithm"].str.lower().map(ALGO_MAP)
    combined_df["Task Type"] = combined_df["task_type"].str.capitalize()

    # 1. Expected Share (%)
    safe_total = combined_df["feat_total_count"].replace(0, np.nan)
    combined_df["Expected Share %"] = (
        combined_df["pca_n_components"] / safe_total
    ) * 100

    # 2. Actual Share (%)
    combined_df["Actual Share %"] = combined_df["feat_share_embedded_pct"]

    # 3. Fair Share Gap
    combined_df["Fair Share Gap"] = (
        combined_df["Actual Share %"] - combined_df["Expected Share %"]
    )

    # 4. Fair Share Multiplier
    safe_expected = combined_df["Expected Share %"].replace(0, np.nan)
    combined_df["Fair Share Multiplier"] = combined_df["Actual Share %"] / safe_expected

    return combined_df


def generate_gap_scatter_plot(df, save_path):
    """Generates a scatter plot of Sample Count vs. Fair Share Gap."""
    plt.figure(figsize=(11, 7))

    task_order = ["Binary", "Multiclass", "Regression"]
    algo_order = ["CatBoost", "LightGBM", "XGBoost"]

    ax = sns.scatterplot(
        data=df,
        x="dataset_samples_count",
        y="Fair Share Gap",
        hue="Task Type",
        style="Algorithm",
        hue_order=task_order,
        style_order=algo_order,
        palette="muted",
        alpha=0.85,
        s=80,
    )

    # Add the neutral baseline (0 Gap)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, zorder=0)

    # Set X-axis to logarithmic scale for sample counts
    ax.set_xscale("log")

    plt.title("")
    plt.ylabel("Fair Share Gap (% Points)\n[> 0 indicates Over-weighting]", fontsize=11)
    plt.xlabel("Dataset Sample Count (Log Scale)", fontsize=11)

    # Extract legend handles and space them out cleanly
    handles, labels = ax.get_legend_handles_labels()
    blank_handle = plt.Line2D([0], [0], color="none")

    if "Algorithm" in labels:
        idx = labels.index("Algorithm")
        labels.insert(idx, "")
        handles.insert(idx, blank_handle)

    ax.legend(
        handles=handles,
        labels=labels,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
    )

    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Generated Sample Scale Gap Scatter plot: {save_path}")


def generate_multiplier_scatter_plot(df, save_path):
    """Generates a scatter plot of Sample Count vs. Fair Share Multiplier (Log-Log scale)."""
    plt.figure(figsize=(11, 7))

    task_order = ["Binary", "Multiclass", "Regression"]
    algo_order = ["CatBoost", "LightGBM", "XGBoost"]

    df["Plot Multiplier"] = df["Fair Share Multiplier"].clip(lower=0.01)

    ax = sns.scatterplot(
        data=df,
        x="dataset_samples_count",
        y="Plot Multiplier",
        hue="Task Type",
        style="Algorithm",
        hue_order=task_order,
        style_order=algo_order,
        palette="muted",
        alpha=0.85,
        s=80,
    )

    # Add the neutral baseline (1.0x Multiplier)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, zorder=0)

    # Set both axes to logarithmic scale
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.title("")
    plt.ylabel(
        "Fair Share Multiplier (Log Scale)\n[> 1.0x indicates Over-weighting]",
        fontsize=11,
    )
    plt.xlabel("Dataset Sample Count (Log Scale)", fontsize=11)

    handles, labels = ax.get_legend_handles_labels()
    blank_handle = plt.Line2D([0], [0], color="none")

    if "Algorithm" in labels:
        idx = labels.index("Algorithm")
        labels.insert(idx, "")
        handles.insert(idx, blank_handle)

    ax.legend(
        handles=handles,
        labels=labels,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
    )

    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Generated Sample Scale Multiplier Scatter plot: {save_path}")


def generate_correlation_csv(df, save_path):
    """Calculates Spearman correlation between sample count and enrichment metrics."""
    summary_data = []

    # Calculate global correlation across all frameworks
    overall_gap_corr, overall_gap_p = spearmanr(
        df["dataset_samples_count"], df["Fair Share Gap"]
    )
    overall_mult_corr, overall_mult_p = spearmanr(
        df["dataset_samples_count"], df["Fair Share Multiplier"]
    )

    summary_data.append(
        {
            "Algorithm": "Overall (All Frameworks)",
            "Gap Correlation (Spearman)": f"{overall_gap_corr:.3f}",
            "Gap P-Value": f"{overall_gap_p:.4f}",
            "Multiplier Correlation (Spearman)": f"{overall_mult_corr:.3f}",
            "Multiplier P-Value": f"{overall_mult_p:.4f}",
        }
    )

    # Calculate correlation strictly per framework
    for algo in ALGO_MAP.values():
        algo_df = df[df["Algorithm"] == algo]
        if algo_df.empty:
            continue

        gap_corr, gap_p = spearmanr(
            algo_df["dataset_samples_count"], algo_df["Fair Share Gap"]
        )
        mult_corr, mult_p = spearmanr(
            algo_df["dataset_samples_count"], algo_df["Fair Share Multiplier"]
        )

        summary_data.append(
            {
                "Algorithm": algo,
                "Gap Correlation (Spearman)": f"{gap_corr:.3f}",
                "Gap P-Value": f"{gap_p:.4f}",
                "Multiplier Correlation (Spearman)": f"{mult_corr:.3f}",
                "Multiplier P-Value": f"{mult_p:.4f}",
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(save_path, index=False)
    print(f"Generated correlation summary table: {save_path}")


def main():
    if not INPUT_PATH.exists():
        print(f"Error: Target file not found at {INPUT_PATH}.")
        return

    df = pd.read_csv(INPUT_PATH)
    processed_df = prepare_sample_scale_data(df)

    if processed_df.empty:
        print("Warning: No matching records found after filtering.")
        return

    generate_gap_scatter_plot(processed_df, OUTPUT_GAP_SCATTER)
    generate_multiplier_scatter_plot(processed_df, OUTPUT_MULTIPLIER_SCATTER)
    generate_correlation_csv(processed_df, OUTPUT_CSV)


if __name__ == "__main__":
    main()
