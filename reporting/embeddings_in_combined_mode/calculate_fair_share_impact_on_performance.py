"""
This script analyzes the relationship between the Fair Share Gap (algorithmic choice)
and the Performance Delta (predictive outcome).
It merges the 'combined' and 'raw-only' execution modes, calculates a normalized
Relative Percentage Improvement to handle bounded/unbounded metrics on the same scale,
generates a four-quadrant scatter plot, calculates the Spearman rank correlation,
and outputs segmented descriptive metrics (both Mean and Median).
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
OUTPUT_SCATTER_RELATIVE = "outputs/performance_delta_scatter.png"
OUTPUT_CSV_RELATIVE = "outputs/performance_correlation_summary.csv"
OUTPUT_SEGMENTED_CSV_RELATIVE = "outputs/segmented_performance_summary.csv"
# ==============================================================================

# Resolve absolute paths
SCRIPT_DIR = Path(__file__).parent.resolve()
INPUT_PATH = SCRIPT_DIR / INPUT_CSV_RELATIVE
OUTPUT_SCATTER = SCRIPT_DIR / OUTPUT_SCATTER_RELATIVE
OUTPUT_CSV = SCRIPT_DIR / OUTPUT_CSV_RELATIVE
OUTPUT_SEGMENTED_CSV = SCRIPT_DIR / OUTPUT_SEGMENTED_CSV_RELATIVE

# Ensure the outputs directory exists
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Map raw algorithm strings to presentation names
ALGO_MAP = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost"}

# Metrics where a lower value indicates better performance (Error metrics)
LOWER_IS_BETTER = ["1-auroc", "log_loss", "rmse", "mae", "mse", "error"]


def prepare_performance_data(df):
    """Merges raw-only and combined modes to calculate Percentage Delta and Fair Share Gap."""
    # Filter for target algorithms
    base_df = df[df["algorithm"].str.lower().isin(ALGO_MAP.keys())].copy()
    base_df["Algorithm"] = base_df["algorithm"].str.lower().map(ALGO_MAP)
    base_df["Task Type"] = base_df["task_type"].str.capitalize()

    # Split into 'combined' and 'raw-only' dataframes
    combined_df = base_df[base_df["mode"].str.lower() == "combined"].copy()
    raw_df = base_df[base_df["mode"].str.lower() == "raw-only"].copy()

    # Calculate Fair Share Gap for the combined mode
    safe_total = combined_df["feat_total_count"].replace(0, np.nan)
    expected_share = (combined_df["pca_n_components"] / safe_total) * 100
    combined_df["Fair Share Gap"] = (
        combined_df["feat_share_embedded_pct"] - expected_share
    )

    # Select columns to merge
    combined_subset = combined_df[
        [
            "dataset",
            "Algorithm",
            "Task Type",
            "primary_metric",
            "eval_primary_value",
            "Fair Share Gap",
        ]
    ]
    raw_subset = raw_df[["dataset", "Algorithm", "eval_primary_value"]]

    # Merge on dataset and algorithm to align the rows
    merged_df = pd.merge(
        combined_subset,
        raw_subset,
        on=["dataset", "Algorithm"],
        suffixes=("_combined", "_raw"),
    )

    # Calculate Relative Percentage Improvement
    def calculate_percentage_delta(row):
        metric = str(row["primary_metric"]).lower()
        baseline = row["eval_primary_value_raw"]
        hybrid = row["eval_primary_value_combined"]

        # Prevent division by zero
        if baseline == 0:
            return 0.0

        if metric in LOWER_IS_BETTER:
            # Lower is better: Positive % means hybrid error is lower than baseline
            return ((baseline - hybrid) / abs(baseline)) * 100
        else:
            # Higher is better: Positive % means hybrid score is higher than baseline
            return ((hybrid - baseline) / abs(baseline)) * 100

    merged_df["Relative Improvement %"] = merged_df.apply(
        calculate_percentage_delta, axis=1
    )

    # Drop any extreme statistical anomalies (e.g., negative R2 scores jumping passing through 0)
    # Clip extreme percentage values to keep the plot readable
    merged_df["Plot Improvement %"] = merged_df["Relative Improvement %"].clip(
        lower=-100, upper=200
    )

    return merged_df


def generate_performance_scatter_plot(df, save_path):
    """Generates a four-quadrant scatter plot mapping algorithmic choice to predictive outcome."""
    plt.figure(figsize=(11, 8))

    task_order = ["Binary", "Multiclass", "Regression"]
    algo_order = ["CatBoost", "LightGBM", "XGBoost"]

    ax = sns.scatterplot(
        data=df,
        x="Fair Share Gap",
        y="Plot Improvement %",
        hue="Task Type",
        style="Algorithm",
        hue_order=task_order,
        style_order=algo_order,
        palette="muted",
        alpha=0.85,
        s=80,
    )

    # Add Quadrant Baselines
    # Horizontal line: 0% Improvement (Above = Hybrid Wins, Below = Raw Wins)
    ax.axhline(0, color="black", linestyle="-", linewidth=1.2, alpha=0.7, zorder=0)
    # Vertical line: 0 Fair Share Gap (Right = Favors Embeddings, Left = Favors Raw)
    ax.axvline(0, color="black", linestyle="-", linewidth=1.2, alpha=0.7, zorder=0)

    # Add quadrant annotations to guide the reader
    font_kws = dict(color="gray", style="italic", weight="bold", alpha=0.5, fontsize=10)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    ax.text(
        x_max * 0.98,
        y_max * 0.95,
        "Embeddings Favored &\nPerformance Improved",
        ha="right",
        va="top",
        **font_kws,
    )
    ax.text(
        x_min * 0.98,
        y_max * 0.95,
        "Raw Favored &\nPerformance Improved",
        ha="left",
        va="top",
        **font_kws,
    )

    # Adjust y-axis slightly for better vertical padding
    plt.ylim(y_min - 5, y_max + 10)

    plt.title("")
    plt.ylabel(
        "Relative Performance Improvement (%)\n[> 0 indicates Hybrid Outperformed Raw Baseline]",
        fontsize=11,
    )
    plt.xlabel(
        "Fair Share Gap (% Points)\n[> 0 indicates Over-weighting Embeddings]",
        fontsize=11,
    )

    # Extract legend handles and space cleanly
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
    print(f"Generated Performance Delta Scatter plot: {save_path}")


def generate_correlation_csv(df, save_path):
    """Calculates Spearman correlation between Fair Share Gap and Relative Improvement %."""
    summary_data = []

    # Global correlation
    overall_corr, overall_p = spearmanr(
        df["Fair Share Gap"], df["Relative Improvement %"]
    )

    summary_data.append(
        {
            "Algorithm": "Overall (All Frameworks)",
            "Correlation (Spearman)": f"{overall_corr:.3f}",
            "P-Value": f"{overall_p:.4f}",
        }
    )

    # Per-framework correlation
    for algo in ALGO_MAP.values():
        algo_df = df[df["Algorithm"] == algo]
        if algo_df.empty:
            continue

        corr, p = spearmanr(
            algo_df["Fair Share Gap"], algo_df["Relative Improvement %"]
        )

        summary_data.append(
            {
                "Algorithm": algo,
                "Correlation (Spearman)": f"{corr:.3f}",
                "P-Value": f"{p:.4f}",
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(save_path, index=False)
    print(f"Generated performance correlation summary table: {save_path}")


def generate_segmented_metrics_csv(df, save_path):
    """Segments data into behavioral buckets and calculates mean and median performance improvement."""

    # Define the behavioral bins for the Fair Share Gap
    bins = [-np.inf, 0, 30, np.inf]
    labels = [
        "Negative Gap (< 0%) [Ignore Embeddings]",
        "Moderate Positive Gap (0% to 30%) [Sweet Spot]",
        "Extreme Positive Gap (> 30%) [Regularization Shield]",
    ]

    # Segment the data
    df_segmented = df.copy()
    df_segmented["Behavioral Segment"] = pd.cut(
        df_segmented["Fair Share Gap"], bins=bins, labels=labels, right=False
    )

    # Group and aggregate for both mean and median
    summary = (
        df_segmented.groupby("Behavioral Segment", observed=False)
        .agg(
            Number_of_Runs=("Relative Improvement %", "count"),
            Mean_Relative_Improvement_Pct=("Relative Improvement %", "mean"),
            Median_Relative_Improvement_Pct=("Relative Improvement %", "median"),
        )
        .reset_index()
    )

    # Format the percentages for display/CSV
    summary["Mean_Relative_Improvement_Pct"] = summary[
        "Mean_Relative_Improvement_Pct"
    ].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
    summary["Median_Relative_Improvement_Pct"] = summary[
        "Median_Relative_Improvement_Pct"
    ].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")

    summary.to_csv(save_path, index=False)

    print("\n=======================================================================")
    print("                 SEGMENTED DESCRIPTIVE METRICS                         ")
    print("=======================================================================")
    print(summary.to_string(index=False))
    print("=======================================================================\n")
    print(f"Generated segmented metrics table: {save_path}")


def main():
    if not INPUT_PATH.exists():
        print(f"Error: Target file not found at {INPUT_PATH}.")
        return

    df = pd.read_csv(INPUT_PATH)
    processed_df = prepare_performance_data(df)

    if processed_df.empty:
        print("Warning: No matching records found after filtering.")
        return

    generate_performance_scatter_plot(processed_df, OUTPUT_SCATTER)
    generate_correlation_csv(processed_df, OUTPUT_CSV)
    generate_segmented_metrics_csv(processed_df, OUTPUT_SEGMENTED_CSV)


if __name__ == "__main__":
    main()
