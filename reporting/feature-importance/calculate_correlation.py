"""
Description:
This script analyzes the relationship between the Fair Share Gap (algorithmic choice)
and the Performance Delta against the native TabPFN baseline.
It generates a 4-quadrant scatter plot, segmented descriptive metrics, and an
expanded Spearman correlation table that breaks down the statistical relationship
Overall, by Framework, by Scale, and by Task Type, retaining full p-value precision.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
INPUT_PATH = SCRIPT_DIR / "../summary_matrix_refactored.csv"

OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_SCATTER = OUTPUT_DIR / "tabpfn_correlation_scatter.png"
OUTPUT_CSV = OUTPUT_DIR / "tabpfn_correlation_summary.csv"
OUTPUT_SEGMENTED_CSV = OUTPUT_DIR / "tabpfn_segmented_summary.csv"

# Map raw algorithm strings to presentation names
ALGO_MAP = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost"}


def prepare_performance_data(df):
    """Calculates Fair Share Gap and the precise % improvement over TabPFN."""
    combined_df = df[df["mode"].str.lower() == "combined"].copy()

    # Filter for target algorithms and clean names
    combined_df = combined_df[
        combined_df["algorithm"].str.lower().isin(ALGO_MAP.keys())
    ].copy()
    combined_df["Algorithm"] = combined_df["algorithm"].str.lower().map(ALGO_MAP)
    combined_df["Task Type"] = combined_df["task_type"].str.capitalize()

    # Define Size Scale for scenario slicing
    combined_df["Size Scale"] = np.where(
        combined_df["dataset_samples_count"] <= 10000, "Small", "Medium"
    )

    # 1. Calculate Fair Share Gap
    safe_total = combined_df["feat_total_count"].replace(0, np.nan)
    combined_df["Expected Share %"] = (
        combined_df["pca_n_components"] / safe_total
    ) * 100
    combined_df["Fair Share Gap"] = (
        combined_df["feat_share_embedded_pct"] - combined_df["Expected Share %"]
    )

    # 2. Calculate Precise Improvement over Native TabPFN
    def calculate_tabpfn_improvement(row):
        hybrid = row["eval_primary_value"]
        delta = row["eval_primary_delta"]

        # If baseline was not logged (e.g., TabPFN failed on this dataset), skip
        if pd.isna(delta):
            return np.nan

        # Reconstruct native TabPFN absolute score (delta = hybrid - baseline -> baseline = hybrid - delta)
        baseline = hybrid - delta

        if baseline == 0:
            return 0.0

        # All primary metrics (1-AUROC, LogLoss, RMSE) are minimization targets.
        # Positive % means the hybrid model's error is lower than TabPFN's error.
        return ((baseline - hybrid) / abs(baseline)) * 100

    combined_df["Relative Improvement %"] = combined_df.apply(
        calculate_tabpfn_improvement, axis=1
    )

    # Drop rows where TabPFN baseline was missing
    combined_df = combined_df.dropna(subset=["Relative Improvement %"])

    # Clip extreme percentage values to keep the plot readable, preserving distribution integrity
    combined_df["Plot Improvement %"] = combined_df["Relative Improvement %"].clip(
        lower=-100, upper=200
    )

    return combined_df


def generate_performance_scatter_plot(df, save_path):
    """Generates a 4-quadrant scatter plot mapping algorithmic gap to outcome vs TabPFN."""
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
    ax.axhline(0, color="black", linestyle="-", linewidth=1.2, alpha=0.7, zorder=0)
    ax.axvline(0, color="black", linestyle="-", linewidth=1.2, alpha=0.7, zorder=0)

    # Quadrant Annotations
    font_kws = dict(color="gray", style="italic", weight="bold", alpha=0.5, fontsize=10)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    ax.text(
        x_max * 0.98,
        y_max * 0.95,
        "Embeddings Favored &\nBeat TabPFN",
        ha="right",
        va="top",
        **font_kws,
    )
    ax.text(
        x_min * 0.98,
        y_max * 0.95,
        "Raw Features Favored &\nBeat TabPFN",
        ha="left",
        va="top",
        **font_kws,
    )
    ax.text(
        x_max * 0.98,
        y_min + (y_max - y_min) * 0.05,
        "Embeddings Favored &\nLost to TabPFN",
        ha="right",
        va="bottom",
        **font_kws,
    )
    ax.text(
        x_min * 0.98,
        y_min + (y_max - y_min) * 0.05,
        "Raw Features Favored &\nLost to TabPFN",
        ha="left",
        va="bottom",
        **font_kws,
    )

    plt.ylim(y_min - 5, y_max + 10)
    plt.ylabel(
        "Relative Performance Improvement over native TabPFN (%)\n[> 0 indicates Hybrid Won]",
        fontsize=11,
    )
    plt.xlabel(
        "Fair Share Gap (% Points)\n[> 0 indicates Over-weighting Embeddings]",
        fontsize=11,
    )

    # Clean Legend Formatting
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
    print(f"Generated Scatter Plot: {save_path.name}")


def generate_correlation_csv(df, save_path):
    """Calculates Spearman correlation across various analytical scenarios."""

    # Define the exact data slices for scenario analysis
    slices = {
        "1. Overall (All Runs)": df,
        "2. Framework: CatBoost": df[df["Algorithm"] == "CatBoost"],
        "3. Framework: LightGBM": df[df["Algorithm"] == "LightGBM"],
        "4. Framework: XGBoost": df[df["Algorithm"] == "XGBoost"],
        "5. Scale: Small (<= 10K)": df[df["Size Scale"] == "Small"],
        "6. Scale: Medium (> 10K)": df[df["Size Scale"] == "Medium"],
        "7. Task: Binary": df[df["Task Type"] == "Binary"],
        "8. Task: Multiclass": df[df["Task Type"] == "Multiclass"],
        "9. Task: Regression": df[df["Task Type"] == "Regression"],
    }

    summary_data = []

    for scenario_name, slice_df in slices.items():
        # Ensure we have enough data to calculate a valid correlation
        if len(slice_df) < 3:
            continue

        corr, p = spearmanr(
            slice_df["Fair Share Gap"], slice_df["Relative Improvement %"]
        )

        summary_data.append(
            {
                "Analytical Scenario": scenario_name,
                "Count (N)": len(slice_df),
                "Correlation (Spearman)": f"{corr:.3f}" if pd.notna(corr) else "N/A",
                # Removed the format cap, passing the raw float value for scientific notation rendering
                "P-Value": f"{p:.6f}" if pd.notna(p) else "N/A",
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(save_path, index=False)

    print("\n=======================================================================")
    print("                SPEARMAN CORRELATION BY SCENARIO                       ")
    print("=======================================================================")
    print(summary_df.to_string(index=False))
    print("=======================================================================\n")
    print(f"Generated Correlation Table: {save_path.name}")


def generate_segmented_metrics_csv(df, save_path):
    """Segments algorithmic choices into 3 behavioral buckets to expose synergy."""
    bins = [-np.inf, 0, 30, np.inf]
    labels = [
        "Negative Gap (< 0%) [Ignored Embeddings]",
        "Moderate Gap (0% to 30%) [Balanced Synergy]",
        "Extreme Gap (> 30%) [Embeddings as Shield]",
    ]

    df_segmented = df.copy()
    df_segmented["Behavioral Segment"] = pd.cut(
        df_segmented["Fair Share Gap"], bins=bins, labels=labels, right=False
    )

    summary = (
        df_segmented.groupby("Behavioral Segment", observed=False)
        .agg(
            Number_of_Runs=("Relative Improvement %", "count"),
            Mean_Improvement_vs_TabPFN=("Relative Improvement %", "mean"),
            Median_Improvement_vs_TabPFN=("Relative Improvement %", "median"),
        )
        .reset_index()
    )

    summary["Mean_Improvement_vs_TabPFN"] = summary["Mean_Improvement_vs_TabPFN"].apply(
        lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
    )
    summary["Median_Improvement_vs_TabPFN"] = summary[
        "Median_Improvement_vs_TabPFN"
    ].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")

    summary.to_csv(save_path, index=False)
    print(f"Generated Segmented Metrics Table: {save_path.name}")


def main():
    if not INPUT_PATH.exists():
        print(f"Error: Target file not found at {INPUT_PATH.resolve()}")
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
