"""
This script performs a task-stratified fair share analysis on GBDT feature reliance.
It groups the data by 'task_type' and plots the Fair Share Gap and Multiplier.
Consistent with previous charts, it annotates the exact median value directly
inside each grouped box for immediate readability. It also includes a diagonal
scatter plot to visually test the "Volume Hypothesis" (Expected vs Actual Share).
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
OUTPUT_GAP_RELATIVE = "outputs/task_stratification_fair_share_gap.png"
OUTPUT_MULTIPLIER_RELATIVE = "outputs/task_stratification_fair_share_multiplier.png"
OUTPUT_SCATTER_RELATIVE = "outputs/task_stratification_volume_scatter.png"
OUTPUT_CSV_RELATIVE = "outputs/task_stratification_summary.csv"
# ==============================================================================

# Resolve absolute paths
SCRIPT_DIR = Path(__file__).parent.resolve()
INPUT_PATH = SCRIPT_DIR / INPUT_CSV_RELATIVE
OUTPUT_GAP = SCRIPT_DIR / OUTPUT_GAP_RELATIVE
OUTPUT_MULTIPLIER = SCRIPT_DIR / OUTPUT_MULTIPLIER_RELATIVE
OUTPUT_SCATTER = SCRIPT_DIR / OUTPUT_SCATTER_RELATIVE
OUTPUT_CSV = SCRIPT_DIR / OUTPUT_CSV_RELATIVE

# Ensure the outputs directory exists
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Map raw algorithm strings to presentation names
ALGO_MAP = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost"}


def prepare_stratified_data(df):
    """Calculates Fair Share metrics and cleans the task_type column for plotting."""
    combined_df = df[df["mode"].str.lower() == "combined"].copy()

    # Filter for the main GBDT algorithms
    combined_df = combined_df[
        combined_df["algorithm"].str.lower().isin(ALGO_MAP.keys())
    ].copy()
    combined_df["Algorithm"] = combined_df["algorithm"].str.lower().map(ALGO_MAP)

    # Standardize Task Type names (e.g., 'binary' -> 'Binary')
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


def generate_stratified_gap_plot(df, save_path):
    """Generates a grouped box plot for Fair Share Gap annotated with internal medians."""
    plt.figure(figsize=(12, 6))

    task_order = ["Binary", "Multiclass", "Regression"]
    algo_order = ["CatBoost", "LightGBM", "XGBoost"]

    ax = sns.boxplot(
        data=df,
        x="Task Type",
        y="Fair Share Gap",
        hue="Algorithm",
        order=task_order,
        hue_order=algo_order,
        palette="muted",
    )

    # Add the neutral baseline
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, zorder=0)

    # Inject calculated medians inside the grouped boxes
    # Seaborn dodges grouped boxes based on the width (default 0.8). For 3 hues, offsets are roughly -0.266, 0, +0.266
    offsets = [-0.266, 0, 0.266]

    for task_idx, task in enumerate(task_order):
        for algo_idx, algo in enumerate(algo_order):
            subset = df[(df["Task Type"] == task) & (df["Algorithm"] == algo)]
            if subset.empty:
                continue

            median_val = subset["Fair Share Gap"].median()
            sign = "+" if median_val > 0 else ""

            ax.text(
                task_idx + offsets[algo_idx],
                median_val,
                f"{sign}{median_val:.1f}%",
                ha="center",
                va="center",
                color="black",
                weight="bold",
                fontsize=8.5,
                bbox=dict(
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="none",
                    boxstyle="round,pad=0.15",
                ),
            )

    plt.title("")
    plt.ylabel("Fair Share Gap (% Points)\n[> 0 indicates Over-weighting]", fontsize=11)
    plt.xlabel("Task Type", fontsize=11)

    # Position legend outside the plot to avoid obscuring data
    ax.legend(
        title="Framework", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True
    )

    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Generated Stratified Fair Share Gap plot: {save_path}")


def generate_stratified_multiplier_plot(df, save_path):
    """Generates a log-scale grouped box plot for Fair Share Multiplier annotated with internal medians."""
    plt.figure(figsize=(12, 6))

    task_order = ["Binary", "Multiclass", "Regression"]
    algo_order = ["CatBoost", "LightGBM", "XGBoost"]

    # Clip absolute zero multipliers to a small positive number
    df["Plot Multiplier"] = df["Fair Share Multiplier"].clip(lower=0.01)

    ax = sns.boxplot(
        data=df,
        x="Task Type",
        y="Plot Multiplier",
        hue="Algorithm",
        order=task_order,
        hue_order=algo_order,
        palette="muted",
    )

    # Add the neutral baseline
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, zorder=0)
    ax.set_yscale("log")

    # Inject calculated medians inside the grouped boxes
    offsets = [-0.266, 0, 0.266]

    for task_idx, task in enumerate(task_order):
        for algo_idx, algo in enumerate(algo_order):
            subset = df[(df["Task Type"] == task) & (df["Algorithm"] == algo)]
            if subset.empty:
                continue

            median_val = subset["Fair Share Multiplier"].median()

            ax.text(
                task_idx + offsets[algo_idx],
                median_val,
                f"{median_val:.2f}x",
                ha="center",
                va="center",
                color="black",
                weight="bold",
                fontsize=8.5,
                bbox=dict(
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="none",
                    boxstyle="round,pad=0.15",
                ),
            )

    plt.title("")
    plt.ylabel(
        "Fair Share Multiplier (Log Scale)\n[> 1.0x indicates Over-weighting]",
        fontsize=11,
    )
    plt.xlabel("Task Type", fontsize=11)

    ax.legend(
        title="Framework", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True
    )

    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Generated Stratified Fair Share Multiplier plot: {save_path}")


def generate_volume_scatter_plot(df, save_path):
    """Generates a scatter plot testing the volume hypothesis (Expected vs Actual Share)."""
    plt.figure(figsize=(10, 8))

    task_order = ["Binary", "Multiclass", "Regression"]
    algo_order = ["CatBoost", "LightGBM", "XGBoost"]

    ax = sns.scatterplot(
        data=df,
        x="Expected Share %",
        y="Actual Share %",
        hue="Task Type",
        style="Algorithm",
        hue_order=task_order,
        style_order=algo_order,
        palette="muted",
        alpha=0.85,
        s=80,  # Marker size
    )

    # Add the 45-degree diagonal line representing perfect volume proportionality (Gap = 0)
    ax.plot(
        [-5, 105],
        [-5, 105],
        color="red",
        linestyle="--",
        linewidth=1.5,
        zorder=0,
        label="Perfect Proportionality",
    )

    plt.title("")
    plt.ylabel("Actual Predictive Share (%)", fontsize=11)
    plt.xlabel("Expected Share (Physical Column %)", fontsize=11)

    # Lock axes strictly from 0 to 100 with a tiny padding for edge points
    plt.xlim(-2, 102)
    plt.ylim(-2, 102)

    # Extract legend handles to keep it clean, placing it outside the plot
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(
    #     handles=handles,
    #     labels=labels,
    #     bbox_to_anchor=(1.02, 1),
    #     loc="upper left",
    #     frameon=True,
    # )

    # Extract legend handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Create an invisible proxy artist to act as a spacer
    blank_handle = plt.Line2D([0], [0], color="none")

    # Inject a blank line right before "Algorithm"
    if "Algorithm" in labels:
        idx = labels.index("Algorithm")
        labels.insert(idx, "")
        handles.insert(idx, blank_handle)

    # Inject a blank line right before "Perfect Proportionality"
    if "Perfect Proportionality" in labels:
        idx = labels.index("Perfect Proportionality")
        labels.insert(idx, "")
        handles.insert(idx, blank_handle)

    # Render the adjusted legend outside the plot
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
    print(f"Generated Volume Scatter plot: {save_path}")


def generate_summary_csv(df, save_path):
    """Calculates stats grouped by both Task Type and Algorithm."""
    summary_data = []

    task_order = ["Binary", "Multiclass", "Regression"]
    algo_order = ["CatBoost", "LightGBM", "XGBoost"]

    for task in task_order:
        task_df = df[df["Task Type"] == task]

        # Calculate an 'Overall' row for each task type before algorithmic breakdown
        if not task_df.empty:
            gap_mean = task_df["Fair Share Gap"].mean()
            gap_std = task_df["Fair Share Gap"].std()
            gap_med = task_df["Fair Share Gap"].median()
            mult_mean = task_df["Fair Share Multiplier"].mean()
            mult_std = task_df["Fair Share Multiplier"].std()
            mult_med = task_df["Fair Share Multiplier"].median()

            summary_data.append(
                {
                    "Task Type": task,
                    "Algorithm": "Overall (All Frameworks)",
                    "Gap (Mean ± SD)": f"{gap_mean:+.2f} ± {gap_std:.2f}",
                    "Gap (Median)": f"{gap_med:+.2f}%",
                    "Multiplier (Mean ± SD)": f"{mult_mean:.2f} ± {mult_std:.2f}",
                    "Multiplier (Median)": f"{mult_med:.2f}x",
                }
            )

        # Calculate breakdown per framework
        for algo in algo_order:
            algo_df = task_df[task_df["Algorithm"] == algo]
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
                    "Task Type": task,
                    "Algorithm": algo,
                    "Gap (Mean ± SD)": f"{gap_mean:+.2f} ± {gap_std:.2f}",
                    "Gap (Median)": f"{gap_med:+.2f}%",
                    "Multiplier (Mean ± SD)": f"{mult_mean:.2f} ± {mult_std:.2f}",
                    "Multiplier (Median)": f"{mult_med:.2f}x",
                }
            )

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(save_path, index=False)
    print(f"Generated task-stratified summary table: {save_path}")


def main():
    if not INPUT_PATH.exists():
        print(f"Error: Target file not found at {INPUT_PATH}.")
        return

    df = pd.read_csv(INPUT_PATH)
    processed_df = prepare_stratified_data(df)

    if processed_df.empty:
        print("Warning: No matching records found after filtering.")
        return

    generate_stratified_gap_plot(processed_df, OUTPUT_GAP)
    generate_stratified_multiplier_plot(processed_df, OUTPUT_MULTIPLIER)
    generate_volume_scatter_plot(processed_df, OUTPUT_SCATTER)
    generate_summary_csv(processed_df, OUTPUT_CSV)


if __name__ == "__main__":
    main()
