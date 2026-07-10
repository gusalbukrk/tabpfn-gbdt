"""
==============================================================================
Script: generate_faceted_boxplot.py

Description:
Generates a 2x3 faceted box plot overlaid with jittered data points using the
Median-Based Normalized Scores.

Facets:
- Row 1: Overall, Small Scale (<10k), Medium Scale (>=10k)
- Row 2: Binary Tasks, Multiclass Tasks, Regression Tasks
Y-Axis: Normalized Score (Centralized for the entire figure).
X-Axis: Architectural Strategies (Labels on all rows).
==============================================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_CSV = "summary_matrix_refactored.csv"
OUTPUT_PLOT = "outputs/norm_scores_boxplot.png"

# Metrics where a lower value indicates better performance (Error metrics)
LOWER_IS_BETTER = ["1-auroc", "log_loss", "rmse", "mae", "mse", "error"]

# Order of strategies on the X-axis for consistent visual tracking
STRATEGY_ORDER = ["tabpfn_baseline", "combined_*", "raw-only_*", "embed-only_*"]

# Map internal strategy names to formal LaTeX names for the chart labels
LABEL_MAP = {
    "tabpfn_baseline": r"$\text{TabPFN}_{\text{baseline}}$",
    "combined_*": r"$\text{Hybrid}_{\text{raw+embed}}$",
    "raw-only_*": r"$\text{GBDT}_{\text{baseline}}$",
    "embed-only_*": r"$\text{Hybrid}_{\text{embed-only}}$",
}
# ==============================================================================


def determine_paradigm(row):
    """Identifies the architectural paradigm based on mode/algorithm columns."""
    mode = str(row.get("mode", "")).lower().strip()
    algo = str(row.get("algorithm", "")).lower().strip()

    if "tabpfn" in algo or "tabpfn" in mode:
        return "tabpfn_baseline"
    elif "combined" in mode:
        return "combined_*"
    elif "embed-only" in mode or "embed_only" in mode:
        return "embed-only_*"
    elif "raw-only" in mode or "raw_only" in mode or "raw" == mode:
        return "raw-only_*"
    return None


def main():
    # Set high-quality aesthetic theme
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 11,
        }
    )

    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        print(f"Error: Input file not found at {INPUT_CSV}")
        return

    df = pd.read_csv(input_path)

    # 1. Map Paradigms & Clean Task Types
    df["Paradigm"] = df.apply(determine_paradigm, axis=1)
    df = df[df["Paradigm"].notna()].copy()
    df["Task Type"] = df["task_type"].str.lower().str.strip().str.capitalize()

    # Classify Dataset Scale (< 10k is Small, >= 10k is Medium)
    if "dataset_samples_count" in df.columns:
        df["dataset_samples_count"] = pd.to_numeric(
            df["dataset_samples_count"], errors="coerce"
        )
        df["Scale"] = np.where(
            df["dataset_samples_count"] < 10000,
            "Small",
            np.where(df["dataset_samples_count"] >= 10000, "Medium", "Unknown"),
        )
    else:
        df["Scale"] = "Unknown"

    # Filter strictly to the three core thesis tasks
    df = df[df["Task Type"].isin(["Binary", "Multiclass", "Regression"])].copy()

    # 2. Compute Median-Based Normalized Scores per Dataset
    dataset_ceilings = []

    for (dataset, metric), sub_df in df.groupby(["dataset", "primary_metric"]):
        is_lower = str(metric).lower() in LOWER_IS_BETTER
        task = sub_df["Task Type"].iloc[0]
        scale = sub_df["Scale"].iloc[0]

        raw_vals = sub_df["eval_primary_value"]

        # Calculate Topline (Best) and Baseline (Median)
        topline = raw_vals.min() if is_lower else raw_vals.max()
        baseline = raw_vals.median()

        # Denominator clipping to 1e-5 to prevent division by zero
        denominator = max(abs(baseline - topline), 1e-5)

        if is_lower:
            # For error metrics, distance from baseline (lower is better)
            sub_df["Norm_Score"] = ((baseline - raw_vals) / denominator).clip(0.0, 1.0)
        else:
            # For accuracy metrics, distance from baseline (higher is better)
            sub_df["Norm_Score"] = ((raw_vals - baseline) / denominator).clip(0.0, 1.0)

        # Extract the Ceiling (Best Score) per Paradigm for this Dataset
        for paradigm in STRATEGY_ORDER:
            p_df = sub_df[sub_df["Paradigm"] == paradigm]
            if p_df.empty:
                continue

            best_score = p_df["Norm_Score"].max()
            dataset_ceilings.append(
                {
                    "dataset": dataset,
                    "Task Type": task,
                    "Scale": scale,
                    "Paradigm": paradigm,
                    "Normalized_Score": best_score,
                }
            )

    plot_df = pd.DataFrame(dataset_ceilings)

    # 3. Define the 6 Facets for the 2x3 Grid
    facets = [
        {"title": "Overall Benchmark", "data": plot_df},
        {"title": "Small Scale", "data": plot_df[plot_df["Scale"] == "Small"]},
        {"title": "Medium Scale", "data": plot_df[plot_df["Scale"] == "Medium"]},
        {"title": "Binary Tasks", "data": plot_df[plot_df["Task Type"] == "Binary"]},
        {
            "title": "Multiclass Tasks",
            "data": plot_df[plot_df["Task Type"] == "Multiclass"],
        },
        {
            "title": "Regression Tasks",
            "data": plot_df[plot_df["Task Type"] == "Regression"],
        },
    ]

    # Build the Faceted Plot Structure (2 Rows, 3 Columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=True)
    axes_flat = axes.flatten()

    colors = {
        "tabpfn_baseline": "#4c72b0",
        "combined_*": "#55a868",
        "raw-only_*": "#c44e52",
        "embed-only_*": "#8172b3",
    }

    print("Generating chart layers...")
    for i, facet in enumerate(facets):
        ax = axes_flat[i]
        facet_data = facet["data"]
        distinct_datasets = facet_data["dataset"].nunique()

        sns.boxplot(
            data=facet_data,
            x="Paradigm",
            y="Normalized_Score",
            hue="Paradigm",
            legend=False,
            order=STRATEGY_ORDER,
            ax=ax,
            width=0.5,
            boxprops=dict(alpha=0.3),
            palette=colors,
            showfliers=False,
        )

        sns.stripplot(
            data=facet_data,
            x="Paradigm",
            y="Normalized_Score",
            hue="Paradigm",
            legend=False,
            order=STRATEGY_ORDER,
            ax=ax,
            palette=colors,
            size=6,
            jitter=0.2,
            alpha=0.8,
            linewidth=0.5,
            edgecolor="auto",
        )

        ax.set_title(
            f"{facet['title']}\n(n = {distinct_datasets} datasets)",
            pad=12,
            weight="bold",
        )
        ax.set_xlabel("")
        ax.set_ylim(-0.05, 1.05)

        # Explicitly remove the individual Y-axis labels
        ax.set_ylabel("")

        # Map internal strategy names to formal LaTeX labels for ALL rows
        ax.set_xticks(range(len(STRATEGY_ORDER)))
        raw_ticks = STRATEGY_ORDER
        clean_ticks = [LABEL_MAP.get(tick, tick) for tick in raw_ticks]
        ax.set_xticklabels(clean_ticks, rotation=15, ha="right")

    # Global adjustments: Add h_pad so the new top-row labels don't hit bottom-row titles
    plt.tight_layout(pad=2.0, h_pad=4.0, rect=[0.03, 0, 1, 1])

    # Apply a single, centered Y-axis label to the entire figure
    fig.supylabel(
        "Normalized Score\n(Worst 0.0 $\\rightarrow$ 1.0 Best)",
        x=0.015,
        fontweight="bold",
        fontsize=13,
    )

    # 4. Save Output
    output_path = Path(OUTPUT_PLOT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Success! Clean 2x3 faceted box plot successfully saved to: {output_path}")


if __name__ == "__main__":
    main()
