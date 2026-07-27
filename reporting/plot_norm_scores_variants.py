"""
Script Description:
Generates a 2x3 faceted box plot overlaid with jittered data points using the
Median-Based Normalized Scores. Visualizes the top 3 individual variants for
each specific strata (facet) rather than overarching paradigms.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Ensures file paths are strictly relative to this script's physical location, not the CWD
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(".").resolve()  # Fallback if executed in a Jupyter notebook

INPUT_CSV = SCRIPT_DIR / "summary_matrix_refactored.csv"
OUTPUT_PLOT = SCRIPT_DIR / "outputs" / "norm_scores_boxplot_variants.png"

# Metrics where a lower value indicates better performance (Error metrics)
LOWER_IS_BETTER = ["1-auroc", "log_loss", "rmse", "mae", "mse", "error"]

# Map internal variant names to formal LaTeX names for the chart labels
LABEL_MAP = {
    "baseline_tabpfn": r"$\text{TabPFN}_{\text{baseline}}$",
    "combined_lightgbm": r"$\text{LightGBM}_{\text{raw+embed}}$",
    "combined_catboost": r"$\text{CatBoost}_{\text{raw+embed}}$",
    "combined_xgboost": r"$\text{XGBoost}_{\text{raw+embed}}$",
    "baseline_lightgbm": r"$\text{LightGBM}_{\text{baseline}}$",
    "baseline_catboost": r"$\text{CatBoost}_{\text{baseline}}$",
    "baseline_xgboost": r"$\text{XGBoost}_{\text{baseline}}$",
    "embed-only_lightgbm": r"$\text{LightGBM}_{\text{embed-only}}$",
    "embed-only_catboost": r"$\text{CatBoost}_{\text{embed-only}}$",
    "embed-only_xgboost": r"$\text{XGBoost}_{\text{embed-only}}$",
}

# Fixed colors for all 10 variants
VARIANT_COLORS = {
    # Distinct colors for the 6 variants that can appear in the top 3
    "baseline_tabpfn": "#4c72b0",
    "combined_lightgbm": "#55a868",
    "combined_catboost": "#c44e52",
    "baseline_lightgbm": "#d7bb55",
    "embed-only_catboost": "#8172b3",
    # Gray for the variants that never make the top 3
    "baseline_catboost": "#b0b0b0",
    "combined_xgboost": "#b0b0b0",
    "baseline_xgboost": "#b0b0b0",
    "embed-only_xgboost": "#b0b0b0",
    "embed-only_lightgbm": "#b0b0b0",
}
# ==============================================================================


def determine_variant(row):
    """Constructs the variant identifier matching the leaderboard naming convention."""
    mode = str(row.get("mode", "")).lower().strip()
    algo = str(row.get("algorithm", "")).lower().strip()

    if "tabpfn" in algo or "tabpfn" in mode:
        return "baseline_tabpfn"
    elif "combined" in mode:
        return f"combined_{algo}"
    elif "embed-only" in mode or "embed_only" in mode:
        return f"embed-only_{algo}"
    elif "raw-only" in mode or "raw_only" in mode or "raw" == mode:
        return f"baseline_{algo}"

    # Fallback if 'Strategy' column exists
    return row.get("Strategy", "unknown")


def get_top_n_variants(facet_df, n=3):
    """Calculates mean normalized scores and returns the filtered DF and ordered top N variants."""
    means = facet_df.groupby("Variant")["Normalized_Score"].mean().reset_index()
    top_variants = (
        means.sort_values(by="Normalized_Score", ascending=False)
        .head(n)["Variant"]
        .tolist()
    )

    filtered_df = facet_df[facet_df["Variant"].isin(top_variants)].copy()
    filtered_df["Variant"] = pd.Categorical(
        filtered_df["Variant"], categories=top_variants, ordered=True
    )
    return filtered_df, top_variants


def main():
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

    if not INPUT_CSV.exists():
        print(f"Error: Input file not found at {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)

    # 1. Map Variants & Clean Task Types
    df["Variant"] = df.apply(determine_variant, axis=1)
    df = df[df["Variant"] != "unknown"].copy()
    df["Task Type"] = df["task_type"].str.lower().str.strip().str.capitalize()

    # Classify Dataset Scale (< 10k is Small, >= 10k is Medium)
    if "dataset_samples_count" in df.columns:
        df["dataset_samples_count"] = pd.to_numeric(
            df["dataset_samples_count"], errors="coerce"
        )
        df["Scale"] = np.where(
            df["dataset_samples_count"] <= 10000,
            "Small",
            np.where(df["dataset_samples_count"] > 10000, "Medium", "Unknown"),
        )
    else:
        df["Scale"] = "Unknown"

    df = df[df["Task Type"].isin(["Binary", "Multiclass", "Regression"])].copy()

    # 2. Compute Median-Based Normalized Scores per Dataset across ALL variants
    dataset_scores = []

    for (dataset, metric), sub_df in df.groupby(["dataset", "primary_metric"]):
        is_lower = str(metric).lower() in LOWER_IS_BETTER
        task = sub_df["Task Type"].iloc[0]
        scale = sub_df["Scale"].iloc[0]

        raw_vals = sub_df["eval_primary_value"]

        # Calculate Topline (Best) and Baseline (Median) across all 10 models
        topline = raw_vals.min() if is_lower else raw_vals.max()
        baseline = raw_vals.median()
        denominator = max(abs(baseline - topline), 1e-5)

        if is_lower:
            sub_df["Norm_Score"] = ((baseline - raw_vals) / denominator).clip(0.0, 1.0)
        else:
            sub_df["Norm_Score"] = ((raw_vals - baseline) / denominator).clip(0.0, 1.0)

        # Collect scores for all available variants on this dataset
        for _, row in sub_df.iterrows():
            dataset_scores.append(
                {
                    "dataset": dataset,
                    "Task Type": task,
                    "Scale": scale,
                    "Variant": row["Variant"],
                    "Normalized_Score": row["Norm_Score"],
                }
            )

    plot_df = pd.DataFrame(dataset_scores)

    # 3. Define the 6 Facets for the 2x3 Grid
    facets = [
        {"title": "Overall", "data": plot_df},
        {"title": "Small-Scale", "data": plot_df[plot_df["Scale"] == "Small"]},
        {"title": "Medium-Scale", "data": plot_df[plot_df["Scale"] == "Medium"]},
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

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=True)
    axes_flat = axes.flatten()

    print("Generating chart layers...")
    for i, facet in enumerate(facets):
        ax = axes_flat[i]

        # Filter to top 3 variants dynamically for this specific facet
        facet_data, top_variants = get_top_n_variants(facet["data"], n=3)
        distinct_datasets = facet_data["dataset"].nunique()

        sns.boxplot(
            data=facet_data,
            x="Variant",
            y="Normalized_Score",
            hue="Variant",
            legend=False,
            order=top_variants,
            ax=ax,
            width=0.5,
            boxprops=dict(alpha=0.3),
            palette=VARIANT_COLORS,
            showfliers=False,
            dodge=False,
            showmeans=True,
            meanprops=dict(
                marker="D",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=6,
            ),
        )

        sns.stripplot(
            data=facet_data,
            x="Variant",
            y="Normalized_Score",
            hue="Variant",
            legend=False,
            order=top_variants,
            ax=ax,
            palette=VARIANT_COLORS,
            size=6,
            jitter=0.2,
            alpha=0.8,
            linewidth=0.5,
            edgecolor="auto",
            dodge=False,
        )

        ax.set_title(
            f"{facet['title']}\n(n = {distinct_datasets} datasets)",
            pad=12,
            weight="bold",
        )
        ax.set_xlabel("")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("")

        # Apply specific X-axis labels for this subplot
        ax.set_xticks(range(len(top_variants)))
        clean_ticks = [LABEL_MAP.get(tick, tick) for tick in top_variants]
        ax.set_xticklabels(clean_ticks, rotation=15, ha="right")

    plt.tight_layout(pad=2.0, h_pad=4.0, rect=[0.03, 0, 1, 1])

    fig.supylabel(
        "Normalized Score",
        x=0.015,
        fontweight="bold",
        fontsize=13,
    )

    # 4. Save Output
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Success! Clean 2x3 faceted box plot successfully saved to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
