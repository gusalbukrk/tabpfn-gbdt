"""
Description:
This script processes the benchmark summary matrix to generate a "Fair Share"
feature importance analysis. It calculates the Expected Share of embeddings based
on physical column counts, computes the Fair Share Gap (Actual - Expected), and
generates a 2x3 scenario grid and a dataset-level diverging horizontal cascade.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent

# Input (Using the original refactored matrix since SHAP is skipped)
INPUT_CSV_PATH = SCRIPT_DIR / "../summary_matrix_refactored.csv"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# Explicit Outputs for Fair Share Analysis
OUTPUT_GRID_CHART = OUTPUT_DIR / "fair_share_2x3_grid.png"
OUTPUT_DATASET_CHART = OUTPUT_DIR / "fair_share_dataset_cascade.png"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "fair_share_scenario_summary.csv"
OUTPUT_DATASET_CSV = OUTPUT_DIR / "fair_share_dataset_summary.csv"

# ==============================================================================
# VISUAL CONFIGURATION
# ==============================================================================
# Academic Color Palette (Diverging for Positive/Negative Gaps)
COLOR_POSITIVE = "#4C72B0"  # Blue for over-indexing on embeddings
COLOR_NEGATIVE = "#55A868"
COLOR_NEUTRAL = "#404040"  # Dark gray for the baseline


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading data from {INPUT_CSV_PATH}...")
    df_full = pd.read_csv(INPUT_CSV_PATH)

    # Filter strictly for Hybrid executions
    df = df_full[df_full["mode"] == "combined"].copy()

    # Clean framework names and define scale
    df["algorithm"] = df["algorithm"].str.capitalize()
    df["task_type"] = df["task_type"].str.capitalize()
    df["size_scale"] = np.where(df["dataset_samples_count"] <= 10000, "Small", "Medium")

    # ==============================================================================
    # 1. CALCULATE FAIR SHARE METRICS
    # ==============================================================================
    # Expected Share: (PCA Components / Total Features) * 100
    safe_total = df["feat_total_count"].replace(0, np.nan)
    df["Expected Share %"] = (df["pca_n_components"] / safe_total) * 100

    # Actual Share: Raw metric recorded
    df["Actual Share %"] = df["feat_share_embedded_pct"]

    # Fair Share Gap: Actual - Expected (Percentage Points)
    df["Fair Share Gap"] = df["Actual Share %"] - df["Expected Share %"]

    # Fair Share Multiplier: Actual / Expected
    safe_expected = df["Expected Share %"].replace(0, np.nan)
    df["Fair Share Multiplier"] = df["Actual Share %"] / safe_expected

    print(f"Fair Share metrics calculated for {len(df)} runs.")

    # ==============================================================================
    # 2. GENERATE 2x3 GRID CHART (GAP BAR CHARTS)
    # ==============================================================================
    print("Generating 2x3 Fair Share Gap Grid...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))

    def plot_gap_bar(ax, data_slice, title):
        if data_slice.empty:
            ax.set_title(f"{title}\n(No Data)", pad=15)
            ax.axis("off")
            return

        # Group by algorithm and calculate mean gap
        grouped = data_slice.groupby("algorithm")[["Fair Share Gap"]].mean()

        # Calculate macro-average for the "Avg." bar
        slice_mean = data_slice[["Fair Share Gap"]].mean()
        baseline_df = pd.DataFrame([slice_mean], index=["Avg."])

        # Combine Avg. with the algorithms
        combined_grouped = pd.concat([baseline_df, grouped])

        # Determine bar colors (Blue for >0, Red for <0)
        colors = [
            COLOR_POSITIVE if val >= 0 else COLOR_NEGATIVE
            for val in combined_grouped["Fair Share Gap"]
        ]

        # Plot bars
        bars = ax.bar(
            combined_grouped.index,
            combined_grouped["Fair Share Gap"],
            color=colors,
            width=0.6,
        )

        # Draw neutral baseline at Y=0
        ax.axhline(0, color=COLOR_NEUTRAL, linestyle="--", linewidth=1.5, zorder=0)

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_ylabel("Fair Share Gap (% Points)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)

        # Add dynamic padding to Y-axis so labels never get cut off
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(min(y_min * 1.2, -10), max(y_max * 1.2, 10))

        # Add text labels with exact pixel padding
        for bar in bars:
            val = bar.get_height()
            sign = "+" if val > 0 else ""

            # Use 'offset points' to ensure a strict 5-pixel gap regardless of chart scale
            y_pad = 5 if val > 0 else -5
            va = "bottom" if val > 0 else "top"

            ax.annotate(
                f"{sign}{val:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, y_pad),
                textcoords="offset points",
                ha="center",
                va=va,
                color="black",
                fontweight="bold",
                fontsize=10,
            )

    # Row 1: Overall and by Scale
    plot_gap_bar(axes[0, 0], df, "1. Overall")
    plot_gap_bar(
        axes[0, 1], df[df["size_scale"] == "Small"], "2. Small Datasets (<= 10K)"
    )
    plot_gap_bar(
        axes[0, 2], df[df["size_scale"] == "Medium"], "3. Medium Datasets (> 10K)"
    )

    # Row 2: By Task Type
    plot_gap_bar(
        axes[1, 0], df[df["task_type"] == "Binary"], "4. Binary Classification"
    )
    plot_gap_bar(
        axes[1, 1], df[df["task_type"] == "Multiclass"], "5. Multiclass Classification"
    )
    plot_gap_bar(axes[1, 2], df[df["task_type"] == "Regression"], "6. Regression")

    plt.tight_layout(w_pad=4.0, h_pad=5.0)
    plt.savefig(OUTPUT_GRID_CHART, dpi=300, bbox_inches="tight")
    plt.close()

    # ==============================================================================
    # 3. GENERATE DATASET-LEVEL DIVERGING CASCADE
    # ==============================================================================
    print("Generating Dataset Diverging Cascade...")

    # Average across the 3 algorithms per dataset
    df_dataset = (
        df.groupby("dataset")[
            [
                "Fair Share Gap",
                "Fair Share Multiplier",
                "Actual Share %",
                "Expected Share %",
            ]
        ]
        .mean()
        .reset_index()
    )

    # Merge metadata
    meta = df[
        ["dataset", "task_type", "size_scale", "pca_n_components", "feat_total_count"]
    ].drop_duplicates()
    df_dataset = pd.merge(df_dataset, meta, on="dataset", how="left")

    # Sort by Fair Share Gap for the cascade
    df_dataset = df_dataset.sort_values("Fair Share Gap", ascending=True).reset_index(
        drop=True
    )

    # Export the dataset-level CSV
    df_dataset.to_csv(OUTPUT_DATASET_CSV, index=False)

    # Horizontal Diverging Bar Chart
    fig_h, ax_h = plt.subplots(figsize=(10, 18))

    colors_h = [
        COLOR_POSITIVE if val >= 0 else COLOR_NEGATIVE
        for val in df_dataset["Fair Share Gap"]
    ]

    ax_h.barh(
        df_dataset["dataset"], df_dataset["Fair Share Gap"], color=colors_h, height=0.7
    )
    ax_h.axvline(0, color=COLOR_NEUTRAL, linestyle="-", linewidth=2, zorder=0)

    ax_h.set_title(
        "Dataset-Level Fair Share Gap Cascade", pad=15, fontweight="bold", fontsize=14
    )
    ax_h.set_xlabel(
        "Fair Share Gap (Percentage Points vs. Physical Column Count)", fontsize=12
    )
    ax_h.set_ylabel("Dataset", fontsize=12)

    # Adjust X limits to be symmetrical for visual balance
    max_abs_gap = df_dataset["Fair Share Gap"].abs().max() + 5
    ax_h.set_xlim(-max_abs_gap, max_abs_gap)

    plt.tight_layout()
    plt.savefig(OUTPUT_DATASET_CHART, dpi=300, bbox_inches="tight")
    plt.close()

    # ==============================================================================
    # 4. GENERATE SUMMARY STATISTICS TABLE
    # ==============================================================================
    print("Generating Scenario Summary CSV...")

    slices = {
        "1. Overall": df,
        "2. Scale: Small": df[df["size_scale"] == "Small"],
        "3. Scale: Medium": df[df["size_scale"] == "Medium"],
        "4. Task: Binary": df[df["task_type"] == "Binary"],
        "5. Task: Multiclass": df[df["task_type"] == "Multiclass"],
        "6. Task: Regression": df[df["task_type"] == "Regression"],
    }

    stats_records = []
    for slice_name, df_slice in slices.items():
        if df_slice.empty:
            continue

        gap_col = df_slice["Fair Share Gap"]
        mult_col = df_slice["Fair Share Multiplier"]

        stats_records.append(
            {
                "Scenario": slice_name,
                "Count (Runs)": len(gap_col),
                "Mean Expected Share (%)": round(
                    df_slice["Expected Share %"].mean(), 2
                ),
                "Mean Actual Share (%)": round(df_slice["Actual Share %"].mean(), 2),
                "Mean Fair Share Gap": round(gap_col.mean(), 2),
                "Median Fair Share Gap": round(gap_col.median(), 2),
                "Mean Multiplier (x)": round(mult_col.mean(), 2),
                "Median Multiplier (x)": round(mult_col.median(), 2),
            }
        )

    df_stats = pd.DataFrame(stats_records)
    df_stats.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print("\n" + "=" * 70)
    print(f" [SUCCESS] FAIR SHARE SUITE DEPLOYED IN: {OUTPUT_DIR.resolve()}")
    print("=" * 70)
    print(f"1. {OUTPUT_GRID_CHART.name}")
    print(f"2. {OUTPUT_DATASET_CHART.name}")
    print(f"3. {OUTPUT_SUMMARY_CSV.name}")
    print(f"4. {OUTPUT_DATASET_CSV.name}\n")


if __name__ == "__main__":
    main()
