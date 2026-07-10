"""
Description:
This script processes benchmark data to analyze feature importance in hybrid
models (Raw + Embeddings). It generates a 2x3 scenario grid chart where each
subplot includes an "Overall" baseline bar followed by framework-specific bars,
a dataset-level horizontal stacked bar chart, and two CSV tables summarizing
feature importance statistics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
# Resolve the exact directory where this script is physically located
SCRIPT_DIR = Path(__file__).resolve().parent

# Define input and output paths relative to the script location
INPUT_CSV_PATH = SCRIPT_DIR / "../summary_matrix_refactored.csv"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# Define explicit output file paths so they can be easily modified
OUTPUT_GRID_CHART = OUTPUT_DIR / "2x3_scenario_importance.png"
OUTPUT_DATASET_CHART = OUTPUT_DIR / "dataset_horizontal_importance.png"
OUTPUT_DATASET_CSV = OUTPUT_DIR / "dataset_importance_breakdown.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "summary_statistics.csv"

# ==============================================================================
# VISUAL CONFIGURATION
# ==============================================================================
# Academic Color Palette (Blue for Embeddings, Green for Raw)
COLOR_EMBED = "#4C72B0"
COLOR_RAW = "#55A868"


def main():
    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {INPUT_CSV_PATH}...")
    df_full = pd.read_csv(INPUT_CSV_PATH)

    # Filter strictly for Hybrid_raw+embed (mode == 'combined')
    df = df_full[df_full["mode"] == "combined"].copy()

    # Ensure correct data types
    df["feat_share_embedded_pct"] = df["feat_share_embedded_pct"].astype(float)
    df["feat_share_raw_pct"] = df["feat_share_raw_pct"].astype(float)

    # Apply the exact benchmark size threshold (<= 10000 is Small)
    df["size_scale"] = np.where(df["dataset_samples_count"] <= 10000, "Small", "Medium")

    # Capitalize for cleaner plot labels
    df["algorithm"] = df["algorithm"].str.capitalize()
    df["task_type"] = df["task_type"].str.capitalize()

    print(f"Data filtered: {len(df)} rows for Hybrid configurations.")

    # ==============================================================================
    # 1. GENERATE 2x3 GRID CHART (6 SCENARIOS WITH BASELINE BARS)
    # ==============================================================================
    print("Generating 2x3 Grid Chart...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))

    # Helper function to plot a single stacked bar chart with an Overall baseline
    def plot_stacked_bar(ax, data_slice, group_col, title):
        if data_slice.empty:
            ax.set_title(f"{title}\n(No Data)", pad=15)
            ax.axis("off")
            return

        # Compute the breakdown for the specific group (e.g., by Algorithm)
        grouped = data_slice.groupby(group_col)[
            ["feat_share_embedded_pct", "feat_share_raw_pct"]
        ].mean()

        # Calculate the micro-average across the entire data slice to serve as the baseline
        slice_mean = data_slice[
            ["feat_share_embedded_pct", "feat_share_raw_pct"]
        ].mean()
        baseline_df = pd.DataFrame([slice_mean], index=["Avg."])

        # Prepend the Overall baseline row so it renders as the first column
        combined_grouped = pd.concat([baseline_df, grouped])

        # Plot the 4 columns
        combined_grouped.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=[COLOR_EMBED, COLOR_RAW],
            width=0.6,
            legend=False,
        )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Importance Share (%)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)

        # Add percentage text labels inside the bars
        for c in ax.containers:
            labels = [f"{v.get_height():.1f}%" if v.get_height() > 5 else "" for v in c]
            ax.bar_label(
                c,
                labels=labels,
                label_type="center",
                color="white",
                fontweight="bold",
                fontsize=10,
            )

    # Row 1: Overall and by Dataset Size
    plot_stacked_bar(axes[0, 0], df, "algorithm", "1. Overall")
    plot_stacked_bar(
        axes[0, 1],
        df[df["size_scale"] == "Small"],
        "algorithm",
        "2. Small Datasets (<= 10K)",
    )
    plot_stacked_bar(
        axes[0, 2],
        df[df["size_scale"] == "Medium"],
        "algorithm",
        "3. Medium Datasets (> 10K)",
    )

    # Row 2: By Task Type
    plot_stacked_bar(
        axes[1, 0],
        df[df["task_type"] == "Binary"],
        "algorithm",
        "4. Binary Classification",
    )
    plot_stacked_bar(
        axes[1, 1],
        df[df["task_type"] == "Multiclass"],
        "algorithm",
        "5. Multiclass Classification",
    )
    plot_stacked_bar(
        axes[1, 2], df[df["task_type"] == "Regression"], "algorithm", "6. Regression"
    )

    # Add a single global legend at the top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        ["TabPFN Embeddings", "Raw Features"],
        loc="upper center",
        ncol=2,
        fontsize=12,
        bbox_to_anchor=(0.5, 1.02),
    )

    # Structure layout padding
    plt.tight_layout(rect=[0, 0, 1, 0.95], w_pad=4.0, h_pad=5.0)
    plt.savefig(OUTPUT_GRID_CHART, dpi=300, bbox_inches="tight")
    plt.close()

    # ==============================================================================
    # 2. GENERATE PER-DATASET TABLE & HORIZONTAL STACKED CHART
    # ==============================================================================
    print("Generating Dataset-Level Horizontal Chart and Table...")

    # Aggregate by dataset (average across the 3 algorithms)
    df_dataset = (
        df.groupby("dataset")[["feat_share_embedded_pct", "feat_share_raw_pct"]]
        .mean()
        .reset_index()
    )

    # Merge task type and size metadata for the CSV table
    meta = df[["dataset", "task_type", "size_scale"]].drop_duplicates()
    df_dataset = pd.merge(df_dataset, meta, on="dataset", how="left")

    # Sort primarily by embedding share for a clean visual cascade
    df_dataset = df_dataset.sort_values(
        "feat_share_embedded_pct", ascending=True
    ).reset_index(drop=True)

    # --- Export Dataset Table ---
    df_dataset.to_csv(OUTPUT_DATASET_CSV, index=False)

    # --- Generate Horizontal Chart ---
    fig_h, ax_h = plt.subplots(figsize=(10, 18))

    df_dataset.set_index("dataset")[
        ["feat_share_embedded_pct", "feat_share_raw_pct"]
    ].plot(
        kind="barh", stacked=True, ax=ax_h, color=[COLOR_EMBED, COLOR_RAW], width=0.8
    )

    # Title left blank intentionally, keeping the requested padding
    ax_h.set_title("", pad=15)
    ax_h.set_xlabel("Importance Share (%)", fontsize=12)
    ax_h.set_ylabel("Dataset", fontsize=12)
    ax_h.set_xlim(0, 100)
    ax_h.legend(
        ["TabPFN Embeddings", "Raw Features"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DATASET_CHART, dpi=300, bbox_inches="tight")
    plt.close()

    # ==============================================================================
    # 3. GENERATE SUMMARY STATISTICS TABLE
    # ==============================================================================
    print("Generating Summary Statistics Table...")

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

        embed_col = df_slice["feat_share_embedded_pct"]
        stats_records.append(
            {
                "Scenario": slice_name,
                "Count (Runs)": len(embed_col),
                "Mean Embedding Share (%)": round(embed_col.mean(), 2),
                "Median (%)": round(embed_col.median(), 2),
                "Std Dev": round(embed_col.std(), 2),
                "Min (%)": round(embed_col.min(), 2),
                "Max (%)": round(embed_col.max(), 2),
            }
        )

    df_stats = pd.DataFrame(stats_records)
    df_stats.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print("\n" + "=" * 70)
    print(f" [SUCCESS] ALL ARTIFACTS GENERATED IN: {OUTPUT_DIR.resolve()}")
    print("=" * 70)
    print(f"1. {OUTPUT_GRID_CHART.name}")
    print(f"2. {OUTPUT_DATASET_CHART.name}")
    print(f"3. {OUTPUT_DATASET_CSV.name}")
    print(f"4. {OUTPUT_SUMMARY_CSV.name}\n")


if __name__ == "__main__":
    main()
