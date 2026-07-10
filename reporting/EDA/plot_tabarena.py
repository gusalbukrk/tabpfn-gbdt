# ==============================================================================
# DISCLAIMER: CATEGORICAL PERCENTAGE CALCULATION
# This script calculates categorical percentages based purely on the structural
# features passed to the GBDT models (excluding the target variable). This
# ensures the output strictly mirrors the exact reality of the current pipeline.
#
# For a detailed methodological explanation, refer to:
# NOTE ON CATEGORICAL PERCENTAGES.md
#
# WARNING: If the experiments are ever rerun using the metadata fixes
# recommended in the .md file, it will be necessary to update the mathematical
# logic in this script to accurately reflect those changes.
# ==============================================================================


# ==============================================================================
# Script: plot_tabarena.py
# Description: Generates an academically optimized bubble scatter plot of the
#              TabArena benchmark. Maps dataset size (X) vs. raw feature count (Y)
#              on log scales with an open canvas layout. Task types use redundant
#              encoding. Bubble sizes represent the semantically correct
#              categorical feature percentage (including the target variable).
#              Applies controlled multiplicative jitter to resolve overlaps.
# ==============================================================================

import os
import warnings

# Force a non-interactive backend to completely suppress GUI canvas environment warnings
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# Suppress potential layout or font warnings that clutter console output
warnings.filterwarnings("ignore")

# --- Configuration (Paths relative to this script's location) ---
SUMMARY_MATRIX_REL_PATH = "../../outputs/training-consolidated/summary_matrix.csv"
ARCHIVES_REL_DIR = "../../archives"


def generate_corpus_bubble_chart():
    # --------------------------------------------------------------------------
    # 1. Establish Absolute Paths Based on Script Location
    # --------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    matrix_abs_path = os.path.abspath(os.path.join(script_dir, SUMMARY_MATRIX_REL_PATH))
    archives_abs_dir = os.path.abspath(os.path.join(script_dir, ARCHIVES_REL_DIR))

    if not os.path.exists(matrix_abs_path):
        raise FileNotFoundError(
            f"Summary matrix file not found at: {matrix_abs_path}\n"
            "Please check the SUMMARY_MATRIX_REL_PATH variable at the top of the script."
        )

    # --------------------------------------------------------------------------
    # 2. Load and Deduplicate Summary Matrix Data
    # --------------------------------------------------------------------------
    df_raw = pd.read_csv(matrix_abs_path)
    df_datasets = df_raw.drop_duplicates(subset=["dataset"]).copy()

    # --------------------------------------------------------------------------
    # 3. Dynamically Compute Experimental Categorical % from Local .npz Files
    # --------------------------------------------------------------------------
    categorical_percentages = []

    for idx, row in df_datasets.iterrows():
        dataset_name = str(row["dataset"])
        npz_filename = f"{dataset_name}.npz"
        npz_path = os.path.join(archives_abs_dir, npz_filename)

        if os.path.exists(npz_path):
            try:
                npz_data = np.load(npz_path, allow_pickle=True)

                # 1. Total features exactly as seen in the X matrix
                total_features = int(
                    npz_data["n_features_raw"]
                    if "n_features_raw" in npz_data
                    else npz_data["X_train_raw"].shape[1]
                )

                # 2. Categorical features exactly as extracted by Pandas and passed to the GBDT
                cat_features_count = len(npz_data["cat_features"])

                # 3. Pure structural percentage (Target 'y' is excluded because it is not in X)
                cat_pct = (
                    (cat_features_count / total_features) * 100
                    if total_features > 0
                    else 0.0
                )
                categorical_percentages.append(cat_pct)
            except Exception:
                categorical_percentages.append(0.0)
        else:
            categorical_percentages.append(0.0)

    df_datasets["categorical_pct"] = categorical_percentages

    # --------------------------------------------------------------------------
    # Z-ORDER FIX: Sort DataFrame by bubble size descending.
    # Sorting descending ensures large bubbles are drawn first (underneath)
    # and small bubbles are drawn last (on top).
    # --------------------------------------------------------------------------
    df_datasets = df_datasets.sort_values(by="categorical_pct", ascending=False)

    # --------------------------------------------------------------------------
    # VISUAL JITTER FIX: Apply a controlled multiplicative jitter to resolve
    # overlap ghosting lines for datasets with near-identical dimensions.
    # Seed is hardcoded to keep the output completely deterministic.
    # --------------------------------------------------------------------------
    np.random.seed(42)
    jitter_factor_x = np.random.uniform(0.97, 1.03, len(df_datasets))
    jitter_factor_y = np.random.uniform(0.97, 1.03, len(df_datasets))

    df_datasets["jittered_samples"] = (
        df_datasets["dataset_samples_count"] * jitter_factor_x
    )
    df_datasets["jittered_features"] = (
        df_datasets["dataset_raw_features_count"] * jitter_factor_y
    )

    # --------------------------------------------------------------------------
    # 4. Construct Explicit Legend and Marker Mapping with Subgroup Counts
    # --------------------------------------------------------------------------
    task_counts = df_datasets["task_type"].value_counts()
    bin_num = task_counts.get("binary", 0)
    mul_num = task_counts.get("multiclass", 0)
    reg_num = task_counts.get("regression", 0)

    legend_bin = f"binary (n={bin_num})"
    legend_mul = f"multiclass (n={mul_num})"
    legend_reg = f"regression (n={reg_num})"

    legend_mapping = {
        "binary": legend_bin,
        "multiclass": legend_mul,
        "regression": legend_reg,
    }
    df_datasets["task_type_legend"] = df_datasets["task_type"].map(legend_mapping)

    # Custom Muted Academic Color Palette
    # palette_mapping = {
    #     legend_bin: "#2b5c8f",  # Deep Slate Blue
    #     legend_mul: "#d97d38",  # Warm Terracotta/Amber
    #     legend_reg: "#4e8c61",  # Muted Sage Green
    # }
    palette_mapping = {
        legend_bin: "#4477AA",  # Distinct Blue
        legend_mul: "#EE6677",  # Red/Rose
        legend_reg: "#228833",  # Deep Crisp Green
    }

    markers_mapping = {
        legend_bin: "o",  # Circle
        legend_mul: "s",  # Square
        legend_reg: "^",  # Triangle
    }

    # --------------------------------------------------------------------------
    # 5. Render the Polished Bubble Chart
    # --------------------------------------------------------------------------
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 7))

    # Swapped inputs to use the new jittered tracking columns
    sns.scatterplot(
        data=df_datasets,
        x="jittered_samples",
        y="jittered_features",
        hue="task_type_legend",
        # style="task_type_legend",
        size="categorical_pct",
        sizes=(90, 330),
        palette=palette_mapping,
        # markers=markers_mapping,
        alpha=0.8,
        edgecolor="white",
        linewidth=1.2,
        ax=ax,
    )

    # Configure Logarithmic scales to handle severe data layout skew
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Clean scalar layout formatting for ticks instead of scientific notations
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    # Open up the canvas framing (Despine top and right structural borders)
    sns.despine(left=False, bottom=False, top=True, right=True)

    # --------------------------------------------------------------------------
    # 6. Labels, Titles, and Legend Post-Processing
    # --------------------------------------------------------------------------
    ax.set_title(
        f"TabArena (n={len(df_datasets)} datasets)",
        fontsize=13,
        pad=15,
        weight="bold",
    )
    ax.set_xlabel("Number of Samples (Log Scale)", fontsize=11, labelpad=10)
    ax.set_ylabel("Number of Features (Log Scale)", fontsize=11, labelpad=10)

    # Post-process structural strings to ensure clean presentation within legend box
    handles, labels = ax.get_legend_handles_labels()

    cleaned_handles = []
    cleaned_labels = []

    # Target pixel area matching the 40% marker interpolation: 186
    target_legend_marker_size = 186

    for handle, label in zip(handles, labels):
        if label == "task_type_legend":
            cleaned_handles.append(handle)
            cleaned_labels.append("Task Type")
        elif label == "categorical_pct":
            # Inject an entirely invisible rectangle element to serve as an empty
            # spacer row right before starting the next subgroup block.
            spacer = mpatches.Rectangle(
                (0, 0), 1, 1, fill=False, edgecolor="none", visible=False
            )
            cleaned_handles.append(spacer)
            cleaned_labels.append("")

            cleaned_handles.append(handle)
            cleaned_labels.append("Categorical Features (%)")
        else:
            try:
                val = float(label)
                cleaned_handles.append(handle)
                cleaned_labels.append(f"{int(val)}%")
            except ValueError:
                # Type check the handle to adjust size safely based on its matplotlib class
                if hasattr(handle, "set_sizes"):
                    handle.set_sizes([target_legend_marker_size])
                elif hasattr(handle, "set_markersize"):
                    handle.set_markersize(np.sqrt(target_legend_marker_size))

                cleaned_handles.append(handle)
                cleaned_labels.append(label)

    ax.legend(
        cleaned_handles,
        cleaned_labels,
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )

    # --------------------------------------------------------------------------
    # 7. Safe Script-Relative File Output Pipeline
    # --------------------------------------------------------------------------
    plt.tight_layout()
    output_filepath = os.path.join(script_dir, "tabarena.png")
    plt.savefig(output_filepath, dpi=300)
    print(f"Graph successfully generated and saved to: {output_filepath}")


if __name__ == "__main__":
    generate_corpus_bubble_chart()
