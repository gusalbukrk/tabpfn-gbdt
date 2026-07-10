"""
==============================================================================
Script: compare_geometries.py

Description:
Performs a comparative analysis between the raw embedding metrics and the
post-PCA embedding metrics. Merges dataset-level files, computes structural
deltas (Information Loss), and compares their aggregated summary statistics.

Outputs are written to the current outputs directory.
==============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Paths as specified by the user's execution environment
CURRENT_PCA_CSV = Path("outputs/pca_intrinsic_evaluation.csv")
BEFORE_PCA_CSV = Path(
    "../embeddings-before-pca/outputs/intrinsic_evaluation_metrics.csv"
)

# Outputs
OUTPUT_DIR = Path("outputs_comparison")
DETAILED_COMP_CSV = OUTPUT_DIR / "detailed_geometric_comparison.csv"
SUMMARY_COMP_CSV = OUTPUT_DIR / "summary_geometric_comparison.csv"
# ==============================================================================


def load_and_standardize(filepath: Path, label: str):
    """Loads a metrics dataframe and standardizes column names for alignment."""
    if not filepath.exists():
        raise FileNotFoundError(f"Missing required file: {filepath.resolve()}")

    df = pd.read_csv(filepath)
    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["task_type"] = df["task_type"].astype(str).str.lower().str.strip()

    # Identify Silhouette column
    sil_col = [col for col in df.columns if "sil" in col.lower()]
    # Identify k-NN column
    knn_col = [
        col for col in df.columns if "knn" in col.lower() or "purity" in col.lower()
    ]

    if not sil_col or not knn_col:
        raise ValueError(
            f"Could not automatically map metric columns in {filepath.name}"
        )

    rename_dict = {sil_col[0]: f"{label}_silhouette", knn_col[0]: f"{label}_knn_purity"}

    # Carry over dimensions if they exist
    dim_col = [col for col in df.columns if "dim" in col.lower()]
    if dim_col:
        rename_dict[dim_col[0]] = f"{label}_dimensions"

    df = df.rename(columns=rename_dict)

    keep_cols = ["dataset", "task_type", f"{label}_silhouette", f"{label}_knn_purity"]
    if dim_col:
        keep_cols.append(f"{label}_dimensions")

    return df[keep_cols]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GEOMETRIC TOPOLOGY COMPARATOR ENGINE")
    print("=" * 70)

    try:
        # Load and cleanly standardize both source files
        print(f"Reading Current PCA: {CURRENT_PCA_CSV.name}...")
        df_after = load_and_standardize(CURRENT_PCA_CSV, "after_pca")

        print(f"Reading Baseline (Before PCA): {BEFORE_PCA_CSV.name}...")
        df_before = load_and_standardize(BEFORE_PCA_CSV, "before_pca")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease verify your file paths. Defaults are configured as:")
        print(f"  - Current: {CURRENT_PCA_CSV}")
        print(f"  - Before:  {BEFORE_PCA_CSV}")
        return
    except Exception as e:
        print(f"\n[ERROR] Failed to load/standardize datasets: {e}")
        return

    # --- 1. DETAILED DATASET COMPARISON ---
    # Merge on dataset key
    merged_df = pd.merge(df_before, df_after, on=["dataset", "task_type"], how="inner")

    if merged_df.empty:
        print(
            "[WARNING] Merge yielded 0 overlapping datasets. Check dataset name alignment."
        )
        return

    print(f"Successfully aligned {len(merged_df)} datasets for comparative profiling.")

    # Calculate Deltas (Before - After)
    # Positive delta = Structural loss due to PCA
    merged_df["silhouette_delta"] = (
        merged_df["before_pca_silhouette"] - merged_df["after_pca_silhouette"]
    )
    merged_df["knn_purity_loss"] = (
        merged_df["before_pca_knn_purity"] - merged_df["after_pca_knn_purity"]
    )

    # Rearrange columns beautifully
    col_order = ["dataset", "task_type"]
    if "before_pca_dimensions" in merged_df.columns:
        col_order.append("before_pca_dimensions")
    if "after_pca_dimensions" in merged_df.columns:
        col_order.append("after_pca_dimensions")

    col_order += [
        "before_pca_silhouette",
        "after_pca_silhouette",
        "silhouette_delta",
        "before_pca_knn_purity",
        "after_pca_knn_purity",
        "knn_purity_loss",
    ]

    merged_df = merged_df[col_order].round(4)
    merged_df.to_csv(DETAILED_COMP_CSV, index=False)
    print(f"-> Detailed dataset comparison written to: {DETAILED_COMP_CSV}")

    # --- 2. AGGREGATED COHORT SUMMARIES ---
    metrics_to_agg = [
        "before_pca_silhouette",
        "after_pca_silhouette",
        "silhouette_delta",
        "before_pca_knn_purity",
        "after_pca_knn_purity",
        "knn_purity_loss",
    ]

    # Compute aggregations by task type
    task_summary = merged_df.groupby("task_type")[metrics_to_agg].mean().reset_index()

    # Compute global baseline average across all combined datasets
    global_dict = {"task_type": "all_combined"}
    for metric in metrics_to_agg:
        global_dict[metric] = merged_df[metric].mean()
    global_summary = pd.DataFrame([global_dict])

    final_summary = pd.concat([task_summary, global_summary], ignore_index=True).round(
        4
    )
    final_summary.to_csv(SUMMARY_COMP_CSV, index=False)
    print(f"-> Aggregated summary comparison written to: {SUMMARY_COMP_CSV}")

    # --- 3. CONSOLE REPORT GENERATION ---
    print("\n" + "=" * 70)
    print("MACRO TOPOLOGICAL COMPARISON SUMMARY (AVERAGES)")
    print("=" * 70)
    print(
        f"{'Task Type':<15} | {'Before kNN':<11} | {'After kNN':<11} | {'Purity Loss':<11}"
    )
    print("-" * 70)
    for _, row in final_summary.iterrows():
        print(
            f"{row['task_type'].upper():<15} | {row['before_pca_knn_purity']:<11.4f} | {row['after_pca_knn_purity']:<11.4f} | {row['knn_purity_loss']:<11.4f}"
        )
    print("=" * 70)
    print("Interpretation:")
    print(" * Positive Purity Loss indicates PCA degraded local neighborhood topology.")
    print(
        " * Negative metrics are highlighted if the pipeline improved representation density."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
