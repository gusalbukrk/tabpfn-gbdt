"""
==============================================================================
Script: generate_metric_summary.py

Description:
Reads the intrinsic evaluation metrics CSV, computes macro-aggregations
grouped by task type (binary, multiclass, regression), appends an overall
global cohort summary row, and exports the final matrix to the root directory.
==============================================================================
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_CSV = Path("outputs/intrinsic_evaluation_metrics.csv")
OUTPUT_CSV = Path("./summary.csv")
# ==============================================================================


def main():
    if not INPUT_CSV.exists():
        print(
            f"Error: Target file not found at '{INPUT_CSV.resolve()}'. Ensure the evaluation script ran successfully."
        )
        return

    print(f"Loading intrinsic metrics from {INPUT_CSV.name}...")
    df = pd.read_csv(INPUT_CSV)

    # Standardize string formatting for clean grouping
    df["task_type"] = df["task_type"].str.lower().str.strip()

    # Core continuous metrics to aggregate
    metrics = ["silhouette_score", "knn_purity"]
    stats = ["mean", "median", "std", "min", "max"]

    print("Computing descriptive topological statistics grouped by task type...")
    # Generate aggregations grouped by architectural task
    grouped_summary = df.groupby("task_type")[metrics].agg(stats)

    # Flatten multi-index column headers for standard CSV parsing compatibility
    grouped_summary.columns = [
        f"{metric}_{stat}" for metric, stat in grouped_summary.columns
    ]
    grouped_summary = grouped_summary.reset_index()

    print("Generating global cohort baseline summary...")
    # Generate macro-aggregations across the entire 51-dataset space unconditionally
    global_dict = {"task_type": "all_combined"}
    for metric in metrics:
        global_dict[f"{metric}_mean"] = df[metric].mean()
        global_dict[f"{metric}_median"] = df[metric].median()
        global_dict[f"{metric}_std"] = df[metric].std()
        global_dict[f"{metric}_min"] = df[metric].min()
        global_dict[f"{metric}_max"] = df[metric].max()

    global_summary = pd.DataFrame([global_dict])

    # Interlock tables cleanly into a singular presentation matrix
    final_summary = pd.concat([grouped_summary, global_summary], ignore_index=True)

    # Enforce standard academic 4-decimal precision limit
    final_summary = final_summary.round(4)

    # Export structured output matrix to root destination
    final_summary.to_csv(OUTPUT_CSV, index=False)
    print(
        f"\nExecution Complete! Summary ledger successfully written to: {OUTPUT_CSV.resolve()}"
    )


if __name__ == "__main__":
    main()
