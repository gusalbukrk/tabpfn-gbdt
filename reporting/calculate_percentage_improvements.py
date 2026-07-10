"""
==============================================================================
Script: calculate_median_percentage_improvement.py

Description:
Calculates the Unconditional Median Percentage Improvement (% Delta) Matrix
across all datasets and sub-categories (Overall, Task Types, and Sizes).

Why this script strictly uses the Median:
- Calculating the arithmetic mean of percentage ratios where the denominator
  can approach zero leads to mathematical explosions (e.g., -5000% penalties).
- The Median is completely robust to these unbounded outliers, providing the
  true, stable representation of typical framework performance gains relative
  to traditional raw baseline engineering.

Output is strictly sorted descending by the Overall percentage improvement.
==============================================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_CSV = "summary_matrix_refactored.csv"
OUTPUT_MEDIAN_CSV = "outputs/percentage_improvement_median.csv"

# Metrics where a lower value indicates better performance (Error metrics)
LOWER_IS_BETTER = ["1-auroc", "log_loss", "rmse", "mae", "mse", "error"]
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
    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        print(f"Error: Input file not found at {INPUT_CSV}")
        return

    df = pd.read_csv(input_path)

    # 1. Map Paradigms
    df["Paradigm"] = df.apply(determine_paradigm, axis=1)
    df = df[df["Paradigm"].notna()].copy()

    # 2. Extract Subsets Metadata (Task Type and Size)
    df["Task Type"] = df["task_type"].str.lower().str.strip()

    # Map Size based on row count thresholds
    row_col = next(
        (
            col
            for col in df.columns
            if any(x in col.lower() for x in ["instance", "sample", "row"])
            and "count" in col.lower()
        ),
        None,
    )
    if row_col and df[row_col].dtype in [np.int64, np.float64]:
        df["Size"] = np.where(df[row_col] < 10000, "Small", "Medium")
    else:
        size_col = next((c for c in df.columns if "size" in c.lower()), None)
        if size_col:
            df["Size"] = (
                np.where(df[size_col] < 10000, "Small", "Medium")
                if df[size_col].dtype in [np.int64, np.float64]
                else df[size_col].str.capitalize()
            )
        else:
            df["Size"] = "Unknown"

    # 3. Calculate Percentage Improvement relative to raw-only_*
    dataset_ceilings = []

    for (dataset, metric), sub_df in df.groupby(["dataset", "primary_metric"]):
        is_lower = str(metric).lower() in LOWER_IS_BETTER
        task = sub_df["Task Type"].iloc[0]
        d_size = sub_df["Size"].iloc[0]

        # Isolate the baseline value (best performance achieved by raw-only_*)
        raw_baseline_df = sub_df[sub_df["Paradigm"] == "raw-only_*"]
        if raw_baseline_df.empty:
            continue

        raw_baseline_val = (
            raw_baseline_df["eval_primary_value"].min()
            if is_lower
            else raw_baseline_df["eval_primary_value"].max()
        )

        if raw_baseline_val == 0:
            continue  # Avoid division by zero anomalies

        # Extract the ceiling (best performance) for each paradigm on this dataset
        for paradigm in ["tabpfn_baseline", "combined_*", "embed-only_*", "raw-only_*"]:
            p_df = sub_df[sub_df["Paradigm"] == paradigm]
            if p_df.empty:
                continue

            p_val = (
                p_df["eval_primary_value"].min()
                if is_lower
                else p_df["eval_primary_value"].max()
            )

            # Calculate Percentage Delta
            if is_lower:
                # Lower is better (Error): Positive value means strategy error is lower than raw baseline
                pct_improvement = (
                    (raw_baseline_val - p_val) / abs(raw_baseline_val)
                ) * 100
            else:
                # Higher is better (Score): Positive value means strategy score is higher than raw baseline
                pct_improvement = (
                    (p_val - raw_baseline_val) / abs(raw_baseline_val)
                ) * 100

            dataset_ceilings.append(
                {
                    "dataset": dataset,
                    "Task Type": task,
                    "Size": d_size,
                    "Paradigm": paradigm,
                    "Pct_Improvement": pct_improvement,
                }
            )

    ceilings_df = pd.DataFrame(dataset_ceilings)

    # 4. Define Subset Filters
    filters = {
        "Overall": lambda d: d,
        "Classification": lambda d: d[
            d["Task Type"].isin(["binary", "multiclass", "classification"])
        ],
        "Regression": lambda d: d[d["Task Type"] == "regression"],
        "Binary": lambda d: d[d["Task Type"] == "binary"],
        "Multiclass": lambda d: d[d["Task Type"] == "multiclass"],
        "Small": lambda d: d[d["Size"] == "Small"],
        "Medium": lambda d: d[d["Size"] == "Medium"],
    }

    # 5. Compute Metrics for Sorting
    paradigms = ["tabpfn_baseline", "combined_*", "embed-only_*", "raw-only_*"]
    overall_medians = {}

    for paradigm in paradigms:
        p_subset = ceilings_df[ceilings_df["Paradigm"] == paradigm]
        if not p_subset.empty:
            overall_medians[paradigm] = p_subset["Pct_Improvement"].median()
        else:
            overall_medians[paradigm] = -9999.0

    sorted_by_median = sorted(paradigms, key=lambda p: overall_medians[p], reverse=True)

    # 6. Generate Median Matrix
    median_matrix = []
    for paradigm in sorted_by_median:
        row_data = {"Architecture Paradigm": paradigm}
        for col_name, filter_func in filters.items():
            subset_df = filter_func(ceilings_df)
            p_subset = subset_df[subset_df["Paradigm"] == paradigm]
            if p_subset.empty:
                row_data[col_name] = "N/A"
            else:
                med = p_subset["Pct_Improvement"].median()
                row_data[col_name] = (
                    f"{med:+.2f}%" if paradigm != "raw-only_*" else "0.00%"
                )
        median_matrix.append(row_data)

    df_median_out = pd.DataFrame(median_matrix)

    # 7. Save Output
    Path(OUTPUT_MEDIAN_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_median_out.to_csv(OUTPUT_MEDIAN_CSV, index=False)

    print(
        "\n=============================================================================================="
    )
    print(
        "               MEDIAN MACRO PERCENTAGE IMPROVEMENT MATRIX (REF: RAW-ONLY)                      "
    )
    print(
        "=============================================================================================="
    )
    print(df_median_out.to_string(index=False))
    print(
        "==============================================================================================\n"
    )
    print(f"Median table saved to: {OUTPUT_MEDIAN_CSV}")


if __name__ == "__main__":
    main()
