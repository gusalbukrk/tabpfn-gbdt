"""
==============================================================================
Script: calculate_alternative_norm_scores.py

Description:
Calculates an Alternative Normalized Score matrix using standard Min-Max scaling.
THIS IS EXPLICITLY DIFFERENT from the TabArena paper's methodology.

Output is strictly sorted descending by the Overall score across all
10 individual model configurations.
==============================================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_CSV = "../summary_matrix_refactored.csv"
OUTPUT_CSV = "outputs/norm_scores_alternative_individual.csv"

# Metrics where a lower value indicates better performance
LOWER_IS_BETTER = ["1-auroc", "log_loss", "rmse", "mae", "mse", "error"]
# ==============================================================================


def determine_configuration(row):
    """Maps the row to one of the 10 specific model configurations."""
    mode = str(row.get("mode", "")).lower().strip()
    algo = str(row.get("algorithm", "")).lower().strip()

    if "tabpfn" in algo or "tabpfn" in mode:
        return "baseline_tabpfn"

    # Normalize mode names to match the 10 configurations standard
    if mode in ["raw-only", "raw_only", "raw", "baseline"]:
        norm_mode = "baseline"
    elif mode in ["embed-only", "embed_only"]:
        norm_mode = "embed-only"
    else:
        norm_mode = mode

    return f"{norm_mode}_{algo}"


def format_cell(mean_val, std_val):
    """Formats cell strings to display Mean ± Standard Deviation."""
    if pd.isna(mean_val):
        return ""
    if pd.isna(std_val) or std_val == 0.0:
        return f"{mean_val:.3f} ± 0.000"
    return f"{mean_val:.3f} ± {std_val:.3f}"


def main():
    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        print(f"Error: Input file not found at {INPUT_CSV}")
        return

    df = pd.read_csv(input_path)

    # 1. Map Configurations
    df["Configuration"] = df.apply(determine_configuration, axis=1)
    df = df[df["Configuration"].notna()].copy()

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

    # 3. Calculate Min-Max Alternative Normalized Scores
    dataset_scores = []
    all_configs = df["Configuration"].unique()

    for (dataset, metric), sub_df in df.groupby(["dataset", "primary_metric"]):
        is_lower = str(metric).lower() in LOWER_IS_BETTER
        task = sub_df["Task Type"].iloc[0]
        d_size = sub_df["Size"].iloc[0]

        raw_vals = sub_df["eval_primary_value"]
        min_val = raw_vals.min()
        max_val = raw_vals.max()
        val_range = max_val - min_val

        # Calculate normalized score per row
        # After this scaling, 1.0 is ALWAYS the best score, and 0.0 is ALWAYS the worst.
        if val_range == 0:
            sub_df["Alt_Norm_Score"] = 0.0
        else:
            if is_lower:
                sub_df["Alt_Norm_Score"] = (max_val - raw_vals) / val_range
            else:
                sub_df["Alt_Norm_Score"] = (raw_vals - min_val) / val_range

        # 4. Extract the Score per Configuration for this Dataset
        for config in all_configs:
            c_df = sub_df[sub_df["Configuration"] == config]
            if c_df.empty:
                continue

            # Since 1.0 is always best now, we just take the max Alt_Norm_Score
            best_score = c_df["Alt_Norm_Score"].max()

            dataset_scores.append(
                {
                    "dataset": dataset,
                    "Task Type": task,
                    "Size": d_size,
                    "Configuration": config,
                    "Score": best_score,
                }
            )

    scores_df = pd.DataFrame(dataset_scores)

    # 5. Define Subset Filters
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

    # 6. Dynamically Sort Configurations by Overall Score
    overall_means = {}
    for config in all_configs:
        c_subset = scores_df[scores_df["Configuration"] == config]
        if not c_subset.empty:
            overall_means[config] = c_subset["Score"].mean()
        else:
            overall_means[config] = -1.0  # Push to bottom if no data exists

    sorted_configs = sorted(
        overall_means.keys(), key=lambda c: overall_means[c], reverse=True
    )

    # 7. Aggregate Metrics using the sorted order
    matrix_data = []

    for config in sorted_configs:
        row_data = {"Model Configuration": config}

        for col_name, filter_func in filters.items():
            subset_df = filter_func(scores_df)
            c_subset = subset_df[subset_df["Configuration"] == config]

            if c_subset.empty:
                row_data[col_name] = format_cell(np.nan, np.nan)
            else:
                scores = c_subset["Score"]
                mean_s = scores.mean()
                std_s = scores.std(ddof=1) if len(scores) > 1 else 0.0
                row_data[col_name] = format_cell(mean_s, std_s)

        matrix_data.append(row_data)

    output_df = pd.DataFrame(matrix_data)

    # 8. Save Output
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_CSV, index=False)

    print(
        "\n=============================================================================================="
    )
    print(
        "                      ALTERNATIVE NORMALIZED SCORES MATRIX (MIN-MAX)                            "
    )
    print(
        "=============================================================================================="
    )
    print(output_df.to_string(index=False))
    print(
        "==============================================================================================\n"
    )
    print(f"Matrix successfully saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
