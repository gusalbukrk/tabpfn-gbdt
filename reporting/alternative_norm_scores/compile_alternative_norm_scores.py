"""
==============================================================================
Script: calculate_alternative_norm_scores.py

Description:
Calculates an Alternative Normalized Score matrix using standard Min-Max scaling.
THIS IS EXPLICITLY DIFFERENT from the TabArena paper's methodology.

Why this differs from TabArena:
- The TabArena paper (and the original leaderboard generation) anchors the median
  configuration at 0 and clips/truncates all worse-performing models to 0. This
  creates a "black hole" that erases the magnitude of catastrophic model failures.
- This alternative script uses strict Min-Max scaling per dataset. The absolute
  worst configuration is scaled to 0.0, the absolute best to 1.0, and all other
  configurations maintain their exact mathematical relative distances in between.
  This prevents information loss and reveals the true variance and failure distances
  of the baseline models.

Output is strictly sorted descending by the Overall score.
==============================================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_CSV = "../summary_matrix_refactored.csv"
OUTPUT_CSV = "outputs/norm_scores_alternative.csv"

# Metrics where a lower value indicates better performance
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

    # 3. Calculate Min-Max Alternative Normalized Scores
    dataset_ceilings = []

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

        # 4. Extract the Ceiling (Best Score) per Paradigm for this Dataset
        for paradigm in ["tabpfn_baseline", "combined_*", "embed-only_*", "raw-only_*"]:
            p_df = sub_df[sub_df["Paradigm"] == paradigm]
            if p_df.empty:
                continue

            # Since 1.0 is always best now, we just take the max Alt_Norm_Score
            best_score = p_df["Alt_Norm_Score"].max()

            dataset_ceilings.append(
                {
                    "dataset": dataset,
                    "Task Type": task,
                    "Size": d_size,
                    "Paradigm": paradigm,
                    "Ceiling_Score": best_score,
                }
            )

    ceilings_df = pd.DataFrame(dataset_ceilings)

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

    # 6. Dynamically Sort Paradigms by Overall Score
    overall_means = {}
    for paradigm in ["tabpfn_baseline", "combined_*", "embed-only_*", "raw-only_*"]:
        p_subset = ceilings_df[ceilings_df["Paradigm"] == paradigm]
        if not p_subset.empty:
            overall_means[paradigm] = p_subset["Ceiling_Score"].mean()
        else:
            overall_means[paradigm] = -1.0  # Push to bottom if no data exists

    sorted_paradigms = sorted(
        overall_means.keys(), key=lambda p: overall_means[p], reverse=True
    )

    # 7. Aggregate Metrics using the sorted order
    matrix_data = []

    for paradigm in sorted_paradigms:
        row_data = {"Architecture Paradigm": paradigm}

        for col_name, filter_func in filters.items():
            subset_df = filter_func(ceilings_df)
            p_subset = subset_df[subset_df["Paradigm"] == paradigm]

            if p_subset.empty:
                row_data[col_name] = format_cell(np.nan, np.nan)
            else:
                scores = p_subset["Ceiling_Score"]
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
