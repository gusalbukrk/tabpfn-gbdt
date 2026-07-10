# ==============================================================================
# COLUMN DEFINITIONS FOR THE SUMMARY TABLE:
#
# - Categorical (%): The percentage of raw features within the feature matrix X
#   that were programmatically identified as categorical and evaluated by the
#   GBDT pipeline (excluding the target variable y).
#
# - Sparsity (%): The proportion of missing elements (NaN values) across the
#   entire feature matrix X, reflecting dataset completeness.
#
# - Imbalance Ratio: Calculated strictly for classification tasks as the ratio
#   of majority class instances to minority class instances (N_majority / N_minority).
#   The ideal value is 1.00 (perfect class balance). Higher values indicate
#   severe class imbalance, which typically degrades classification performance.
#
# - Target Skewness: Calculated strictly for regression tasks to measure the
#   asymmetry of the target variable y distribution. An ideal value is 0.00
#   (perfect symmetry, like a normal distribution). Highly positive or negative
#   values (beyond |1.00|) indicate heavy tails or severe outliers, which heavily
#   penalize standard gradient split criteria like Mean Squared Error (MSE).
#
# - Coefficient of Variation (CV): Calculated strictly for regression tasks as
#   the ratio of the target's standard deviation to its mean (σ / μ). It provides
#   a dimensionless measure of relative dispersion. Lower values closer to 0.00
#   indicate that target values are tightly clustered around the mean relative to
#   their scale, whereas high values signify substantial relative volatility.
# ==============================================================================

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
# Script: export_corpus_summary_table.py
# Description: Extracts structural, imbalance, and regression complexity profiles
#              for every individual dataset in the TabArena corpus. Sorts the
#              repository sequentially by sample size (ascending) and exports
#              to Markdown, dataset-level CSV, and an aggregate summary statistics CSV.
# ==============================================================================

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- Configuration (Paths relative to this script's location) ---
SUMMARY_MATRIX_REL_PATH = "../../outputs/training-consolidated/summary_matrix.csv"
ARCHIVES_REL_DIR = "../archives"


def generate_dataset_summary_table():
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
    # 2. Load, Deduplicate, and Order Summary Matrix Data by Sample Count
    # --------------------------------------------------------------------------
    df_raw = pd.read_csv(matrix_abs_path)
    df_datasets = df_raw.drop_duplicates(subset=["dataset"]).copy()

    # Enforce strict sequential ordering from smallest to largest dataset size
    df_datasets = df_datasets.sort_values(by="dataset_samples_count", ascending=True)

    # --------------------------------------------------------------------------
    # 3. Extract and Calculate Comprehensive Row-Level Metrics
    # --------------------------------------------------------------------------
    table_rows = []

    for idx, row in df_datasets.iterrows():
        dataset_name = str(row["dataset"])
        npz_filename = f"{dataset_name}.npz"
        npz_path = os.path.join(archives_abs_dir, npz_filename)

        # Baseline fallbacks from summary matrix
        samples = (
            int(row["dataset_samples_count"]) if "dataset_samples_count" in row else 0
        )
        features = (
            int(row["dataset_raw_features_count"])
            if "dataset_raw_features_count" in row
            else 0
        )
        task_str = str(row["task_type"]) if "task_type" in row else "binary"

        classes_display = "-"
        cat_pct = 0.0
        sparsity_pct = 0.0
        imbalance_ratio_display = "-"
        target_skew_display = "-"
        target_cv_display = "-"

        if os.path.exists(npz_path):
            try:
                npz_data = np.load(npz_path, allow_pickle=True)

                # Overwrite shapes with precise archive array shapes if available
                if "X_train_raw" in npz_data:
                    features = int(npz_data["X_train_raw"].shape[1])
                elif "n_features_raw" in npz_data:
                    features = int(npz_data["n_features_raw"])

                if "task_type" in npz_data:
                    task_str = str(npz_data["task_type"].item())

                # Class quantity configuration
                if task_str == "binary":
                    classes_display = "2"
                elif task_str == "multiclass":
                    if "y_train" in npz_data:
                        classes_display = str(len(np.unique(npz_data["y_train"])))
                    elif "n_classes" in npz_data:
                        classes_display = str(npz_data["n_classes"].item())
                    else:
                        classes_display = "3+"
                    if classes_display == "2":  # Safe parsing validation
                        task_str = "binary"
                else:
                    classes_display = "-"

                # Pure structural categorical feature percentage
                cat_features_count = (
                    len(npz_data["cat_features"]) if "cat_features" in npz_data else 0
                )
                cat_pct = (cat_features_count / features) * 100 if features > 0 else 0.0

                # Matrix sparsity percentage derivation
                if "missing_value_ratio" in npz_data:
                    sparsity_pct = float(npz_data["missing_value_ratio"].item()) * 100

                # Target metric profiling based on operational paradigm
                if task_str in ["binary", "multiclass"]:
                    if "y_train" in npz_data:
                        y_arr = npz_data["y_train"].ravel()
                        _, counts = np.unique(y_arr, return_counts=True)
                        ir_val = float(np.max(counts) / np.min(counts))
                        imbalance_ratio_display = f"{ir_val:.2f}"
                    elif "target_profile_val" in npz_data:
                        # Synchronized transformation step to obtain majority/minority distribution
                        minority_ratio = float(npz_data["target_profile_val"].item())
                        if minority_ratio > 0:
                            ir_val = (1.0 / minority_ratio) - 1.0
                            imbalance_ratio_display = f"{ir_val:.2f}"
                        else:
                            imbalance_ratio_display = "1.00"
                else:
                    # Regression tasks: extract metrics directly from target array arrays
                    y_arr = None
                    if "y_train" in npz_data:
                        y_arr = npz_data["y_train"]
                    elif "y_train_raw" in npz_data:
                        y_arr = npz_data["y_train_raw"]

                    if y_arr is not None and len(y_arr) > 0:
                        y_series = pd.Series(y_arr.ravel())

                        # 1. Target Skewness calculation
                        skew_val = y_series.skew()
                        target_skew_display = (
                            f"{skew_val:.2f}" if not pd.isna(skew_val) else "0.00"
                        )

                        # 2. Coefficient of Variation (CV = std / mean) calculation
                        mean_val = y_series.mean()
                        std_val = y_series.std()
                        if mean_val != 0:
                            target_cv_display = f"{std_val / mean_val:.2f}"
                        else:
                            target_cv_display = "0.00"
            except Exception:
                pass
        else:
            if task_str == "binary":
                classes_display = "2"

        table_rows.append(
            {
                "Name": dataset_name,
                "Samples": samples,
                "Features": features,
                "Classes": classes_display,
                "Categorical (%)": f"{cat_pct:.2f}".rstrip("0").rstrip("."),
                "Sparsity (%)": f"{sparsity_pct:.2f}".rstrip("0").rstrip("."),
                "Imbalance Ratio": imbalance_ratio_display,
                "Target Skewness": target_skew_display,
                "Coefficient of Variation": target_cv_display,
            }
        )

    df_out = pd.DataFrame(table_rows)

    # Clean up empty decimal strings resulting from formatting rstrip
    df_out["Categorical (%)"] = df_out["Categorical (%)"].apply(
        lambda x: "0" if x == "" else x
    )
    df_out["Sparsity (%)"] = df_out["Sparsity (%)"].apply(
        lambda x: "0" if x == "" else x
    )

    # Sort alphabetically by dataset name before saving and aggregating
    df_out = df_out.sort_values(by="Name", ascending=True).reset_index(drop=True)

    # --------------------------------------------------------------------------
    # 4. Save Dataset CSV Output and Generate Markdown Text
    # --------------------------------------------------------------------------
    csv_filepath = os.path.join(script_dir, "tabarena_datasets_profiling.csv")
    df_out.to_csv(csv_filepath, index=False)
    print(f"Dataset CSV file successfully saved to: {csv_filepath}")

    markdown_table = df_out.to_markdown(index=False)
    print("\n### ACADEMIC SUMMARY TABLE FOR DISSERTATION/PAPER ###")
    print(markdown_table)

    # --------------------------------------------------------------------------
    # 5. Compute and Save Aggregate Metrics CSV Profile
    # --------------------------------------------------------------------------
    columns_to_aggregate = [
        "Samples",
        "Features",
        "Classes",
        "Categorical (%)",
        "Sparsity (%)",
        "Imbalance Ratio",
        "Target Skewness",
        "Coefficient of Variation",
    ]

    aggregate_rows = []

    for col in columns_to_aggregate:
        # Convert column series to numeric float values safely
        numeric_series = pd.to_numeric(
            df_out[col].replace("-", np.nan), errors="coerce"
        )
        valid_series = numeric_series.dropna()

        # FILTER: If evaluating class quantity, isolate strictly multiclass spaces (> 2 classes)
        if col == "Classes":
            valid_series = valid_series[valid_series > 2]

        if not valid_series.empty:
            min_val = valid_series.min()
            max_val = valid_series.max()
            med_val = valid_series.median()
            mean_val = valid_series.mean()
            std_val = valid_series.std() if len(valid_series) > 1 else 0.0

            # Apply clean string conversions depending on column type requirements
            if col in ["Samples", "Features", "Classes"]:
                min_str = f"{int(min_val)}"
                max_str = f"{int(max_val)}"
                med_str = (
                    f"{int(med_val)}" if med_val.is_integer() else f"{med_val:.1f}"
                )
                mean_sd_str = f"{mean_val:.2f} ± {std_val:.2f}"
            else:
                min_str = f"{min_val:.2f}".rstrip("0").rstrip(".")
                max_str = f"{max_val:.2f}".rstrip("0").rstrip(".")
                med_str = f"{med_val:.2f}".rstrip("0").rstrip(".")
                mean_sd_str = f"{mean_val:.2f} ± {std_val:.2f}"

            # Append structural suffix indicators cleanly
            if "%" in col:
                min_str += "%"
                max_str += "%"
                med_str += "%"

            aggregate_rows.append(
                {
                    "Data Dimension": col,
                    "Minimum": "0" if min_str == "" else min_str,
                    "Maximum": "0" if max_str == "" else max_str,
                    "Median": "0" if med_str == "" else med_str,
                    "Mean ± SD": mean_sd_str,
                }
            )
        else:
            aggregate_rows.append(
                {
                    "Data Dimension": col,
                    "Minimum": "-",
                    "Maximum": "-",
                    "Median": "-",
                    "Mean ± SD": "-",
                }
            )

    df_agg = pd.DataFrame(aggregate_rows)

    # Save Aggregates summary to a clean CSV
    agg_csv_filepath = os.path.join(
        script_dir, "tabarena_datasets_profiling_aggregates.csv"
    )
    df_agg.to_csv(agg_csv_filepath, index=False)
    print(f"Aggregate Summary CSV file successfully saved to: {agg_csv_filepath}")

    print("\n### CORPUS AGGREGATION SUMMARY TABLE ###")
    print(df_agg.to_markdown(index=False))


if __name__ == "__main__":
    generate_dataset_summary_table()
