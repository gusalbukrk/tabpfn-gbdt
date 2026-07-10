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
# Script: export_categorical_percentages.py
# Description: Extracts metadata from the TabArena datasets using the summary
#              matrix and individual .npz archives, computes the structural
#              percentage of categorical features (strictly what was passed
#              to the GBDT models, excluding the target), and exports to CSV.
# ==============================================================================

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- Configuration (Paths relative to this script's location) ---
SUMMARY_MATRIX_REL_PATH = "../../outputs/training-consolidated/summary_matrix.csv"
ARCHIVES_REL_DIR = "../archives"


def export_categorical_percentages():
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
    # 3. Compute Experimental Categorical % (Matching GBDT Pipeline Reality)
    # --------------------------------------------------------------------------
    output_rows = []

    for idx, row in df_datasets.iterrows():
        dataset_name = str(row["dataset"])
        npz_filename = f"{dataset_name}.npz"
        npz_path = os.path.join(archives_abs_dir, npz_filename)

        cat_pct = 0.0
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
            except Exception:
                cat_pct = 0.0
        else:
            cat_pct = 0.0

        output_rows.append({"name": dataset_name, "categorical_perc": cat_pct})

    # --------------------------------------------------------------------------
    # 4. Construct Output DataFrame, Format Numbers, and Save Target CSV
    # --------------------------------------------------------------------------
    df_out = pd.DataFrame(output_rows)

    # Format cleanly to 2 decimal places, stripping trailing zeros
    df_out["categorical_perc"] = (
        df_out["categorical_perc"]
        .apply(lambda x: f"{x:.2f}")
        .str.rstrip("0")
        .str.rstrip(".")
    )

    # Save output CSV
    output_filepath = os.path.join(script_dir, "tabarena_categorical_percentages.csv")
    df_out.to_csv(output_filepath, index=False)
    print(f"CSV file successfully generated and saved to: {output_filepath}")


if __name__ == "__main__":
    export_categorical_percentages()
