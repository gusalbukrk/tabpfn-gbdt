"""
This script filters the project summary matrix to exclude standalone TabPFN baseline rows
and extracts key structural columns, including the algorithm, execution mode, dataset dimensions,
feature share metrics, task type, and primary/secondary evaluation metrics.
"""

from pathlib import Path
import pandas as pd

# ==============================================================================
# CONFIGURATION: DECLARE FILE PATHS HERE (Relative to script location)
# ==============================================================================
INPUT_CSV_RELATIVE = "../summary_matrix_refactored.csv"
OUTPUT_CSV_RELATIVE = "summary.csv"
# ==============================================================================

# Resolve absolute paths relative to the directory containing this script
SCRIPT_DIR = Path(__file__).parent.resolve()
INPUT_PATH = SCRIPT_DIR / INPUT_CSV_RELATIVE
OUTPUT_PATH = SCRIPT_DIR / OUTPUT_CSV_RELATIVE


def main():
    # Load dataset
    df = pd.read_csv(INPUT_PATH)

    # Filter out TabPFN baseline configurations
    filtered_df = df[df["algorithm"].str.lower() != "tabpfn"]

    # Target features for structural mapping
    columns_to_extract = [
        "dataset",
        "task_type",
        "algorithm",
        "mode",
        "dataset_samples_count",
        "dataset_raw_features_count",
        "pca_n_components",
        "feat_total_count",
        "feat_share_embedded_pct",
        "feat_share_raw_pct",
        "primary_metric",
        "eval_primary_value",
        "secondary_metric",
        "eval_secondary_value",
    ]

    # Subset the specified columns
    extracted_df = filtered_df[columns_to_extract]

    # Save outputs
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    extracted_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Filtered matrix successfully generated at: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
