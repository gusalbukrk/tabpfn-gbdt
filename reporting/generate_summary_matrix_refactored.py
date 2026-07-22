# get summary_matrix.csv content and create a separate tabpfn line for each dataset

import pandas as pd
from pathlib import Path

# =====================================================================
# PATH CONFIGURATION
# =====================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
# Adjust CSV_PATH to wherever your raw full CSV actually lives
CSV_PATH = (
    SCRIPT_DIR / "../outputs/training-consolidated/summary_matrix.csv"
).resolve()
OUTPUT_FILE = SCRIPT_DIR / "summary_matrix_refactored.csv"

# =====================================================================
# 1. LOAD DATA
# =====================================================================
print(f"Loading data from: {CSV_PATH}...")
df_full = pd.read_csv(CSV_PATH)

# =====================================================================
# 2. STANDARDIZE GBDT ROWS
# =====================================================================
# Create the merged mode_algorithm identifier
df_gbdt = df_full.copy()
df_gbdt["mode_algorithm"] = df_gbdt["mode"] + "_" + df_gbdt["algorithm"]
df_gbdt["mode_algorithm"] = df_gbdt["mode_algorithm"].str.replace(
    "raw-only_", "baseline_", regex=False
)

# =====================================================================
# 3. CREATE TABPFN ROWS
# =====================================================================
# Extract one instance of TabPFN per dataset using the dataset-level columns
df_tabpfn = df_full.drop_duplicates(subset=["dataset"]).copy()

# for the TabPFN lines, keep only the overarching dataset metadata and the TabPFN specific scores
cols_to_keep = [
    "dataset",
    "task_type",
    "dataset_samples_count",
    "dataset_raw_features_count",
    "primary_metric",
    "secondary_metric",
    "tabpfn_primary_score",
    "tabpfn_secondary_score",
]
df_tabpfn = df_tabpfn[cols_to_keep]

# Map the TabPFN scores to the standard evaluation columns
df_tabpfn = df_tabpfn.rename(
    columns={
        "tabpfn_primary_score": "eval_primary_value",
        "tabpfn_secondary_score": "eval_secondary_value",
    }
)

# Explicitly set the identifiers for TabPFN
df_tabpfn["algorithm"] = "tabpfn"
df_tabpfn["mode"] = "baseline"
df_tabpfn["mode_algorithm"] = "baseline_tabpfn"

# =====================================================================
# 4. MERGE & CLEANUP
# =====================================================================
# Concatenate the datasets (pandas will automatically fill missing columns in TabPFN with NaN)
df_refactored = pd.concat([df_gbdt, df_tabpfn], ignore_index=True)

# Drop the old TabPFN columns to finalize the refactoring
df_refactored = df_refactored.drop(
    columns=["tabpfn_primary_score", "tabpfn_secondary_score"]
)

# Sort for readability
df_refactored = df_refactored.sort_values(by=["dataset", "mode_algorithm"]).reset_index(
    drop=True
)

# =====================================================================
# 5. EXPORT
# =====================================================================
df_refactored.to_csv(OUTPUT_FILE, index=False)
print(f"Refactored Wide Matrix successfully saved to: {OUTPUT_FILE}")
