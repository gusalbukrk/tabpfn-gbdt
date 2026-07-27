"""
Calculates detailed distribution statistics (Min, P10, Q1, Median, Q3, IQR, P90, Max, Mean)
for median-based normalized scores of tabular ML models. Enforces custom variant ordering
(TabPFN -> Raw+Embed -> GBDT Baseline -> Embed-Only, with LightGBM -> CatBoost -> XGBoost)
and outputs overall and strata summaries to CSV files in the ./outputs directory.
"""

import pandas as pd
import numpy as np
import os

# --- Configuration & File Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(SCRIPT_DIR, "summary_matrix_refactored.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
OUTPUT_OVERALL_FILE = os.path.join(OUTPUT_DIR, "quantile_summaries.csv")
OUTPUT_STRATA_FILE = os.path.join(OUTPUT_DIR, "quantile_summaries_by_strata.csv")

SMALL_SCALE_THRESHOLD = 2500

# Explicitly define model variant hierarchy
VARIANT_ORDER = [
    "baseline_tabpfn",
    "combined_lightgbm",
    "combined_catboost",
    "combined_xgboost",
    "baseline_lightgbm",
    "baseline_catboost",
    "baseline_xgboost",
    "embed-only_lightgbm",
    "embed-only_catboost",
    "embed-only_xgboost",
]

STRATA_ORDER = ["binary", "multiclass", "regression", "small-scale", "medium-scale"]

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load data
df = pd.read_csv(INPUT_FILE)

# 2. Extract and rename relevant columns
df = df[
    [
        "dataset",
        "task_type",
        "dataset_samples_count",
        "primary_metric",
        "eval_primary_value",
        "mode_algorithm",
    ]
].copy()

df = df.rename(
    columns={
        "task_type": "task",
        "primary_metric": "metric_name",
        "eval_primary_value": "metric_value",
        "mode_algorithm": "variant",
    }
)

# 3. Clean numeric columns
df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce")
df["dataset_samples_count"] = pd.to_numeric(
    df["dataset_samples_count"], errors="coerce"
)
df = df.dropna(subset=["metric_value"])

# 4. Define scale strata
df["scale_strata"] = np.where(
    df["dataset_samples_count"] < SMALL_SCALE_THRESHOLD, "small-scale", "medium-scale"
)


# 5. Define optimization direction
def is_higher_better(metric_str):
    """Determine if a metric should be maximized."""
    m = str(metric_str).upper()
    return m in ["R2_SCORE", "ROC_AUC", "ACCURACY", "F1", "BALANCED_ACCURACY"]


# 6. Compute Median-Based Normalized Scores
def compute_median_normalized(group):
    metric = group["metric_name"].iloc[0]
    higher_is_better = is_higher_better(metric)

    vals = group["metric_value"]
    median_val = vals.median()

    if higher_is_better:
        best_val = vals.max()
        if best_val == median_val:
            scores = np.where(vals >= best_val, 1.0, 0.0)
        else:
            scores = (vals - median_val) / (best_val - median_val)
    else:
        best_val = vals.min()
        if best_val == median_val:
            scores = np.where(vals <= best_val, 1.0, 0.0)
        else:
            scores = (median_val - vals) / (median_val - best_val)

    group["norm_score"] = np.clip(scores, 0.0, 1.0)
    return group


df = df.groupby("dataset", group_keys=False).apply(compute_median_normalized)


# 7. Helper function to compute all distribution statistics
def calc_stats(group):
    scores = group["norm_score"]
    q1 = scores.quantile(0.25)
    q3 = scores.quantile(0.75)
    return pd.Series(
        {
            "Min": scores.min(),
            "P10": scores.quantile(0.10),
            "Q1": q1,
            "Median": scores.median(),
            "Q3": q3,
            "IQR": q3 - q1,
            "P90": scores.quantile(0.90),
            "Max": scores.max(),
            "Mean": scores.mean(),
        }
    )


# 8. Calculate Overall Quantiles and Enforce Variant Order
overall_summary = (
    df.groupby("variant", group_keys=False).apply(calc_stats).reset_index()
)
overall_summary["variant"] = pd.Categorical(
    overall_summary["variant"], categories=VARIANT_ORDER, ordered=True
)
overall_summary = overall_summary.sort_values("variant").reset_index(drop=True).round(3)

overall_summary.to_csv(OUTPUT_OVERALL_FILE, index=False)

# 9. Calculate Strata Quantiles (Tasks + Scale) and Enforce Hierarchical Order
task_summary = (
    df.groupby(["task", "variant"], group_keys=False).apply(calc_stats).reset_index()
)
task_summary = task_summary.rename(columns={"task": "strata"})

scale_summary = (
    df.groupby(["scale_strata", "variant"], group_keys=False)
    .apply(calc_stats)
    .reset_index()
)
scale_summary = scale_summary.rename(columns={"scale_strata": "strata"})

strata_summary = pd.concat([task_summary, scale_summary], ignore_index=True)

# Apply categorical ordering to strata and variant columns
strata_summary["strata"] = pd.Categorical(
    strata_summary["strata"], categories=STRATA_ORDER, ordered=True
)
strata_summary["variant"] = pd.Categorical(
    strata_summary["variant"], categories=VARIANT_ORDER, ordered=True
)

strata_summary = (
    strata_summary.sort_values(["strata", "variant"]).reset_index(drop=True).round(3)
)

# Reorder columns so 'strata' and 'variant' come first
stat_cols = ["Min", "P10", "Q1", "Median", "Q3", "IQR", "P90", "Max", "Mean"]
strata_summary = strata_summary[["strata", "variant"] + stat_cols]

strata_summary.to_csv(OUTPUT_STRATA_FILE, index=False)
