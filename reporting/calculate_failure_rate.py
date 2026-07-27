"""
Calculates dataset median-based normalized scores per variant. Computes the failure rate
(percentage of datasets where a variant achieves a normalized score of 0.0)
across the overall benchmark and stratified by task type and scale.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.resolve()

INPUT_FILE = SCRIPT_DIR / "summary_matrix_refactored.csv"
OUTPUT_FILE = SCRIPT_DIR / "outputs" / "failure_rates.csv"

VALUE_COL = "eval_primary_value"

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

STRATA_ORDER = ["overall", "small", "medium", "binary", "multiclass", "regression"]
# ---------------------


def determine_variant(row):
    """Exact variant mapping from the boxplot/leaderboard standards."""
    mode = str(row.get("mode", "")).lower().strip()
    algo = str(row.get("algorithm", "")).lower().strip()

    if "tabpfn" in algo or "tabpfn" in mode:
        return "baseline_tabpfn"
    elif "combined" in mode:
        return f"combined_{algo}"
    elif "embed-only" in mode or "embed_only" in mode:
        return f"embed-only_{algo}"
    elif "raw-only" in mode or "raw_only" in mode or "raw" == mode:
        return f"baseline_{algo}"

    return row.get("Strategy", "unknown")


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    # 1. Map Variants & Clean Data
    df["Variant"] = df.apply(determine_variant, axis=1)
    df = df[df["Variant"].isin(VARIANT_ORDER)].copy()

    df["Task Type"] = df["task_type"].str.lower().str.strip()
    df = df[df["Task Type"].isin(["binary", "multiclass", "regression"])].copy()

    if "dataset_samples_count" in df.columns:
        df["dataset_samples_count"] = pd.to_numeric(
            df["dataset_samples_count"], errors="coerce"
        )
        # Ensure we use the exact scale threshold
        df["Scale"] = np.where(df["dataset_samples_count"] <= 10000, "small", "medium")
    else:
        df["Scale"] = "unknown"

    # 2. Compute Median-Based Normalized Scores (Unified Logic)
    df["normalized_score"] = 0.0

    for (dataset, metric), sub_df in df.groupby(["dataset", "primary_metric"]):
        raw_vals = sub_df[VALUE_COL]

        # Topline (Best) is always the minimum since all metrics are minimized
        topline = raw_vals.min()
        baseline = raw_vals.median()

        # Leaderboard Tie-Breaker Logic
        if topline == baseline:
            # If the median IS the best score, award 1.0 to the winners and 0.0 to the rest
            scores = pd.Series(0.0, index=raw_vals.index)
            scores[raw_vals == topline] = 1.0
        else:
            # Standard relative distance logic
            scores = (baseline - raw_vals) / (baseline - topline)
            # Bound the scores between 0.0 and 1.0
            scores = scores.clip(0.0, 1.0)

        df.loc[sub_df.index, "normalized_score"] = scores

    # 3. Expand dataset into all required strata
    df_overall = df.assign(strata="overall")
    df_task = df.assign(strata=df["Task Type"])
    df_scale = df.assign(strata=df["Scale"])

    df_expanded = pd.concat([df_overall, df_task, df_scale], ignore_index=True)

    # Drop any rows that don't fit our exact strata definitions
    df_expanded = df_expanded[df_expanded["strata"].isin(STRATA_ORDER)]

    # 4. Count total datasets per strata and variant
    totals = (
        df_expanded.groupby(["strata", "Variant"])["dataset"]
        .nunique()
        .reset_index(name="total_datasets")
    )

    # 5. Count failures (score <= 0.0)
    failures = (
        df_expanded[df_expanded["normalized_score"] <= 0.0]
        .groupby(["strata", "Variant"])["dataset"]
        .nunique()
        .reset_index(name="zero_count")
    )

    # 6. Merge results and compute percentages
    results = pd.merge(totals, failures, on=["strata", "Variant"], how="left").fillna(0)

    results["zero_count"] = results["zero_count"].astype(int)
    results["failure_rate"] = (results["zero_count"] / results["total_datasets"]).round(
        4
    )
    results["failure_pct"] = (results["failure_rate"] * 100).round(2)

    # 7. Apply strict sorting
    results["strata"] = pd.Categorical(
        results["strata"], categories=STRATA_ORDER, ordered=True
    )
    results["Variant"] = pd.Categorical(
        results["Variant"], categories=VARIANT_ORDER, ordered=True
    )

    results = results.sort_values(["strata", "Variant"]).reset_index(drop=True)
    results = results.rename(columns={"Variant": "mode_algorithm"})

    # 8. Save to output
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"Failure rates successfully saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
