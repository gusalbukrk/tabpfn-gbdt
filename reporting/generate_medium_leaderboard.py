"""
Generates a leaderboard based on the median-normalized scores, filtered strictly
to medium-scale datasets (samples > 10,000). Calculates the mean score per variant
with 95% bootstrap confidence intervals and assigns medals (🥇, 🥈, 🥉) to the top 3
strategies in each task category. Appends the dataset count (N/Total) to the headers.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration ---
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(".").resolve()

INPUT_FILE = SCRIPT_DIR / "summary_matrix_refactored.csv"
OUTPUT_FILE = SCRIPT_DIR / "outputs" / "leaderboard_medium_scale.csv"

VALUE_COL = "eval_primary_value"
BOOTSTRAP_ITERATIONS = 1000

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

STRATA_ORDER = [
    "Overall",
    "Classification",
    "Regression",
    "Binary",
    "Multiclass",
]
# ---------------------


def determine_variant(row):
    """Exact variant mapping from the established standard."""
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


def get_bootstrap_ci(data, iterations=BOOTSTRAP_ITERATIONS):
    """Calculates the 95% confidence interval of the mean using bootstrapping."""
    if len(data) < 2:
        return data.mean(), data.mean(), data.mean()

    means = [
        np.mean(np.random.choice(data, size=len(data), replace=True))
        for _ in range(iterations)
    ]
    return np.mean(data), np.percentile(means, 2.5), np.percentile(means, 97.5)


def apply_medals(col):
    """Ranks the column values (higher is better) and prepends medals."""
    # Extract just the numeric mean for sorting by parsing the string before the '['
    try:
        numeric_means = (
            col.astype(str).str.extract(r"([0-9]+\.[0-9]+)")[0].astype(float)
        )
    except Exception:
        return col

    # Get the thresholds for top 1, 2, 3
    unique_scores = numeric_means.dropna().unique()
    unique_scores.sort()
    unique_scores = unique_scores[::-1]  # Descending

    if len(unique_scores) == 0:
        return col

    top1 = unique_scores[0] if len(unique_scores) > 0 else None
    top2 = unique_scores[1] if len(unique_scores) > 1 else None
    top3 = unique_scores[2] if len(unique_scores) > 2 else None

    formatted_col = []
    for val, num in zip(col, numeric_means):
        if pd.isna(num):
            formatted_col.append("-")
        elif num == top1:
            formatted_col.append(f"🥇 {val}")
        elif num == top2:
            formatted_col.append(f"🥈 {val}")
        elif num == top3:
            formatted_col.append(f"🥉 {val}")
        else:
            formatted_col.append(str(val))

    return formatted_col


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    # 1. Map Variants & Clean Task Types
    df["Variant"] = df.apply(determine_variant, axis=1)
    df = df[df["Variant"].isin(VARIANT_ORDER)].copy()
    df["Task Type"] = df["task_type"].str.lower().str.strip().str.capitalize()

    # 2. Filter STRICTLY to Medium Scale (> 10000 samples)
    if "dataset_samples_count" in df.columns:
        df["dataset_samples_count"] = pd.to_numeric(
            df["dataset_samples_count"], errors="coerce"
        )
        df = df[df["dataset_samples_count"] > 10000].copy()
    else:
        print("Error: 'dataset_samples_count' column missing. Cannot filter scale.")
        return

    if df.empty:
        print("Warning: No datasets found with > 10,000 samples. Exiting.")
        return

    # 3. Compute Median-Based Normalized Scores (Unified Logic)
    dataset_scores = []

    for (dataset, metric), sub_df in df.groupby(["dataset", "primary_metric"]):
        task = sub_df["Task Type"].iloc[0]
        raw_vals = sub_df[VALUE_COL]

        # Minimization logic
        topline = raw_vals.min()
        baseline = raw_vals.median()

        if topline == baseline:
            scores = pd.Series(0.0, index=raw_vals.index)
            scores[raw_vals == topline] = 1.0
        else:
            scores = (baseline - raw_vals) / (baseline - topline)
            scores = scores.clip(0.0, 1.0)

        for idx, val in scores.items():
            dataset_scores.append(
                {
                    "dataset": dataset,
                    "Variant": sub_df.loc[idx, "Variant"],
                    "Task Type": task,
                    "Score": val,
                }
            )

    score_df = pd.DataFrame(dataset_scores)

    # 4. Define Strata DataFrames & Map Header Names
    strata_data = []
    header_mapping = {}

    # Calculate the total unique datasets across all tasks for the denominator
    total_datasets = score_df["dataset"].nunique()

    # Overall
    n_overall = score_df["dataset"].nunique()
    header_mapping["Overall"] = f"Overall ({n_overall}/{total_datasets})"
    for variant, grp in score_df.groupby("Variant"):
        mean_val, ci_low, ci_high = get_bootstrap_ci(grp["Score"].values)
        strata_data.append(
            {
                "Variant": variant,
                "Strata": "Overall",
                "Value": f"{mean_val:.3f} [{ci_low:.3f} - {ci_high:.3f}]",
            }
        )

    # Classification (Binary + Multiclass)
    class_df = score_df[score_df["Task Type"].isin(["Binary", "Multiclass"])]
    n_class = class_df["dataset"].nunique()
    header_mapping["Classification"] = f"Classification ({n_class}/{total_datasets})"
    for variant, grp in class_df.groupby("Variant"):
        mean_val, ci_low, ci_high = get_bootstrap_ci(grp["Score"].values)
        strata_data.append(
            {
                "Variant": variant,
                "Strata": "Classification",
                "Value": f"{mean_val:.3f} [{ci_low:.3f} - {ci_high:.3f}]",
            }
        )

    # Specific Task Types (Regression, Binary, Multiclass)
    for task in ["Regression", "Binary", "Multiclass"]:
        task_df = score_df[score_df["Task Type"] == task]
        n_task = task_df["dataset"].nunique()
        header_mapping[task] = f"{task} ({n_task}/{total_datasets})"
        for variant, grp in task_df.groupby("Variant"):
            mean_val, ci_low, ci_high = get_bootstrap_ci(grp["Score"].values)
            strata_data.append(
                {
                    "Variant": variant,
                    "Strata": task,
                    "Value": f"{mean_val:.3f} [{ci_low:.3f} - {ci_high:.3f}]",
                }
            )

    results_long = pd.DataFrame(strata_data)

    # 5. Pivot to Leaderboard format
    leaderboard = results_long.pivot(index="Variant", columns="Strata", values="Value")

    # Reorder columns logically before renaming
    leaderboard = leaderboard.reindex(columns=STRATA_ORDER)

    # Apply dynamic column names with dataset counts
    leaderboard = leaderboard.rename(columns=header_mapping)
    leaderboard = leaderboard.reindex(VARIANT_ORDER)

    # 6. Apply Medals per column
    for col in leaderboard.columns:
        leaderboard[col] = apply_medals(leaderboard[col])

    # Reset index to make 'Strategy' a standard column
    leaderboard = leaderboard.reset_index()
    leaderboard = leaderboard.rename(columns={"Variant": "Strategy"})
    leaderboard.columns.name = None

    # 7. Save
    leaderboard.to_csv(OUTPUT_FILE, index=False)
    print(f"Medium-scale leaderboard successfully saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    np.random.seed(42)  # Ensure reproducible bootstrap CIs
    main()
