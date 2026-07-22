"""
==============================================================================
Script: compile_and_aggregate_paradigms.py
Description: Centralized pipeline that:
               1. Compiles per-dataset leaderboards into a unified matrix.
               2. Computes "True" paradigm ceilings (split strategies).
               3. Computes "Merged" paradigm ceilings (hybrid pooled strategies).
             Uses vectorized bootstrapping to calculate 95% Confidence Intervals
             for the Oracle Ceilings to measure cross-dataset stability.
Outputs: All files are saved to a dedicated 'paradigm_aggregations' folder.
==============================================================================
"""

import os
import glob
import pandas as pd
import numpy as np

# =====================================================================
# PATH CONFIGURATION
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, "leaderboards/dataset_leaderboards")
)
SUMMARY_MATRIX_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "../outputs/training-consolidated/summary_matrix.csv")
)

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "paradigm_aggregations_oracle")
COMPILED_METRICS_PATH = os.path.join(OUTPUT_DIR, "compiled_dataset_metrics.csv")

# =====================================================================
# STATISTICAL CONFIGURATION
# =====================================================================
BOOTSTRAP_ITERATIONS = 1000
CONFIDENCE_LEVEL = 95
RANDOM_SEED = 42


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def format_score_ci(mean_val, ci_lower, ci_upper):
    """Formats cell strings to display Mean [Lower CI - Upper CI]."""
    return f"{mean_val:.3f} [{ci_lower:.3f} - {ci_upper:.3f}]"


def format_rank_ci(mean_val, ci_lower, ci_upper):
    """Formats rank strings to display Mean [Lower CI - Upper CI]."""
    return f"{mean_val:.2f} [{ci_lower:.2f} - {ci_upper:.2f}]"


# =====================================================================
# PHASE 1: COMPILE METRICS
# =====================================================================
def compile_metrics():
    if not os.path.exists(DATASETS_DIR):
        raise FileNotFoundError(
            f"Critical dataset directory missing at: {DATASETS_DIR}"
        )

    csv_files = glob.glob(os.path.join(DATASETS_DIR, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATASETS_DIR}")

    print(f"Compiling metrics from {len(csv_files)} dataset leaderboards...")
    compiled_records = []

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        dataset_name = filename.replace("leaderboard_", "").replace(".csv", "")

        try:
            df = pd.read_csv(file_path)
            required_cols = ["Strategy", "Rank", "Improvability %", "Norm Score"]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"[Warning] Skipping {filename}: Missing columns {missing_cols}")
                continue

            for _, row in df.iterrows():
                compiled_records.append(
                    {
                        "dataset": dataset_name,
                        "strategy": str(row["Strategy"]).strip(),
                        "rank": float(row["Rank"]),
                        "improvability": float(
                            str(row["Improvability %"]).replace("%", "").strip()
                        ),
                        "norm_score": float(row["Norm Score"]),
                    }
                )
        except Exception as e:
            print(f"Error parsing dataset {dataset_name}: {e}")

    if not compiled_records:
        raise ValueError("No records could be compiled.")

    df_compiled = pd.DataFrame(compiled_records)
    df_compiled = df_compiled.sort_values(
        by=["dataset", "rank"], ascending=[True, True]
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_compiled.to_csv(COMPILED_METRICS_PATH, index=False)

    print(f"Unified Matrix saved to: {COMPILED_METRICS_PATH}")
    return df_compiled


# =====================================================================
# PHASE 2: PARADIGM AGGREGATION ENGINE
# =====================================================================
def build_paradigm_ceilings(df_metrics, paradigms_map, prefix_name):
    if not os.path.exists(SUMMARY_MATRIX_PATH):
        raise FileNotFoundError(f"Summary matrix missing at: {SUMMARY_MATRIX_PATH}")

    # Load and clean summary matrix metadata
    df_meta = pd.read_csv(SUMMARY_MATRIX_PATH)[
        ["dataset", "task_type", "dataset_samples_count"]
    ].drop_duplicates()
    df_meta["task_type"] = df_meta["task_type"].astype(str).str.strip().str.lower()

    df_meta["size_scale"] = np.where(
        df_meta["dataset_samples_count"] <= 10000, "Small", "Medium"
    )

    df_merged = pd.merge(df_metrics, df_meta, on="dataset", how="left")
    dataset_records = []

    # Dataset-by-Dataset Triage (The Oracle Ceiling Logic with Relative Ranking)
    for dataset, df_ds in df_merged.groupby("dataset"):
        task_type = df_ds["task_type"].iloc[0]
        size_scale = df_ds["size_scale"].iloc[0]

        scores = {}
        for p_name, p_strats in paradigms_map.items():
            sub = df_ds[df_ds["strategy"].isin(p_strats)]

            if sub.empty:
                # Failure penalty: negative score ensures it ranks last
                scores[p_name] = -1.0
            else:
                # Oracle selection: get maximum normalized score
                scores[p_name] = sub["norm_score"].max()

        # Rank paradigms relative to each other (higher norm_score = rank 1)
        score_series = pd.Series(scores)
        rank_series = score_series.rank(ascending=False, method="min")

        ds_data = {"dataset": dataset, "task_type": task_type, "size_scale": size_scale}
        for p_name in paradigms_map.keys():
            ds_data[f"{p_name}_score"] = scores[p_name]
            ds_data[f"{p_name}_rank"] = rank_series[p_name]

        dataset_records.append(ds_data)

    df_dataset_matrix = pd.DataFrame(dataset_records)

    # Category Segmentation
    categories_filters = {
        "Overall": lambda df: df,
        "Classification": lambda df: df[
            df["task_type"].isin(["binary", "multiclass", "classification"])
        ],
        "Regression": lambda df: df[df["task_type"] == "regression"],
        "Binary": lambda df: df[df["task_type"] == "binary"],
        "Multiclass": lambda df: df[df["task_type"] == "multiclass"],
        "Small": lambda df: df[df["size_scale"] == "Small"],
        "Medium": lambda df: df[df["size_scale"] == "Medium"],
    }

    paradigms = list(paradigms_map.keys())
    final_score_matrix = {cat: {} for cat in categories_filters}
    final_rank_matrix = {cat: {} for cat in categories_filters}

    # Execute Vectorized Bootstrapping
    for cat_name, filter_func in categories_filters.items():
        df_slice = filter_func(df_dataset_matrix)
        n_datasets = len(df_slice)

        if n_datasets > 0:
            # Re-initialize RNG to guarantee identical bootstrap realities across all paradigms
            rng_score = np.random.default_rng(RANDOM_SEED)
            random_indices_score = rng_score.integers(
                0, n_datasets, size=(BOOTSTRAP_ITERATIONS, n_datasets)
            )

            rng_rank = np.random.default_rng(RANDOM_SEED)
            random_indices_rank = rng_rank.integers(
                0, n_datasets, size=(BOOTSTRAP_ITERATIONS, n_datasets)
            )

        for paradigm in paradigms:
            if n_datasets == 0:
                final_score_matrix[cat_name][paradigm] = "0.000 [0.000 - 0.000]"
                final_rank_matrix[cat_name][paradigm] = "10.00 [10.00 - 10.00]"
                continue

            scores = df_slice[f"{paradigm}_score"].values
            ranks = df_slice[f"{paradigm}_rank"].values

            mean_score = scores.mean()
            mean_rank = ranks.mean()

            if n_datasets == 1:
                # Cannot bootstrap a single sample
                final_score_matrix[cat_name][paradigm] = format_score_ci(
                    mean_score, mean_score, mean_score
                )
                final_rank_matrix[cat_name][paradigm] = format_rank_ci(
                    mean_rank, mean_rank, mean_rank
                )
            else:
                # Apply fast vectorized bootstrapping
                resampled_scores = np.nanmean(scores[random_indices_score], axis=1)
                resampled_ranks = np.nanmean(ranks[random_indices_rank], axis=1)

                p_lower = (100 - CONFIDENCE_LEVEL) / 2.0
                p_upper = 100 - p_lower

                score_lower = np.percentile(resampled_scores, p_lower)
                score_upper = np.percentile(resampled_scores, p_upper)

                rank_lower = np.percentile(resampled_ranks, p_lower)
                rank_upper = np.percentile(resampled_ranks, p_upper)

                final_score_matrix[cat_name][paradigm] = format_score_ci(
                    mean_score, score_lower, score_upper
                )
                final_rank_matrix[cat_name][paradigm] = format_rank_ci(
                    mean_rank, rank_lower, rank_upper
                )

    # Format Output Rows
    score_output_rows, rank_output_rows = [], []
    for paradigm in paradigms:
        score_row = {"Architecture Paradigm": paradigm}
        rank_row = {"Architecture Paradigm": paradigm}

        for cat in categories_filters:
            score_row[cat] = final_score_matrix[cat][paradigm]
            rank_row[cat] = final_rank_matrix[cat][paradigm]

        score_output_rows.append(score_row)
        rank_output_rows.append(rank_row)

    df_final_score = (
        pd.DataFrame(score_output_rows)
        .set_index("Architecture Paradigm")
        .loc[paradigms]
        .reset_index()
    )
    df_final_rank = (
        pd.DataFrame(rank_output_rows)
        .set_index("Architecture Paradigm")
        .loc[paradigms]
        .reset_index()
    )

    # Save outputs
    out_score_path = os.path.join(OUTPUT_DIR, f"matrix_{prefix_name}_norm_scores.csv")
    out_rank_path = os.path.join(OUTPUT_DIR, f"matrix_{prefix_name}_average_ranks.csv")

    df_final_score.to_csv(out_score_path, index=False)
    df_final_rank.to_csv(out_rank_path, index=False)

    print("-" * 70)
    print(f" [{prefix_name.upper()}] PARADIGM CEILINGS CALCULATED ")
    print(f" Saved to: {out_score_path}")
    print(f" Saved to: {out_rank_path}")


# =====================================================================
# MAIN EXECUTION ROUTER
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" STARTING AGGREGATION PIPELINE ")
    print("=" * 70)

    # 1. Compile base metrics into memory
    df_compiled_metrics = compile_metrics()

    # 2. Define Paradigm Maps
    true_paradigms = {
        "tabpfn_baseline": ["baseline_tabpfn"],
        "combined_*": ["combined_lightgbm", "combined_catboost", "combined_xgboost"],
        "embed-only_*": [
            "embed-only_lightgbm",
            "embed-only_catboost",
            "embed-only_xgboost",
        ],
        "raw-only_*": ["baseline_lightgbm", "baseline_catboost", "baseline_xgboost"],
    }

    merged_paradigms = {
        "tabpfn_baseline": ["baseline_tabpfn"],
        "hybrid_*": [
            "combined_lightgbm",
            "combined_catboost",
            "combined_xgboost",
            "embed-only_lightgbm",
            "embed-only_catboost",
            "embed-only_xgboost",
        ],
        "raw-only_*": ["baseline_lightgbm", "baseline_catboost", "baseline_xgboost"],
    }

    # 3. Execute Processing Blocks
    build_paradigm_ceilings(df_compiled_metrics, true_paradigms, "split")
    build_paradigm_ceilings(df_compiled_metrics, merged_paradigms, "merged")

    print("=" * 70)
    print(" PIPELINE COMPLETE ")
    print("=" * 70)
