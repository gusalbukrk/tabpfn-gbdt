# =====================================================================
# METRIC DESCRIPTIONS
# 1. Average Rank: The mean finishing position (1st, 2nd, 3rd) across all datasets.
# 2. Improvability %: The percentage gap between a model's error and the absolute best error.
# 3. Normalized Score: Scaled from 0.0 to 1.0 (median to best) dynamically among the 3 contenders.
# =====================================================================

import pandas as pd
import numpy as np
import sys
import os
import shutil
import matplotlib.colors as mcolors
import re
from pathlib import Path

# =====================================================================
# PATH CONFIGURATION
# =====================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "leaderboards_3way"
DATASET_LEADERBOARDS_DIR = OUTPUT_DIR / "dataset_leaderboards"

# Ensure base output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# 0. CLEANUP OLD ARTIFACTS
# =====================================================================
print("Cleaning up old 3-way leaderboard artifacts...")
artifacts = (
    list(OUTPUT_DIR.glob("leaderboard_*.csv"))
    + list(OUTPUT_DIR.glob("leaderboard_*.html"))
    + list(OUTPUT_DIR.glob("matrix_win_rate.*"))
    + list(OUTPUT_DIR.glob("matrix.csv"))
)

for f in artifacts:
    try:
        os.remove(f)
    except OSError:
        pass

if DATASET_LEADERBOARDS_DIR.exists():
    try:
        shutil.rmtree(DATASET_LEADERBOARDS_DIR)
    except OSError:
        pass

# =====================================================================
# 1. CONFIGURATION & DATA EXTRACTION
# =====================================================================
CSV_PATH = (
    SCRIPT_DIR / "../../outputs/training-consolidated/summary_matrix.csv"
).resolve()
SLIM_CSV_PATH = OUTPUT_DIR / "matrix_3way.csv"
BOOTSTRAP_ITERATIONS = 1000
CONFIDENCE_LEVEL = 95
RANDOM_SEED = 42  # For reproducible confidence intervals

print(f"\nLoading data from: {CSV_PATH}...")
df_full = pd.read_csv(CSV_PATH)

df_full["mode_algorithm"] = df_full["mode"] + "_" + df_full["algorithm"]
df_full["mode_algorithm"] = df_full["mode_algorithm"].str.replace(
    "raw-only_", "baseline_", regex=False
)

# Kept task_type and dataset_samples_count here so we can filter the leaderboards later
df_slim = df_full[
    [
        "dataset",
        "task_type",
        "dataset_samples_count",
        "mode_algorithm",
        "eval_primary_value",
    ]
].copy()

# Extract TabPFN
df_tabpfn = df_full.drop_duplicates(subset=["dataset"])[
    ["dataset", "task_type", "dataset_samples_count", "tabpfn_primary_score"]
].copy()
df_tabpfn = df_tabpfn.rename(columns={"tabpfn_primary_score": "eval_primary_value"})
df_tabpfn["mode_algorithm"] = "baseline_tabpfn"

df_slim = pd.concat([df_slim, df_tabpfn], ignore_index=True)

# =====================================================================
# --- NEW: STRICT 3-VARIANT ISOLATION FILTER ---
# =====================================================================
target_algos = [
    "baseline_tabpfn",
    "combined_lightgbm",
    "combined_catboost",
]

# Filter down to just the 3 contenders to simulate a vacuum universe
df_slim = df_slim[df_slim["mode_algorithm"].isin(target_algos)].copy()

# Define the absolute perfect universe BEFORE dropping NaNs to prevent blind spots
all_datasets = df_slim["dataset"].unique()
expected_rows = len(all_datasets) * len(target_algos)

df_slim = df_slim.dropna(subset=["eval_primary_value"])

# =====================================================================
# 2. STRICT MATRIX VALIDATION (Fails fast if runs are missing/duplicated)
# =====================================================================

# Check for duplicates
if df_slim.duplicated(subset=["dataset", "mode_algorithm"]).any():
    print("\n[CRITICAL ERROR] Duplicates found in your matrix!")
    duplicates = df_slim[
        df_slim.duplicated(subset=["dataset", "mode_algorithm"], keep=False)
    ]
    print(duplicates.to_string())
    sys.exit(1)

# Check for missing data based on the pre-calculated absolute universe
if len(df_slim) != expected_rows:
    print(
        f"\n[CRITICAL ERROR] Incomplete Matrix! Expected {expected_rows} rows, but got {len(df_slim)}."
    )

    # Identify exactly what is missing
    expected_index = pd.MultiIndex.from_product(
        [all_datasets, target_algos],
        names=["dataset", "mode_algorithm"],
    )
    actual_index = df_slim.set_index(["dataset", "mode_algorithm"]).index
    missing = expected_index.difference(actual_index)

    print("\nThe following runs are missing or failed:")
    for ds, algo in missing:
        print(f" - Dataset: {ds} | Algorithm: {algo}")
    print(
        "\n[WARNING] Proceeding with an incomplete matrix for preliminary generation..."
    )

print(
    f"Matrix validation passed: {len(all_datasets)} datasets x {len(target_algos)} contenders perfectly aligned.\n"
)

df_slim = df_slim.sort_values(by=["dataset", "mode_algorithm"]).reset_index(drop=True)
df_slim.to_csv(SLIM_CSV_PATH, index=False)


# =====================================================================
# 3, 4 & 5. ENCAPSULATED LEADERBOARD GENERATOR
# =====================================================================
def get_bootstrapped_metrics(df, metric_col, rng):
    pivot_df = df.pivot(index="dataset", columns="mode_algorithm", values=metric_col)
    algorithms = pivot_df.columns
    actual_means = pivot_df.mean().values
    n_datasets_pivot = len(pivot_df)

    matrix_values = pivot_df.values
    random_indices = rng.integers(
        0, n_datasets_pivot, size=(BOOTSTRAP_ITERATIONS, n_datasets_pivot)
    )

    # Using nanmean safely handles any floating point anomalies during vectorization
    resampled_means = np.nanmean(matrix_values[random_indices], axis=1)

    lower_bounds = np.percentile(
        resampled_means, (100 - CONFIDENCE_LEVEL) / 2.0, axis=0
    )
    upper_bounds = np.percentile(
        resampled_means, 100 - (100 - CONFIDENCE_LEVEL) / 2.0, axis=0
    )

    results = {}
    for idx, algo in enumerate(algorithms):
        results[algo] = {
            "mean": actual_means[idx],
            "ci_lower": lower_bounds[idx],
            "ci_upper": upper_bounds[idx],
        }
    return results, algorithms


def generate_leaderboard(
    df_subset, title, output_path, export_matrices_prefix=None, include_ci=True
):
    if df_subset.empty:
        print(f"Skipping {title}: No datasets found for this category.")
        return None

    df_local = df_subset.copy()

    # --- PER-DATASET METRIC TRANSFORMATION (NOW ISOLATED TO 3 MODELS) ---
    df_local["best_error"] = df_local.groupby("dataset")[
        "eval_primary_value"
    ].transform("min")
    df_local["median_error"] = df_local.groupby("dataset")[
        "eval_primary_value"
    ].transform("median")

    # A. RANK (Will scale 1 to 3)
    df_local["rank"] = df_local.groupby("dataset")["eval_primary_value"].rank(
        method="average", ascending=True
    )

    # B. IMPROVABILITY (%)
    df_local["improvability_pct"] = np.where(
        df_local["eval_primary_value"] == 0,
        0.0,
        (df_local["eval_primary_value"] - df_local["best_error"])
        / df_local["eval_primary_value"]
        * 100,
    )

    # C. NORMALIZED SCORE (Will scale 0.0 to 1.0 strictly among the 3)
    df_local["norm_score"] = np.where(
        df_local["median_error"] == df_local["best_error"],
        np.where(df_local["eval_primary_value"] == df_local["best_error"], 1.0, 0.0),
        (df_local["median_error"] - df_local["eval_primary_value"])
        / (df_local["median_error"] - df_local["best_error"]),
    )
    df_local["norm_score"] = df_local["norm_score"].clip(0.0, 1.0)

    # --- REPRODUCIBLE VECTORIZED BOOTSTRAPPING ---
    rng = np.random.default_rng(RANDOM_SEED)
    rank_stats, verified_algorithms = get_bootstrapped_metrics(df_local, "rank", rng)

    rng = np.random.default_rng(RANDOM_SEED)
    imp_stats, _ = get_bootstrapped_metrics(df_local, "improvability_pct", rng)

    rng = np.random.default_rng(RANDOM_SEED)
    norm_stats, _ = get_bootstrapped_metrics(df_local, "norm_score", rng)

    # --- LEADERBOARD GENERATION ---
    leaderboard_rows = []
    for algo in verified_algorithms:
        if include_ci:
            rank_col = "Average Rank (95% Bootstrap CI)"
            rank_val = f"{rank_stats[algo]['mean']:.2f}  [{rank_stats[algo]['ci_lower']:.2f} - {rank_stats[algo]['ci_upper']:.2f}]"
            imp_col = "Improvability % (95% Bootstrap CI)"
            imp_val = f"{imp_stats[algo]['mean']:.1f}%  [{imp_stats[algo]['ci_lower']:.1f}% - {imp_stats[algo]['ci_upper']:.1f}%]"
            norm_col = "Norm Score (95% Bootstrap CI)"
            norm_val = f"{norm_stats[algo]['mean']:.3f}  [{norm_stats[algo]['ci_lower']:.3f} - {norm_stats[algo]['ci_upper']:.3f}]"
        else:
            rank_col = "Rank"
            rank_val = f"{rank_stats[algo]['mean']:.2f}"
            imp_col = "Improvability %"
            imp_val = f"{imp_stats[algo]['mean']:.1f}%"
            norm_col = "Norm Score"
            norm_val = f"{norm_stats[algo]['mean']:.3f}"

        leaderboard_rows.append(
            {
                "Strategy": algo,
                rank_col: rank_val,
                imp_col: imp_val,
                norm_col: norm_val,
                "_sort_rank": rank_stats[algo]["mean"],
            }
        )

    df_leaderboard = pd.DataFrame(leaderboard_rows)
    df_leaderboard = df_leaderboard.sort_values(by="_sort_rank", ascending=True).drop(
        columns=["_sort_rank"]
    )
    df_leaderboard.reset_index(drop=True, inplace=True)
    df_leaderboard.index += 1

    print("\n" + "=" * 125)
    print(f" 🏆 {title} 🏆")
    print("=" * 125)
    print(df_leaderboard.to_string())
    print("=" * 125)

    df_leaderboard.to_csv(output_path, index=False)
    print(f"Leaderboard successfully saved to: {output_path}")

    return {
        "rank": rank_stats,
        "improvability_pct": imp_stats,
        "norm_score": norm_stats,
    }


# =====================================================================
# 6. EXECUTE LEADERBOARD ROUTER
# =====================================================================
print(
    f"Running {BOOTSTRAP_ITERATIONS} Bootstrap iterations for Confidence Intervals..."
)

results_collection = {}

results_collection["Overall"] = generate_leaderboard(
    df_slim,
    "3-WAY TOURNAMENT GLOBAL LEADERBOARD",
    OUTPUT_DIR / "leaderboard_overall.csv",
)

df_classification = df_slim[df_slim["task_type"].isin(["binary", "multiclass"])]
results_collection["Classification"] = generate_leaderboard(
    df_classification,
    "CLASSIFICATION (BINARY + MULTICLASS)",
    OUTPUT_DIR / "leaderboard_classification.csv",
)

df_binary = df_slim[df_slim["task_type"] == "binary"]
results_collection["Binary"] = generate_leaderboard(
    df_binary,
    "BINARY CLASSIFICATION",
    OUTPUT_DIR / "leaderboard_binary.csv",
)

df_multiclass = df_slim[df_slim["task_type"] == "multiclass"]
results_collection["Multiclass"] = generate_leaderboard(
    df_multiclass,
    "MULTICLASS CLASSIFICATION",
    OUTPUT_DIR / "leaderboard_multiclass.csv",
)

df_regression = df_slim[df_slim["task_type"] == "regression"]
results_collection["Regression"] = generate_leaderboard(
    df_regression, "REGRESSION", OUTPUT_DIR / "leaderboard_regression.csv"
)

df_small = df_slim[df_slim["dataset_samples_count"] <= 10000]
results_collection["Small"] = generate_leaderboard(
    df_small,
    "SMALL DATASETS (<= 5K SAMPLES)",
    OUTPUT_DIR / "leaderboard_small.csv",
)

df_medium = df_slim[df_slim["dataset_samples_count"] > 10000]
results_collection["Medium"] = generate_leaderboard(
    df_medium,
    "MEDIUM DATASETS (> 5K SAMPLES)",
    OUTPUT_DIR / "leaderboard_medium.csv",
)

# =====================================================================
# 7. GENERATE MACRO-SUMMARY MATRICES (WITH HTML VISUALIZATIONS)
# =====================================================================
print("\n" + "=" * 125)
print(" 📊 GENERATING MACRO-SUMMARY MATRICES & VISUAL LEADERBOARDS 📊")
print("=" * 125)


def format_strategy_name(algo_string):
    algo_lower = algo_string.lower()

    if "tabpfn" in algo_lower:
        icon, base_name = "🧠⚡", "TabPFN"
    elif "lightgbm" in algo_lower:
        icon, base_name = "🌳", "LightGBM"
    elif "xgboost" in algo_lower:
        icon, base_name = "🌳", "XGBoost"
    elif "catboost" in algo_lower:
        icon, base_name = "🌳", "CatBoost"
    else:
        icon, base_name = "⚙️", algo_string.split("_")[-1].capitalize()

    state = algo_string.split("_")[0]

    html = f"""
        <td style="background-color: #1a1b26; text-align: center; border-right: none; width: 40px;">
            <div style="background: #24283b; border-radius: 12px; padding: 2px 6px; display: inline-block; font-size: 0.9em; border: 1px solid #414868;">{icon}</div>
        </td>
        <td style="background-color: #1a1b26; border-left: none; font-weight: bold;">
            <span style="color: #7aa2f7; font-size: 1.0em;">{base_name}</span> 
            <span style="color: #a9b1d6; font-weight: 500; font-size: 0.95em;">({state}) ✔↗</span>
        </td>
    """
    return html


def get_gradient_color(val, min_val, max_val, higher_is_better):
    if pd.isna(val):
        return "#1a1b26"

    color_good = np.array(mcolors.to_rgb("#257639"))
    color_bad = np.array(mcolors.to_rgb("#837c2d"))

    if max_val == min_val:
        ratio = 1.0
    else:
        ratio = (val - min_val) / (max_val - min_val)
        if not higher_is_better:
            ratio = 1.0 - ratio

    c_out = color_bad + (color_good - color_bad) * ratio
    return mcolors.to_hex(c_out)


def export_styled_html(df_text, df_means, out_filename, metric_key):
    higher_is_better = metric_key == "norm_score"

    html_content = [
        "<html><head><meta charset='utf-8'><style>",
        "body { background-color: #1a1b26; color: #c0caf5; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; }",
        "table { border-collapse: collapse; width: 100%; border: 1px solid #414868; font-size: 14px; }",
        "th, td { border: 1px solid #414868; padding: 12px 16px; text-align: left; }",
        "th { background-color: #1a1b26; font-weight: bold; border-bottom: 2px solid #414868; }",
        ".header-group { border-bottom: 1px dotted #565f89; padding-bottom: 4px; margin-bottom: 4px; font-size: 0.85em; color: #a9b1d6; }",
        ".val-text { font-weight: bold; color: #ffffff; }",
        "</style></head><body>",
        "<table>",
        "<thead>",
        "<tr>",
        "<th colspan='2' rowspan='2'>Model</th>",
        "<th rowspan='2'>Overall</th>",
        "<th colspan='4' style='text-align: center;'><div class='header-group'>By Task</div></th>",
        "<th colspan='2' style='text-align: center;'><div class='header-group'>By Dataset Size</div></th>",
        "</tr><tr>",
        "<th>Class.</th><th>Regr.</th><th>Binary</th><th>Multi.</th>",
        "<th>Small</th><th>Medium</th>",
        "</tr>",
        "</thead><tbody>",
    ]

    limits = {}
    cols_to_color = [
        "Overall",
        "Classification",
        "Regression",
        "Binary",
        "Multiclass",
        "Small",
        "Medium",
    ]
    for col in cols_to_color:
        if col in df_means.columns:
            raw_floats = df_means[col].dropna()
            if not raw_floats.empty:
                limits[col] = (raw_floats.min(), raw_floats.max())
            else:
                limits[col] = (0, 1)

    for idx in df_text.index:
        row_text = df_text.loc[idx]
        row_mean = df_means.loc[idx]
        html_content.append("<tr>")
        html_content.append(format_strategy_name(row_text["Strategy"]))

        for col in cols_to_color:
            if col in df_text.columns:
                val_text = row_text[col]
                val_mean = row_mean[col]
                min_val, max_val = limits.get(col, (0, 1))
                bg_color = get_gradient_color(
                    val_mean, min_val, max_val, higher_is_better
                )
                html_content.append(
                    f"<td style='background-color: {bg_color}; text-align: center;'>{val_text}</td>"
                )
            else:
                html_content.append("<td style='background-color: #1a1b26;'>N/A</td>")

        html_content.append("</tr>")

    html_content.append("</tbody></table></body></html>")

    with open(out_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(html_content))


metrics_to_export = [
    ("rank", "Average Rank", "leaderboard_rank"),
    ("improvability_pct", "Improvability %", "leaderboard_improvability"),
    ("norm_score", "Normalized Score", "leaderboard_norm_score"),
]

strategies = df_slim["mode_algorithm"].unique()
slice_names = [
    "Overall",
    "Classification",
    "Regression",
    "Binary",
    "Multiclass",
    "Small",
    "Medium",
]

for metric_key, metric_name, out_name_base in metrics_to_export:
    matrix_rows = []

    for algo in strategies:
        row = {"Strategy": algo}
        for slice_name in slice_names:
            res = results_collection.get(slice_name)
            if res is not None and algo in res[metric_key]:
                row[slice_name] = res[metric_key][algo]
            else:
                row[slice_name] = np.nan
        matrix_rows.append(row)

    df_matrix = pd.DataFrame(matrix_rows)

    if "Overall" in df_matrix.columns:
        ascending_order = False if metric_key == "norm_score" else True
        df_matrix["sort_key"] = df_matrix["Overall"].apply(
            lambda x: x["mean"] if isinstance(x, dict) else np.nan
        )
        df_matrix = df_matrix.sort_values(
            by="sort_key", ascending=ascending_order
        ).drop(columns=["sort_key"])

    df_csv = pd.DataFrame({"Strategy": df_matrix["Strategy"]})
    df_html = pd.DataFrame({"Strategy": df_matrix["Strategy"]})
    df_means = pd.DataFrame({"Strategy": df_matrix["Strategy"]})

    for col in slice_names:
        if col in df_matrix.columns:
            raw_vals = (
                df_matrix[col]
                .apply(lambda x: x["mean"] if isinstance(x, dict) else np.nan)
                .dropna()
            )
            unique_vals = sorted(
                raw_vals.unique(), reverse=(metric_key == "norm_score")
            )

            top_1 = unique_vals[0] if len(unique_vals) > 0 else None
            top_2 = unique_vals[1] if len(unique_vals) > 1 else None
            top_3 = unique_vals[2] if len(unique_vals) > 2 else None

            csv_col = []
            html_col = []
            mean_col = []

            for val in df_matrix[col]:
                if not isinstance(val, dict):
                    csv_col.append("N/A")
                    html_col.append("N/A")
                    mean_col.append(np.nan)
                    continue

                mean_val = val["mean"]
                ci_low = val["ci_lower"]
                ci_up = val["ci_upper"]

                if metric_key == "rank":
                    base_str = f"{mean_val:.2f}"
                    ci_str = f"[{ci_low:.2f} - {ci_up:.2f}]"
                elif metric_key == "improvability_pct":
                    base_str = f"{mean_val:.1f}%"
                    ci_str = f"[{ci_low:.1f}% - {ci_up:.1f}%]"
                elif metric_key == "norm_score":
                    base_str = f"{mean_val:.3f}"
                    ci_str = f"[{ci_low:.3f} - {ci_up:.3f}]"

                if mean_val == top_1:
                    base_str = "🥇 " + base_str
                elif mean_val == top_2:
                    base_str = "🥈 " + base_str
                elif mean_val == top_3:
                    base_str = "🥉 " + base_str

                csv_col.append(f"{base_str} {ci_str}")
                html_col.append(
                    f"<span class='val-text'>{base_str}</span><br><span style='font-size: 0.85em; color: #a9b1d6; font-weight: normal;'>{ci_str}</span>"
                )
                mean_col.append(mean_val)

            df_csv[col] = csv_col
            df_html[col] = html_col
            df_means[col] = mean_col

    csv_filename = OUTPUT_DIR / f"{out_name_base}.csv"
    df_csv.to_csv(csv_filename, index=False)

    html_filename = OUTPUT_DIR / f"{out_name_base}.html"
    export_styled_html(df_html, df_means, html_filename, metric_key)

    print(f"Matrix ({metric_name}) exported to: {csv_filename} and {html_filename}")

# =====================================================================
# 8. GENERATE HEAD-TO-HEAD WIN RATE MATRIX (ISOLATED TO 3-WAY)
# =====================================================================
print("\n" + "=" * 125)
print(" ⚔️ GENERATING 3-WAY HEAD-TO-HEAD WIN RATE MATRIX ⚔️")
print("=" * 125)

df_pivot = df_slim.pivot(
    index="dataset", columns="mode_algorithm", values="eval_primary_value"
)

if "Overall" in results_collection and "rank" in results_collection["Overall"]:
    overall_ranks = {
        algo: stats["mean"]
        for algo, stats in results_collection["Overall"]["rank"].items()
    }
    sorted_algos = sorted(df_pivot.columns, key=lambda x: overall_ranks.get(x, 999))
else:
    sorted_algos = df_pivot.columns.tolist()

df_pivot = df_pivot[sorted_algos]
win_matrix = pd.DataFrame(index=sorted_algos, columns=sorted_algos, dtype=float)

for algo_a in sorted_algos:
    for algo_b in sorted_algos:
        if algo_a == algo_b:
            win_matrix.loc[algo_a, algo_b] = np.nan
        else:
            valid_comparisons = df_pivot[[algo_a, algo_b]].dropna()
            total = len(valid_comparisons)
            if total > 0:
                wins = (valid_comparisons[algo_a] < valid_comparisons[algo_b]).sum()
                ties = (valid_comparisons[algo_a] == valid_comparisons[algo_b]).sum()
                win_rate = ((wins + 0.5 * ties) / total) * 100
                win_matrix.loc[algo_a, algo_b] = win_rate


def get_win_rate_color(val):
    if pd.isna(val):
        return "#15161e"
    color_bad = np.array(mcolors.to_rgb("#8f3b3b"))
    color_mid = np.array(mcolors.to_rgb("#24283b"))
    color_good = np.array(mcolors.to_rgb("#257639"))
    val = max(0.0, min(100.0, val))

    if val < 50.0:
        ratio = val / 50.0
        c_out = color_bad + (color_mid - color_bad) * ratio
    else:
        ratio = (val - 50.0) / 50.0
        c_out = color_mid + (color_good - color_mid) * ratio
    return mcolors.to_hex(c_out)


def format_col_header(algo_string):
    algo_lower = algo_string.lower()
    if "tabpfn" in algo_lower:
        icon, base_name = "🧠", "TabPFN"
    elif "lightgbm" in algo_lower:
        icon, base_name = "🌳", "LightGBM"
    elif "catboost" in algo_lower:
        icon, base_name = "🌳", "CatBoost"
    else:
        icon, base_name = "⚙️", algo_string.split("_")[-1].capitalize()
    state = algo_string.split("_")[0][:4] + "."
    return f"""
        <div style="text-align: center; line-height: 1.3;">
            <span style="font-size: 1.2em;">{icon}</span><br>
            <span style="color: #7aa2f7; font-weight: bold; font-size: 1.0em;">{base_name}</span><br>
            <span style="color: #a9b1d6; font-size: 0.95em; font-weight: 500;">({state})</span>
        </div>
    """


html_content = [
    "<html><head><meta charset='utf-8'><style>",
    "body { background-color: #1a1b26; color: #c0caf5; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; }",
    "table { border-collapse: collapse; margin-top: 20px; border: 1px solid #414868; font-size: 13px; }",
    "th, td { border: 1px solid #414868; padding: 10px; text-align: center; }",
    "th { background-color: #1a1b26; font-weight: bold; }",
    ".diag { background: repeating-linear-gradient(45deg, #15161e, #15161e 10px, #1a1b26 10px, #1a1b26 20px); }",
    ".win-text { font-weight: bold; color: #ffffff; text-shadow: 1px 1px 2px rgba(0,0,0,0.8); }",
    "</style></head><body>",
    "<h2>Head-to-Head Win Rate Matrix (3-Way)</h2>",
    "<p style='color: #a9b1d6; font-size: 0.9em;'>Rows represent the challenger. Columns represent the opponent. Values represent the percentage of datasets where the Row model outperformed the Column model.</p>",
    "<table>",
    "<thead><tr>",
    "<th colspan='2' style='text-align: center; font-size: 0.9em; color: #a9b1d6;'>Challenger \\<br>Opponent</th>",
]

for col_algo in sorted_algos:
    html_content.append(f"<th style='width: 80px;'>{format_col_header(col_algo)}</th>")
html_content.append("</tr></thead><tbody>")

for row_algo in sorted_algos:
    html_content.append("<tr>")
    html_content.append(format_strategy_name(row_algo))
    for col_algo in sorted_algos:
        if row_algo == col_algo:
            html_content.append("<td class='diag'></td>")
        else:
            val = win_matrix.loc[row_algo, col_algo]
            bg_color = get_win_rate_color(val)
            html_content.append(
                f"<td style='background-color: {bg_color};'><span class='win-text'>{val:.1f}%</span></td>"
            )
    html_content.append("</tr>")

html_content.append("</tbody></table></body></html>")

win_html_path = OUTPUT_DIR / "matrix_win_rate_3way.html"
with open(win_html_path, "w", encoding="utf-8") as f:
    f.write("\n".join(html_content))

win_csv_path = OUTPUT_DIR / "matrix_win_rate_3way.csv"
win_matrix.to_csv(win_csv_path)

print(f"3-Way Win Rate Matrix exported to: {win_csv_path} and {win_html_path}")
