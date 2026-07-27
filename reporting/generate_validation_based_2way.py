"""
==============================================================================
Description:
Computes Paradigm-level Average Rank and Median-Based Normalized Scores from scratch.
Groups variants into two paradigms: TabPFN Baseline vs. Hybrid Raw+Embed (strictly
LightGBM and XGBoost). Applies validation-based selection by taking the best-performing
variant within a paradigm for each dataset. Exports both a CSV file and a
ready-to-compile LaTeX table with 95% bootstrap confidence intervals.
==============================================================================
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_CSV = SCRIPT_DIR / "summary_matrix_refactored.csv"
OUTPUT_TEX = SCRIPT_DIR / "tex" / "2way_validation-based.tex"
OUTPUT_CSV = SCRIPT_DIR / "outputs" / "2way_validation-based.csv"

VALUE_COL = "eval_primary_value"
BOOTSTRAP_ITERATIONS = 1000

# Map paradigms to their LaTeX representations
PARADIGM_TEX_MAP = {
    "tabpfn_baseline": r"$\text{TabPFN}_{\text{baseline}}$",
    "hybrid_raw_embed": r"$\text{Hybrid}_{\text{raw+embed}}$",
}

PARADIGM_ORDER = [
    "tabpfn_baseline",
    "hybrid_raw_embed",
]

# Expected column order for the output table
TARGET_COLUMNS = [
    "Overall",
    "Classification",
    "Regression",
    "Binary",
    "Multiclass",
    "Small",
    "Medium",
]
# ==============================================================================


def determine_variant(row):
    """Exact variant mapping from the established standards."""
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


def map_paradigm(variant):
    """Maps specific variants to their overarching paradigm."""
    if variant == "baseline_tabpfn":
        return "tabpfn_baseline"
    elif variant in ["combined_lightgbm", "combined_catboost"]:
        return "hybrid_raw_embed"
    return None


def get_bootstrap_ci(data, iterations=BOOTSTRAP_ITERATIONS):
    """Calculates the 95% confidence interval of the mean using bootstrapping."""
    if len(data) < 2:
        return data.mean(), data.mean(), data.mean()

    means = [
        np.mean(np.random.choice(data, size=len(data), replace=True))
        for _ in range(iterations)
    ]
    return np.mean(data), np.percentile(means, 2.5), np.percentile(means, 97.5)


def clean_cell(val):
    """Extracts the main float and the CI block."""
    if pd.isna(val) or val == "-":
        return "-"

    val_str = str(val)
    match = re.search(r"(\d+\.\d+)\s*(\[[^\]]+\])?", val_str)
    if match:
        main_val = match.group(1)
        ci_val = match.group(2)
        if ci_val:
            ci_val = re.sub(r"\s*-\s*", "--", ci_val)
            return f"{main_val} {ci_val}"
        return main_val
    return "-"


def apply_dynamic_rankings(df, target_columns, is_ascending):
    """Calculates top unique values per column and applies typographic styling."""
    df_out = df.copy()

    for col in target_columns:
        valid_floats = []
        for val in df_out[col]:
            if val == "-":
                continue
            try:
                valid_floats.append(float(str(val).split()[0]))
            except ValueError:
                pass

        if not valid_floats:
            continue

        unique_vals = sorted(list(set(valid_floats)), reverse=not is_ascending)
        top_3 = unique_vals[:3]

        format_map = {}
        if len(top_3) > 0:
            format_map[top_3[0]] = r"\textbf{{{}}}"
        if len(top_3) > 1:
            format_map[top_3[1]] = r"\underline{{{}}}"
        if len(top_3) > 2:
            format_map[top_3[2]] = r"\textit{{{}}}"

        def process_cell(val_str):
            if val_str == "-":
                return val_str

            parts = str(val_str).split(" ", 1)
            main_val = parts[0]
            ci_val = parts[1] if len(parts) > 1 else ""

            formatter = "{}"
            try:
                val_float = float(main_val)
                for target_val, fmt in format_map.items():
                    if abs(val_float - target_val) < 1e-9:
                        formatter = fmt
                        break
            except ValueError:
                pass

            formatted_main = formatter.format(main_val)

            if ci_val:
                return f"{formatted_main} {ci_val}"
            else:
                return formatted_main

        df_out[col] = df_out[col].apply(process_cell)

    return df_out


def generate_table_rows(df, is_ascending):
    """Formats the DataFrame rows into LaTeX table syntax."""
    df_work = df.copy()
    for col in TARGET_COLUMNS:
        if col not in df_work.columns:
            df_work[col] = "-"

    for col in TARGET_COLUMNS:
        df_work[col] = df_work[col].apply(clean_cell)

    def sort_helper(x):
        if x == "-":
            return float("inf") if is_ascending else float("-inf")
        try:
            return float(str(x).split()[0])
        except ValueError:
            return float("inf") if is_ascending else float("-inf")

    df_work["sort_val"] = df_work["Overall"].apply(sort_helper)
    df_work = df_work.sort_values(by="sort_val", ascending=is_ascending).drop(
        columns=["sort_val"]
    )

    df_work = apply_dynamic_rankings(df_work, TARGET_COLUMNS, is_ascending)

    paradigm_col = "Paradigm"
    rows = []

    for _, row in df_work.iterrows():
        raw_strat = row[paradigm_col]
        tex_strat = PARADIGM_TEX_MAP.get(raw_strat, raw_strat.replace("_", r"\_"))

        cells = [tex_strat]
        for col in TARGET_COLUMNS:
            cells.append(str(row[col]))

        rows.append(" & ".join(cells) + r" \\")

    return "\n".join(rows)


def build_strata_df(score_df, metric_col):
    """Helper to aggregate bootstrapped metrics across all strata."""
    strata_data = []

    def append_stats(subset_df, strata_name):
        for paradigm in PARADIGM_ORDER:
            par_data = subset_df[subset_df["Paradigm"] == paradigm]
            if len(par_data) == 0:
                continue
            mean_val, ci_low, ci_high = get_bootstrap_ci(par_data[metric_col].values)
            strata_data.append(
                {
                    "Paradigm": paradigm,
                    "Strata": strata_name,
                    "Value": f"{mean_val:.3f} [{ci_low:.3f} - {ci_high:.3f}]",
                }
            )

    append_stats(score_df, "Overall")
    append_stats(
        score_df[score_df["Task Type"].isin(["Binary", "Multiclass"])], "Classification"
    )

    for task in ["Regression", "Binary", "Multiclass"]:
        append_stats(score_df[score_df["Task Type"] == task], task)

    for scale in ["Small", "Medium"]:
        append_stats(score_df[score_df["Scale"] == scale], scale)

    df_long = pd.DataFrame(strata_data)
    df_wide = df_long.pivot(
        index="Paradigm", columns="Strata", values="Value"
    ).reset_index()
    return df_wide


def main():
    if not INPUT_CSV.exists():
        print(f"[ERROR] Missing Input file: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    # 1. Map Variants and Paradigms
    df["Variant"] = df.apply(determine_variant, axis=1)
    df["Paradigm"] = df["Variant"].apply(map_paradigm)
    df = df.dropna(subset=["Paradigm"]).copy()

    # Clean attributes
    df["Task Type"] = df["task_type"].str.lower().str.strip().str.capitalize()
    df = df[df["Task Type"].isin(["Binary", "Multiclass", "Regression"])].copy()

    df["dataset_samples_count"] = pd.to_numeric(
        df.get("dataset_samples_count", 0), errors="coerce"
    )
    df["Scale"] = np.where(df["dataset_samples_count"] <= 10000, "Small", "Medium")

    # 2. Validation-Based Selection (Minimization)
    # For each dataset and paradigm, we select the variant that achieved the lowest error
    idx = df.groupby(["dataset", "Paradigm"])[VALUE_COL].idxmin()
    df_selected = df.loc[idx].copy()

    # 3. Compute Metric (Rank and Norm Score) amongst the Paradigms
    metrics_data = []

    for (dataset, metric), sub_df in df_selected.groupby(["dataset", "primary_metric"]):
        task = sub_df["Task Type"].iloc[0]
        scale = sub_df["Scale"].iloc[0]

        raw_vals = sub_df[VALUE_COL]

        # Calculate Ranks (Lower score is better -> ascending=True)
        ranks = raw_vals.rank(ascending=True, method="average")

        # Calculate Norm Score
        topline = raw_vals.min()
        baseline = raw_vals.median()

        if topline == baseline:
            norm_scores = pd.Series(0.0, index=raw_vals.index)
            norm_scores[raw_vals == topline] = 1.0
        else:
            norm_scores = (baseline - raw_vals) / (baseline - topline)
            norm_scores = norm_scores.clip(0.0, 1.0)

        for idx_row in sub_df.index:
            metrics_data.append(
                {
                    "dataset": dataset,
                    "Paradigm": sub_df.loc[idx_row, "Paradigm"],
                    "Task Type": task,
                    "Scale": scale,
                    "Rank": ranks[idx_row],
                    "Norm_Score": norm_scores[idx_row],
                }
            )

    score_df = pd.DataFrame(metrics_data)

    # 4. Get Dataset Counts for Headers
    total = score_df["dataset"].nunique()
    task_counts = score_df.drop_duplicates(subset=["dataset"])

    reg_count = (task_counts["Task Type"] == "Regression").sum()
    bin_count = (task_counts["Task Type"] == "Binary").sum()
    mc_count = (task_counts["Task Type"] == "Multiclass").sum()
    cls_count = bin_count + mc_count

    small_count = (task_counts["Scale"] == "Small").sum()
    medium_count = (task_counts["Scale"] == "Medium").sum()

    # 5. Generate wide dataframes for LaTeX conversion and CSV export
    df_rank_wide = build_strata_df(score_df, "Rank")
    df_norm_wide = build_strata_df(score_df, "Norm_Score")

    # 6. Format and export CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_rank_csv = df_rank_wide.copy()
    df_rank_csv.insert(0, "Metric", "Average Rank")

    df_norm_csv = df_norm_wide.copy()
    df_norm_csv.insert(0, "Metric", "Normalized Score")

    df_csv = pd.concat([df_rank_csv, df_norm_csv], ignore_index=True)

    # Ensure all target columns exist for CSV
    for col in TARGET_COLUMNS:
        if col not in df_csv.columns:
            df_csv[col] = "-"

    df_csv = df_csv[["Metric", "Paradigm"] + TARGET_COLUMNS]
    df_csv.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # 7. Format to LaTeX
    rank_rows_latex = generate_table_rows(
        df_rank_wide, is_ascending=True
    )  # Rank: Lower is Better
    norm_rows_latex = generate_table_rows(
        df_norm_wide, is_ascending=False
    )  # Norm Score: Higher is Better

    # 8. Construct full LaTeX document
    latex_output = f"""\\begin{{table}}[ht]
\\centering
\\caption{{Paradigm-level performance obtained via validation-based model selection and stratified by task type and dataset scale. Best results are \\textbf{{bolded}} and second best are \\underline{{underlined}}.}}
\\label{{tab:paradigm-validation-results}}
\\resizebox{{\\textwidth}}{{!}}{{%\n\\begin{{tabular}}{{lccccccc}}
\\toprule
Paradigm & Overall & Classification $\\scriptstyle ({cls_count}/{total})$ & Regression $\\scriptstyle ({reg_count}/{total})$ & Binary $\\scriptstyle ({bin_count}/{total})$ & Multiclass $\\scriptstyle ({mc_count}/{total})$ & Small $\\scriptstyle ({small_count}/{total})$ & Medium $\\scriptstyle ({medium_count}/{total})$ \\\\
\\midrule
\\multicolumn{{8}}{{l}}{{\\textbf{{Average Rank ($\\downarrow$)}}}} \\\\
\\midrule
{rank_rows_latex}
\\midrule
\\multicolumn{{8}}{{l}}{{\\textbf{{Normalized Score ($\\uparrow$)}}}} \\\\
\\midrule
{norm_rows_latex}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}"""

    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(latex_output)

    print("\n" + "=" * 80)
    print(f" [SUCCESS] Paradigm subset CSV successfully saved to: {OUTPUT_CSV}")
    print(f" [SUCCESS] Paradigm subset LaTeX table successfully saved to: {OUTPUT_TEX}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    np.random.seed(42)  # Lock random seed for repeatable bootstrap limits
    main()
