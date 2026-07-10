"""
==============================================================================
Script: generate_latex_table.py

Description:
Reads the Rank and Normalized Score CSVs, extracts strictly the main numeric
results, dynamically computes the 1st, 2nd, and 3rd best performing
configurations per column, and exports a ready-to-compile LaTeX table
directly to the tex folder.
==============================================================================
"""

import pandas as pd
import re
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RANKS_CSV = Path("../leaderboards/leaderboard_rank.csv")
NORM_SCORES_CSV = Path("outputs/norm_scores_alternative_individual.csv")
OUTPUT_TEX = Path("tex/configuration_results.tex")

# Map raw strategy strings to their LaTeX representations
STRATEGY_TEX_MAP = {
    "baseline_tabpfn": r"$\text{TabPFN}_{\text{baseline}}$",
    "combined_lightgbm": r"$\text{LGBM}_{\text{raw+embed}}$",
    "combined_catboost": r"$\text{CatBoost}_{\text{raw+embed}}$",
    "combined_xgboost": r"$\text{XGBoost}_{\text{raw+embed}}$",
    "baseline_lightgbm": r"$\text{LGBM}_{\text{baseline}}$",
    "baseline_catboost": r"$\text{CatBoost}_{\text{baseline}}$",
    "embed-only_catboost": r"$\text{CatBoost}_{\text{embed-only}}$",
    "embed-only_lightgbm": r"$\text{LGBM}_{\text{embed-only}}$",
    "embed-only_xgboost": r"$\text{XGBoost}_{\text{embed-only}}$",
    "baseline_xgboost": r"$\text{XGBoost}_{\text{baseline}}$",
}

# The expected column order for the output table
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


def extract_main_number(val):
    """Extracts strictly the main float number, discarding CIs, std devs, and old rankings."""
    if pd.isna(val):
        return "-"

    val_str = str(val)
    # Match the first floating point number
    match = re.search(r"(\d+\.\d+)", val_str)
    if match:
        return match.group(1)
    return "-"


def apply_dynamic_rankings(df, target_columns, is_ascending):
    """Calculates the top 3 unique values per column and appends the textual rank."""
    df_out = df.copy()

    for col in target_columns:
        valid_floats = []
        for val in df_out[col]:
            try:
                valid_floats.append(float(val))
            except ValueError:
                pass

        if not valid_floats:
            continue

        # Get unique sorted values (Ascending for Rank, Descending for Norm Score)
        unique_vals = sorted(list(set(valid_floats)), reverse=not is_ascending)
        top_3 = unique_vals[:3]

        # Build a mapping of float -> Suffix string
        suffix_map = {}
        if len(top_3) > 0:
            suffix_map[top_3[0]] = " [1st]"
        if len(top_3) > 1:
            suffix_map[top_3[1]] = " [2nd]"
        if len(top_3) > 2:
            suffix_map[top_3[2]] = " [3rd]"

        def add_suffix(val_str):
            try:
                val_float = float(val_str)
                for target_val, suffix in suffix_map.items():
                    # Use a tiny epsilon for safe float comparison
                    if abs(val_float - target_val) < 1e-9:
                        return f"{val_str}{suffix}"
            except ValueError:
                pass
            return val_str

        df_out[col] = df_out[col].apply(add_suffix)

    return df_out


def generate_table_rows(df, is_ascending):
    """Formats the DataFrame rows into LaTeX table syntax."""
    # Ensure all target columns exist, fill missing with '-'
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = "-"

    # Clean the cells to extract only the main numbers
    for col in TARGET_COLUMNS:
        df[col] = df[col].apply(extract_main_number)

    # Sort the dataframe based on the Overall column
    def sort_helper(x):
        try:
            return float(x)
        except ValueError:
            return float("inf") if is_ascending else float("-inf")

    df["sort_val"] = df["Overall"].apply(sort_helper)
    df = df.sort_values(by="sort_val", ascending=is_ascending).drop(
        columns=["sort_val"]
    )

    # Apply the dynamic [1st], [2nd], [3rd] tags as suffixes
    df = apply_dynamic_rankings(df, TARGET_COLUMNS, is_ascending)

    strategy_col = df.columns[0]
    rows = []

    for _, row in df.iterrows():
        raw_strat = row[strategy_col]
        tex_strat = STRATEGY_TEX_MAP.get(raw_strat, raw_strat.replace("_", r"\_"))

        cells = [tex_strat]
        for col in TARGET_COLUMNS:
            cells.append(str(row[col]))

        rows.append(" & ".join(cells) + r" \\")

    return "\n".join(rows)


def main():
    if not RANKS_CSV.exists():
        print(f"[ERROR] Missing Rank file: {RANKS_CSV}")
        return
    if not NORM_SCORES_CSV.exists():
        print(f"[ERROR] Missing Norm Score file: {NORM_SCORES_CSV}")
        return

    # Load data
    df_rank = pd.read_csv(RANKS_CSV)
    df_norm = pd.read_csv(NORM_SCORES_CSV)

    # Generate the respective sections
    # Rank is sorted Ascending (lower is better)
    rank_rows_latex = generate_table_rows(df_rank, is_ascending=True)
    # Norm Score is sorted Descending (higher is better)
    norm_rows_latex = generate_table_rows(df_norm, is_ascending=False)

    # Construct the full LaTeX document string
    latex_output = f"""\\begin{{table}}[ht]
\\centering
\\caption{{Configuration-level performance breakdown stratified by task type and dataset scale. Suffixes highlight the top three performing configurations in each column.}}
\\label{{tab:results-1}}
\\resizebox{{\\textwidth}}{{!}}{{%\n\\begin{{tabular}}{{lccccccc}}
\\toprule
Configuration & Overall & Classification & Regression & Binary & Multiclass & Small & Medium \\\\
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

    # Create the directory if it does not exist
    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)

    # Write the output to the file
    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(latex_output)

    print("\n" + "=" * 80)
    print(f" [SUCCESS] Dynamic Ranked LaTeX table successfully saved to: {OUTPUT_TEX}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
