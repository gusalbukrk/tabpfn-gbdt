from pathlib import Path
import pandas as pd

# =====================================================================
# PATH CONFIGURATION (Relative to this script file)
# =====================================================================
SCRIPT_DIR = Path(__file__).parent
CSV_RELATIVE_PATH = SCRIPT_DIR / "../leaderboards/matrix_win_rate.csv"
OUTPUT_TEX_PATH = SCRIPT_DIR / "winrate.tex"
# =====================================================================

# Explicit mapping from CSV strings to your exact LaTeX formatted names
NAME_MAPPING = {
    "baseline_tabpfn": r"$\text{TabPFN}_{\text{baseline}}$",
    "combined_lightgbm": r"$\text{LGBM}_{\text{raw+embed}}$",
    "combined_catboost": r"$\text{CatB}_{\text{raw+embed}}$",
    "combined_xgboost": r"$\text{XGB}_{\text{raw+embed}}$",
    "baseline_lightgbm": r"$\text{LGBM}_{\text{baseline}}$",
    "baseline_catboost": r"$\text{CatB}_{\text{baseline}}$",
    "embed-only_catboost": r"$\text{CatB}_{\text{embed-only}}$",
    "embed-only_lightgbm": r"$\text{LGBM}_{\text{embed-only}}$",
    "embed-only_xgboost": r"$\text{XGB}_{\text{embed-only}}$",
    "baseline_xgboost": r"$\text{XGB}_{\text{baseline}}$",
}


def generate_latex_matrix(file_path):
    df = pd.read_csv(file_path, index_col=0)

    # Drop completely empty rows or columns
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # Map indices and columns to LaTeX math strings using the mapping dictionary
    df.index = [NAME_MAPPING.get(x.strip(), x) for x in df.index]
    df.columns = [NAME_MAPPING.get(x.strip(), x) for x in df.columns]

    latex_lines = []
    latex_lines.append(r"\begin{table}[ht]")
    latex_lines.append(r"\centering")
    latex_lines.append(
        r"\caption{Pairwise win rate matrix across all evaluated configurations.}"
    )
    latex_lines.append(r"\label{tab:win-rate}")

    latex_lines.append(r"\resizebox{\textwidth}{!}{")
    # Changed columns to centered 'c' to keep values nicely aligned under headers
    latex_lines.append(r"\begin{tabular}{l" + "c" * len(df.columns) + "}")
    latex_lines.append(r"\toprule")

    # Empty top-left cell header match
    headers = [" "] + list(df.columns)
    latex_lines.append(" & ".join(headers) + r" \\")
    latex_lines.append(r"\midrule")

    for idx, row in df.iterrows():
        row_elements = [idx]
        for val in row:
            if pd.isna(val) or val == "" or str(val).strip() == "":
                # Using multicolumn to force exact center alignment for the dash
                row_elements.append(r"\multicolumn{1}{c}{-}")
            else:
                row_elements.append(f"{float(val):.2f}\%")
        latex_lines.append(" & ".join(row_elements) + r" \\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"}")
    latex_lines.append(r"\end{table}")

    return "\n".join(latex_lines)


if __name__ == "__main__":
    if not CSV_RELATIVE_PATH.exists():
        print(f"Error: Target CSV file not found at: {CSV_RELATIVE_PATH}")
    else:
        latex_output = generate_latex_matrix(CSV_RELATIVE_PATH)
        OUTPUT_TEX_PATH.write_text(latex_output, encoding="utf-8")
        print(f"Successfully generated and saved LaTeX table to: {OUTPUT_TEX_PATH}")
