import os
import pandas as pd
import numpy as np


def generate_full_dataset_latex_table(
    csv_filename, output_filename="tabarena_full_table.tex"
):
    # --------------------------------------------------------------------------
    # Establish Absolute Paths Based on Script Location
    # --------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_filepath = os.path.abspath(os.path.join(script_dir, csv_filename))
    output_filepath = os.path.abspath(os.path.join(script_dir, output_filename))

    if not os.path.exists(csv_filepath):
        raise FileNotFoundError(f"CSV file not found at: {csv_filepath}")

    # Read the CSV, treating '-' as NaN
    df = pd.read_csv(csv_filepath, na_values=["-"])

    # Define the independent metrics columns and their specific formatting rules
    columns_config = {
        "Samples": "comma",
        "Features": "comma",
        "Classes": "int",
        "Categorical (%)": "pct",
        "Sparsity (%)": "pct",
        "Imbalance Ratio": "float",
    }

    # Helper function to format the numeric values
    def format_val(val, fmt_type):
        if pd.isna(val):
            return "-"

        if fmt_type == "comma":
            return f"{int(val):,}"
        elif fmt_type == "int":
            return f"{int(val)}"
        elif fmt_type == "pct":
            if float(val).is_integer():
                return f"{int(val)}\\%"
            return f"{val:.2f}\\%"
        elif fmt_type == "float":
            s = (
                f"{val:.2f}".rstrip("0").rstrip(".")
                if val.is_integer()
                else f"{val:.2f}"
            )
            if s.startswith("-"):
                return f"${s}$"
            return s

    latex_rows = []

    # Iterate over every dataset row in the CSV
    for _, row in df.iterrows():
        # Escape underscores in dataset names to prevent LaTeX math mode errors
        dataset_name = str(row["Name"]).replace("_", "\\_")
        row_vals = [dataset_name]

        # Process the standard isolated columns
        for col, fmt in columns_config.items():
            if col in row.index:
                val = row[col]
                row_vals.append(format_val(val, fmt))
            else:
                row_vals.append("-")

        # Handle the combined Skewness and CV column
        skew_val = row["Target Skewness"] if "Target Skewness" in row.index else np.nan
        cv_val = (
            row["Coefficient of Variation"]
            if "Coefficient of Variation" in row.index
            else np.nan
        )

        skew_str = format_val(skew_val, "float")
        cv_str = format_val(cv_val, "float")

        # Combine if either metric exists, otherwise output standard dash
        if skew_str == "-" and cv_str == "-":
            row_vals.append("-")
        else:
            row_vals.append(f"{skew_str} / {cv_str}")

        # Join the 8 columns with the '&' separator and close the row
        row_str = " & ".join(row_vals) + " \\\\"
        latex_rows.append(row_str)

    # LaTeX Table Template tailored for individual datasets
    # Tabular structure reduced to 8 columns (lccccccc)
    latex_template = """\\begin{table}[ht]
\\centering
\\caption{Detailed profile of structural dimensions and target complexity metrics for every dataset in the TabArena benchmark.}
\\label{tab:tabarena-full-stats}
\\resizebox{\\textwidth}{!}{
\\begin{tabular}{lccccccc}
\\toprule
Dataset & Samples & Features & Classes & Categorical (\\%) & Sparsity (\\%) & Imbalance Ratio & Skewness / Coef. of Variation \\\\
\\midrule
{rows}
\\bottomrule
\\end{tabular}
}
\\end{table}"""

    # Combine rows and inject into template
    final_latex = latex_template.replace("{rows}", "\n".join(latex_rows))

    print(final_latex)

    # Save to file using the script-relative path
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(final_latex)
    print(f"\nLaTeX table successfully exported to {output_filepath}")


if __name__ == "__main__":
    generate_full_dataset_latex_table(
        "../tabarena_datasets_profiling.csv", "tabarena_datasets_profiling.tex"
    )
