"""
This script computes a consolidated multi-split podium matrix across all
dataset characteristics (Global, Task Type, and Size) using
'./summary_matrix_refactored.csv'. Strategies are ranked dynamically
by 1st place finishes (with 2nd place as a tie-breaker) within each segment.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_CSV = "./summary_matrix_refactored.csv"
OUTPUT_CSV = "./outputs/podium.csv"
# ==============================================================================

# Metrics where a lower value indicates better performance
LOWER_IS_BETTER = ["1-auroc", "log_loss", "rmse", "mae", "mse", "error"]


def determine_strategy(row):
    """Identifies the architectural strategy based on mode/algorithm columns."""
    mode = str(row["mode"]).lower().strip()
    algo = str(row["algorithm"]).lower().strip()

    if "tabpfn" in algo or "tabpfn" in mode:
        return "tabpfn_baseline"
    elif "combined" in mode:
        return "combined"
    elif "embed-only" in mode or "embed_only" in mode:
        return "embed-only"
    elif "raw-only" in mode or "raw_only" in mode or "raw" == mode:
        return "raw-only"
    return None


def main():
    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        print(f"Error: Base file not found at {INPUT_CSV}")
        return

    df = pd.read_csv(input_path)

    # Clean and parse strategies
    df["Strategy"] = df.apply(determine_strategy, axis=1)
    df = df[df["Strategy"].notna()].copy()

    # Clean and parse dataset traits uniform casing
    df["Task Type"] = df["task_type"].str.capitalize()

    # Dynamically map Dataset Size based on row count thresholds (<10k rows vs >=10k rows)
    # Search for row/instance count columns
    row_col = None
    for col in df.columns:
        if (
            any(x in col.lower() for x in ["instance", "sample", "row"])
            and "count" in col.lower()
        ):
            if df[col].dtype in [np.int64, np.float64]:
                row_col = col
                break

    if row_col:
        df["Size"] = np.where(df[row_col] < 10000, "Small", "Medium")
    else:
        # Fallback to checking pre-existing string columns if row count numbers aren't present
        size_col = next((c for c in df.columns if "size" in c.lower()), None)
        if size_col:
            if df[size_col].dtype in [np.int64, np.float64]:
                df["Size"] = np.where(df[size_col] < 10000, "Small", "Medium")
            else:
                df["Size"] = df[size_col].str.capitalize()
        else:
            df["Size"] = "Unknown"

    # Compute absolute best scores and assign ranks per dataset execution block
    rows = []
    for (dataset, metric), sub_df in df.groupby(["dataset", "primary_metric"]):
        is_lower = str(metric).lower() in LOWER_IS_BETTER

        task = sub_df["Task Type"].iloc[0]
        d_size = sub_df["Size"].iloc[0]

        p_scores = {}
        for strategy in ["combined", "tabpfn_baseline", "raw-only", "embed-only"]:
            p_df = sub_df[sub_df["Strategy"] == strategy]
            if p_df.empty:
                continue

            val = (
                p_df["eval_primary_value"].min()
                if is_lower
                else p_df["eval_primary_value"].max()
            )
            p_scores[strategy] = val

        if len(p_scores) == 0:
            continue

        score_series = pd.Series(p_scores)
        ranks = score_series.rank(ascending=is_lower, method="min")

        for strategy, rank in ranks.items():
            rows.append(
                {
                    "dataset": dataset,
                    "Task Type": task,
                    "Size": d_size,
                    "Strategy": strategy,
                    "Rank": int(rank),
                }
            )

    rank_df = pd.DataFrame(rows)
    total_datasets = rank_df["dataset"].nunique()

    # Define the vertical evaluation segments
    segment_configs = [
        ("Global Baseline", None, None, f"All ({total_datasets})"),
        ("Task Type", "Task Type", "Binary", None),
        ("Task Type", "Task Type", "Multiclass", None),
        ("Task Type", "Task Type", "Regression", None),
        ("Dataset Size", "Size", "Small", None),
        ("Dataset Size", "Size", "Medium", None),
    ]

    consolidated_rows = []

    for char_name, filter_col, filter_val, custom_label in segment_configs:
        if filter_col is None:
            sub_rank_df = rank_df.copy()
            subset_label = custom_label
        else:
            sub_rank_df = rank_df[rank_df[filter_col] == filter_val].copy()
            sub_count = sub_rank_df["dataset"].nunique()
            if sub_count == 0:
                continue
            subset_label = f"{filter_val} ({sub_count})"

        # Tally placements for this subset
        strategies = ["combined", "tabpfn_baseline", "raw-only", "embed-only"]
        tally = {s: {1: 0, 2: 0, 3: 0, 4: 0} for s in strategies}

        for _, r in sub_rank_df.iterrows():
            s = r["Strategy"]
            rk = r["Rank"]
            if rk in tally[s]:
                tally[s][rk] += 1

        # Build raw numerical tracking rows for internal sorting
        subset_rows = []
        for s in strategies:
            subset_rows.append(
                {
                    "Dataset Characteristic": char_name,
                    "Subset": subset_label,
                    "Strategy": s,
                    "1st": tally[s][1],
                    "2nd": tally[s][2],
                    "3rd": tally[s][3],
                    "4th": tally[s][4],
                }
            )

        # Sort rows within segment: 1st place descending, then 2nd place descending as tie break
        subset_rows.sort(
            key=lambda x: (x["1st"], x["2nd"], x["3rd"], x["4th"]), reverse=True
        )

        # Format strings and map to final presentation list
        for sr in subset_rows:
            consolidated_rows.append(
                {
                    "Dataset Characteristic": sr["Dataset Characteristic"],
                    "Subset": sr["Subset"],
                    "Strategy": sr["Strategy"],
                    "1st Place (Best Score)": f"{sr['1st']} datasets",
                    "2nd Place": f"{sr['2nd']} datasets",
                    "3rd Place": f"{sr['3rd']} datasets",
                    "4th Place (Worst Score)": f"{sr['4th']} datasets",
                }
            )

    output_df = pd.DataFrame(consolidated_rows)

    # Save output
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_CSV, index=False)

    print(
        "\n========================================================================================================="
    )
    print(
        "                                   CONSOLIDATED STRATEGY PODIUM MATRIX                                    "
    )
    print(
        "========================================================================================================="
    )
    print(output_df.to_string(index=False))
    print(
        "=========================================================================================================\n"
    )
    print(f"Consolidated matrix successfully saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
