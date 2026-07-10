# To collapse the multi-fold, multi-configuration benchmark matrix into a single line per dataset without selection bias or data leakage, this script implements a strict validation-driven audition process. For each dataset, the results are grouped by hyperparameter configuration ("method"), and the average "metric_error_val" (Validation Error) is calculated across all folds. The single configuration that minimizes this cross-validated validation error is crowned the global winner, simulating how an optimal model is selected in a real-world pipeline. Finally, to report an honest performance grade on completely unseen data, the script extracts the "metric_error" (Test Error) for ONLY that winning configuration, averages it across the folds, and writes it as the final score. This separates the hyperparameter selection phase from the final evaluation phase, avoiding the penalization of wide search spaces while ensuring a mathematically sound, peer-review-ready benchmark foundation.

from pathlib import Path
import pandas as pd
from tabarena.nips2025_utils.artifacts import tabarena_method_metadata_collection

# Models to download
MODELS = ["TabPFN-3", "XGBoost", "LightGBM", "CatBoost"]

for model_name in MODELS:
    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"{'='*60}")

    # Get model metadata
    method_metadata = tabarena_method_metadata_collection.get_method_metadata(
        method=model_name
    )

    # Download the results
    print(f"Downloading {model_name} results...")
    method_metadata.method_downloader(verbose=True).download_results()

    # Load the results
    print(f"Loading {model_name} results...")
    df_results = method_metadata.load_model_results()

    # Save all results to CSV
    output_file = Path(f"{model_name.lower()}_all_results.csv")
    df_results.to_csv(output_file, index=False)
    print(f"✓ All results saved to: {output_file}")

    # See all available datasets
    print(f"\nAvailable datasets for {model_name}:")
    all_datasets = df_results["dataset"].unique()
    print(f"Total datasets: {len(all_datasets)}\n")

    # Save results per dataset to separate files
    output_dir = Path(f"{model_name.lower()}_results_per_dataset")
    output_dir.mkdir(exist_ok=True)

    for dataset in sorted(all_datasets):
        df_dataset = df_results[df_results["dataset"] == dataset]
        dataset_file = output_dir / f"{dataset}.csv"
        df_dataset.to_csv(dataset_file, index=False)
        print(f"✓ {dataset}: {len(df_dataset)} rows -> {dataset_file}")

    print(f"\n✓ All per-dataset files saved to: {output_dir}")

    # Create summary: Global Best-Tuned results across folds for each dataset
    print(f"\n\nCreating tuned summary for {model_name}...")
    summary_data = []

    for dataset in sorted(all_datasets):
        df_dataset = df_results[df_results["dataset"] == dataset]

        # 1. Group by hyperparameter configuration ("method") and average across the folds
        df_configs = (
            df_dataset.groupby("method")
            .agg(
                {
                    "metric_error_val": "mean",  # Use Validation error to SELECT the best model
                    "metric_error": "mean",  # The actual Test error we want to report
                    "time_train_s": "mean",
                    "time_infer_s": "mean",
                    "fold": "nunique",  # Just to track how many folds were used
                }
            )
            .reset_index()
        )

        # 2. Find the config with the LOWEST average validation error
        # (TabArena metric_error is structured so lower is always better)
        best_idx = df_configs["metric_error_val"].idxmin()
        best_config = df_configs.loc[best_idx]

        # 3. Save the test metrics for ONLY that winning configuration
        avg_row = {
            "dataset": dataset,
            "model": model_name,  # Added explicitly to allow clean master joining
            "num_folds": best_config["fold"],
            "total_configs_searched": df_dataset["method"].nunique(),
            "winning_config_name": best_config["method"],
            "metric_error_test_tuned": best_config["metric_error"],
            "metric_error_val_tuned": best_config["metric_error_val"],
            "time_train_s_mean": best_config["time_train_s"],
            "time_infer_s_mean": best_config["time_infer_s"],
            "problem_type": df_dataset["problem_type"].iloc[0],
        }
        summary_data.append(avg_row)

    df_summary = pd.DataFrame(summary_data)
    summary_file = Path(f"{model_name.lower()}_summary_tuned_by_dataset.csv")
    df_summary.to_csv(summary_file, index=False)
    print(f"✓ Tuned summary saved to: {summary_file}")


# =====================================================================
# ADDED SECTION: UNIFIED MASTER AGGREGATOR
# =====================================================================
print(f"\n\n{'='*60}")
print("Generating Cross-Model Master Aggregate Summary...")
print(f"{'='*60}")

all_summaries = []

# 1. Load the tuned summaries we just created for each framework
for model_name in MODELS:
    summary_file = Path(f"{model_name.lower()}_summary_tuned_by_dataset.csv")
    if summary_file.exists():
        df_model_summary = pd.read_csv(summary_file)
        all_summaries.append(df_model_summary)

if all_summaries:
    # 2. Stack them vertically into one large matrix
    df_master = pd.concat(all_summaries, ignore_index=True)

    # 3. Enforce a strict categorical sorting order on the model column
    model_order = ["CatBoost", "LightGBM", "XGBoost", "TabPFN-3"]
    df_master["model"] = pd.Categorical(
        df_master["model"], categories=model_order, ordered=True
    )

    # 4. Double sort: First alphabetize by dataset, then by your custom model hierarchy
    df_master = df_master.sort_values(
        by=["dataset", "model"], ascending=[True, True]
    ).reset_index(drop=True)

    # 5. Save the final matrix out to the directory root
    master_output_file = Path("all_models_aggregate_summary.csv")
    df_master.to_csv(master_output_file, index=False)

    print(f"✓ Master cross-model summary saved to: {master_output_file}")
    print(f"✓ Final Matrix Shape: {df_master.shape} (4 architecture rows per dataset)")
    print(f"\nMaster Summary Preview (First 8 lines):")
    print(df_master.head(8).to_string())
else:
    print("⚠ Error: No individual model summary files were found to aggregate.")

print(f"\n\n{'='*60}")
print("✓ Pipeline completely executed and compiled!")
print(f"{'='*60}")
