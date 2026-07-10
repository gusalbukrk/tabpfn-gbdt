"""
Description:
This script processes benchmark data to extract unified TreeSHAP feature
importance percentages across hybrid configurations. It re-loads configurations
via Optuna SQLite databases, re-trains the final models using the original archives,
and computes absolute mean SHAP shares. It incrementally saves three target CSV
outputs (full telemetry, dataset averages, and algorithm breakdown) as the results
roll in, ensuring progress is never lost if execution is interrupted.
"""

import pandas as pd
import numpy as np
import optuna
import shap
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import warnings
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# Suppress verbose output from external processing modules
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
# Resolve the exact directory where this script is physically located
SCRIPT_DIR = Path(__file__).resolve().parent

# Inputs
INPUT_CSV_PATH = SCRIPT_DIR / "../summary_matrix_refactored.csv"
DB_DIR = SCRIPT_DIR / "../../outputs/training-consolidated/databases"
ARCHIVE_DIR = SCRIPT_DIR / "../../archives"

# Outputs Location
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# Explicit Output Files Block
SHAP_SUMMARY_CSV = OUTPUT_DIR / "shap_summary_matrix.csv"
SHAP_DATASET_AVG_CSV = OUTPUT_DIR / "shap_dataset_average.csv"
SHAP_ALGO_BREAKDOWN_CSV = OUTPUT_DIR / "shap_algorithm_breakdown.csv"


def aggregate_shap_values(shap_vals):
    """
    Robustly aggregates SHAP values into a single 1D array of mean absolute importances,
    handling formatting variations across classification and regression regimes.
    """
    if isinstance(shap_vals, list):
        class_means = [np.abs(sv).mean(axis=0) for sv in shap_vals]
        return np.mean(class_means, axis=0)
    elif len(shap_vals.shape) == 3:
        return np.abs(shap_vals).mean(axis=0).mean(axis=1)
    else:
        return np.abs(shap_vals).mean(axis=0)


def main():
    # Ensure targeted output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading base summary matrix from {INPUT_CSV_PATH}...")
    df_full = pd.read_csv(INPUT_CSV_PATH)

    # Filter strictly for hybrid executions
    df_hybrid = df_full[df_full["mode"] == "combined"].copy()
    total_runs = len(df_hybrid)
    print(f"Found {total_runs} hybrid configurations to process.\n")

    updated_rows = []

    # Use enumerate to create a clean, sequential counter (1 to 153)
    for step, (idx, row) in enumerate(df_hybrid.iterrows(), start=1):
        dataset = row["dataset"]
        algo = row["algorithm"].lower()
        task_type = row["task_type"].lower()
        optimal_trees = int(row["optimal_trees"])

        print(
            f"[Run {step}/{total_runs}] Dataset: {dataset} | Algorithm: {algo.upper()} | Task: {task_type.upper()}"
        )

        # Locate execution footprint database matches via glob pattern
        db_matches = list(DB_DIR.glob(f"*_{dataset}_combined_{algo}.db"))
        if not db_matches:
            print(f"  [ERROR] Database missing for {dataset}_{algo}. Skipping.")
            continue
        db_path = db_matches[-1]

        npz_path = ARCHIVE_DIR / f"{dataset}.npz"
        if not npz_path.exists():
            print(f"  [ERROR] Archive missing at {npz_path}. Skipping.")
            continue

        # Extract champion hyperparameter trials
        study_name = db_path.stem
        try:
            study = optuna.load_study(
                study_name=study_name, storage=f"sqlite:///{db_path}"
            )
            best_params = study.best_params
        except Exception as e:
            print(f"  [ERROR] Study loading failed for {db_path.name}: {e}")
            continue

        # Re-ingest original archive tensors
        data = np.load(npz_path, allow_pickle=True)

        y_train_raw = data["y_train"]
        if task_type in ["binary", "multiclass"]:
            le = LabelEncoder()
            y_train = le.fit_transform(y_train_raw)
            num_classes = len(le.classes_)
        else:
            y_train = y_train_raw.astype(np.float64)
            num_classes = 1

        X_train_raw = pd.DataFrame(
            data["X_train_raw"], columns=data["feature_names"]
        ).infer_objects()
        cat_features = list(data["cat_features"])

        # Re-fit deterministic latent dimensionality layer
        pca = PCA(n_components=0.95, random_state=42)
        X_train_embed_pca = pd.DataFrame(pca.fit_transform(data["X_train_embed"]))
        X_train_embed_pca.columns = [
            f"pca_embed_{i}" for i in range(X_train_embed_pca.shape[1])
        ]

        X_train_combined = pd.concat([X_train_raw, X_train_embed_pca], axis=1)
        X_train_combined.columns = X_train_combined.columns.astype(str)

        shap_values = None

        try:
            if algo == "lightgbm":
                for col in cat_features:
                    X_train_combined[col] = X_train_combined[col].astype("category")

                obj, eval_met = (
                    ("binary", "auc")
                    if task_type == "binary"
                    else (
                        ("multiclass", "multi_logloss")
                        if task_type == "multiclass"
                        else ("regression", "rmse")
                    )
                )

                lgb_params = {
                    **best_params,
                    "objective": obj,
                    "metric": eval_met,
                    "verbosity": -1,
                    "seed": 42,
                }
                if task_type == "multiclass":
                    lgb_params["num_class"] = num_classes

                dtrain = lgb.Dataset(X_train_combined, label=y_train)
                model = lgb.train(lgb_params, dtrain, num_boost_round=optimal_trees)

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_train_combined)

            elif algo == "xgboost":
                for col in cat_features:
                    X_train_combined[col] = X_train_combined[col].astype("category")

                obj, eval_met = (
                    ("binary:logistic", "auc")
                    if task_type == "binary"
                    else (
                        ("multi:softprob", "mlogloss")
                        if task_type == "multiclass"
                        else ("reg:squarederror", "rmse")
                    )
                )

                xgb_params = {
                    **best_params,
                    "objective": obj,
                    "eval_metric": eval_met,
                    "tree_method": "hist",
                    "seed": 42,
                }
                if task_type == "multiclass":
                    xgb_params["num_class"] = num_classes

                dtrain = xgb.DMatrix(
                    X_train_combined, label=y_train, enable_categorical=True
                )
                model = xgb.train(xgb_params, dtrain, num_boost_round=optimal_trees)

                explainer = shap.TreeExplainer(model)
                # [FIXED]: XGBoost categorical bypass patch using the compiled DMatrix
                shap_values = explainer.shap_values(dtrain)

            elif algo == "catboost":
                if cat_features:
                    for col in cat_features:
                        col_name = (
                            col
                            if isinstance(col, str)
                            else X_train_combined.columns[col]
                        )
                        X_train_combined[col_name] = (
                            X_train_combined[col_name]
                            .fillna("Missing")
                            .astype(str)
                            .replace(["nan", "NaN", "<NA>", "None"], "Missing")
                        )

                train_pool = cb.Pool(
                    X_train_combined, label=y_train, cat_features=cat_features
                )
                obj = (
                    "Logloss"
                    if task_type == "binary"
                    else ("MultiClass" if task_type == "multiclass" else "RMSE")
                )
                cb_params = {
                    **best_params,
                    "iterations": optimal_trees,
                    "loss_function": obj,
                    "random_seed": 42,
                    "silent": True,
                }

                if task_type in ["binary", "multiclass"]:
                    model = cb.CatBoostClassifier(**cb_params).fit(train_pool)
                else:
                    model = cb.CatBoostRegressor(**cb_params).fit(train_pool)

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(train_pool)

        except Exception as e:
            print(f"  [ERROR] SHAP calculation crash: {e}")
            continue

        if shap_values is not None:
            abs_importance = aggregate_shap_values(shap_values)
            total_shap = abs_importance.sum()
            scaled_importance = (
                (abs_importance / total_shap) * 100
                if total_shap > 0
                else np.zeros_like(abs_importance)
            )

            features = X_train_combined.columns
            embed_share = sum(
                score
                for feat, score in zip(features, scaled_importance)
                if "pca_embed" in feat
            )
            raw_share = sum(
                score
                for feat, score in zip(features, scaled_importance)
                if "pca_embed" not in feat
            )

            row_dict = row.to_dict()
            row_dict["feat_share_embedded_pct"] = embed_share
            row_dict["feat_share_raw_pct"] = raw_share
            row_dict["importance_metric_used"] = "TreeSHAP"

            updated_rows.append(row_dict)
            print(
                f"  [OK] TreeSHAP Extracted: Embed={embed_share:.1f}% | Raw={raw_share:.1f}%"
            )

            # ==============================================================================
            # INCREMENTAL DISK ARTIFACT PIPELINE (SAVING AS RESULTS ROLL IN)
            # ==============================================================================
            df_new = pd.DataFrame(updated_rows)

            # Enforce consistent capitalization matching visualization definitions
            df_new["algorithm"] = df_new["algorithm"].str.capitalize()
            df_new["task_type"] = df_new["task_type"].str.capitalize()
            df_new["size_scale"] = np.where(
                df_new["dataset_samples_count"] <= 10000, "Small", "Medium"
            )

            # 1. Export Complete Consolidated Summary Matrix
            df_new.to_csv(SHAP_SUMMARY_CSV, index=False)

            # 2. Export Micro-Averaged Dataset Significance Proportions
            df_dataset_avg = (
                df_new.groupby("dataset")[
                    ["feat_share_embedded_pct", "feat_share_raw_pct"]
                ]
                .mean()
                .reset_index()
            )
            meta = df_new[["dataset", "task_type", "size_scale"]].drop_duplicates()
            df_dataset_avg = pd.merge(df_dataset_avg, meta, on="dataset", how="left")
            df_dataset_avg.to_csv(SHAP_DATASET_AVG_CSV, index=False)

            # 3. Export Streamlined Model Architecture Breakdown Matrix
            df_algo_breakdown = df_new[
                [
                    "dataset",
                    "algorithm",
                    "task_type",
                    "size_scale",
                    "feat_share_embedded_pct",
                    "feat_share_raw_pct",
                ]
            ]
            df_algo_breakdown.to_csv(SHAP_ALGO_BREAKDOWN_CSV, index=False)

    print("\n" + "=" * 70)
    print(
        f" [SUCCESS] PIPELINE COMPLETE. TRIPLE ARTIFACT MATRICES DEPLOYED IN: {OUTPUT_DIR.resolve()}"
    )
    print("=" * 70)
    print(f"1. Telemetry Aggregate Matrix -> {SHAP_SUMMARY_CSV.name}")
    print(f"2. Dataset Averages Matrix     -> {SHAP_DATASET_AVG_CSV.name}")
    print(f"3. Algorithm Framework Matrix  -> {SHAP_ALGO_BREAKDOWN_CSV.name}\n")


if __name__ == "__main__":
    main()
