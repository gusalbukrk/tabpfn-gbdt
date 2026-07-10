# =====================================================================
# 1. INITIAL SYSTEM IMPORTS & CONFIGURATION
# =====================================================================
import argparse
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import optuna

# We are not using the native WeightsAndBiasesCallback because it forces a timeline mismatch
# on Trial 0 (which has no hyperparameters) and incorrectly overwrites the run's final summary
# values with the parameters of the last trial rather than the best trial.
import pandas as pd
import os
import sys
import time
import json
from datetime import datetime
import logging
import wandb
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    root_mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder

global_start_time = time.time()

# =====================================================================
# 1. COMMAND-LINE ROUTER ARGUMENTS
# =====================================================================
parser = argparse.ArgumentParser(description="TabArena Hybrid Matrix Benchmark Suite")
parser.add_argument(
    "--archive-path",
    type=str,
    required=True,
    help="Path to the target self-contained .npz data archive",
)
parser.add_argument(
    "--algo",
    type=str,
    required=True,
    choices=["catboost", "lightgbm", "xgboost"],
    help="Target gradient boosting algorithm engine to tune and evaluate",
)
parser.add_argument(
    "--mode",
    type=str,
    default="combined",
    choices=["combined", "raw-only", "embed-only", "embed-only-no-pca"],
    help="Execution mode: combined, raw-only, embed-only (PCA compressed), or embed-only-no-pca",
)
parser.add_argument(
    "--trials",
    type=int,
    default=25,  # Adjusted default to match TabArena-Lite recommendation
    help="Number of optimization trials for Optuna (default: 25)",
)
parser.add_argument(
    "--session-id",
    type=str,
    default="standalone",
    help="Parent directory name to encapsulate this execution's outputs (Defaults to 'standalone')",
)
parser.add_argument(
    "--use-wandb",
    action="store_true",
    help="Flag to enable Weights & Biases integration for experiment tracking",
)
args = parser.parse_args()

ARCHIVE_PATH = args.archive_path
ALGO = args.algo.lower()
MODE = args.mode.lower()
TRIALS = args.trials + 1  # plus 1 is the trial w/ the default hyperparameters
SESSION_ID = args.session_id
USE_WANDB = args.use_wandb

DATASET_NAME = os.path.splitext(os.path.basename(ARCHIVE_PATH))[0]
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- EARLY DATA LOAD TO EXTRACT METADATA FOR CLEAN FILENAMES ---
data = np.load(ARCHIVE_PATH, allow_pickle=True)

# Safely extract root dataset name
try:
    root_dataset = str(data["dataset_name"].item())
except KeyError:
    root_dataset = DATASET_NAME

# Construct unified file prefix (e.g., 20260621_churn_combined_lightgbm)
FILE_PREFIX = f"{TIMESTAMP}_{root_dataset}_{MODE}_{ALGO}"

# W&B specific Run Name (Clean, without timestamp)
WANDB_RUN_NAME = f"{root_dataset}_{MODE}_{ALGO}"

# --- CENTRALIZED OUTPUT HIERARCHY ---
BASE_OUT_DIR = os.path.join("outputs", SESSION_ID)
LOGS_DIR = os.path.join(BASE_OUT_DIR, "logs")
DBS_DIR = os.path.join(BASE_OUT_DIR, "databases")
IMPORTANCE_DIR = os.path.join(BASE_OUT_DIR, "importance")
METRICS_DIR = os.path.join(BASE_OUT_DIR, "metrics")
TRIALS_DIR = os.path.join(BASE_OUT_DIR, "trials")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DBS_DIR, exist_ok=True)
os.makedirs(IMPORTANCE_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(TRIALS_DIR, exist_ok=True)

# Set up logging to route exactly to the session log folder using new prefix
log_path = os.path.join(LOGS_DIR, f"{FILE_PREFIX}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
)
print = logging.info
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()

# --- W&B INITIALIZATION & CRASH HANDLER ---
if USE_WANDB:
    wandb.init(
        entity="gusalbukrk-team",
        project="tabarena-benchmark",
        group=f"session-{SESSION_ID}",
        name=WANDB_RUN_NAME,
        job_type="tuning-job",
    )
    print(
        f"[W&B] Child process started for job: {WANDB_RUN_NAME} in group session-{SESSION_ID}"
    )

    # Explicitly prevent intermediate trial parameters from polluting the final summary
    wandb.define_metric("trial_params.*", summary="none")
    wandb.define_metric("optimization_score", summary="none")

    # Exception hook ensures W&B catches unexpected crashes (OOM, Data Errors) cleanly
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.error(
            "Uncaught Pipeline Exception", exc_info=(exc_type, exc_value, exc_traceback)
        )
        wandb.finish(exit_code=1)

    sys.excepthook = handle_exception

# =====================================================================
# 2. SHARED DATA INGESTION & METADATA RESOLUTION
# =====================================================================
print(
    f"Loading archive '{ARCHIVE_PATH}' and initializing pipeline for target: {ALGO.upper()}..."
)
print(f"Execution Mode: {MODE.upper()}")
print(f"Session Output Target: {BASE_OUT_DIR}/")

y_train_raw = data["y_train"]
y_test_raw = data["y_test"]

TASK = str(data["task_type"].item())
print(f"Task Type: {TASK.upper()}")

# Extract TabPFN Baselines for Delta Calculation
try:
    base_p = float(data["baseline_primary_metric_val"].item())
    base_s = float(data["baseline_secondary_metric_val"].item())
    print(f"TabPFN Baselines Loaded - Primary: {base_p:.4f} | Secondary: {base_s:.4f}")
except Exception:
    base_p = None
    base_s = None
    print("TabPFN Baselines not found in archive. Deltas will not be calculated.")

if TASK in ["binary", "multiclass"]:
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    num_classes = len(le.classes_)
else:
    y_train = y_train_raw.astype(np.float64)
    y_test = y_test_raw.astype(np.float64)
    num_classes = 1

X_train_raw = pd.DataFrame(
    data["X_train_raw"], columns=data["feature_names"]
).infer_objects()
X_test_raw = pd.DataFrame(
    data["X_test_raw"], columns=data["feature_names"]
).infer_objects()
cat_features = list(data["cat_features"])

if ALGO in ["lightgbm", "xgboost"]:
    for col in cat_features:
        X_train_raw[col] = X_train_raw[col].astype("category")
        X_test_raw[col] = X_test_raw[col].astype("category")

dataset_samples_count = len(X_train_raw) + len(X_test_raw)
dataset_raw_features_count = X_train_raw.shape[1]

# =====================================================================
# 3. HYBRID FEATURE PIPELINE (Dynamic Mode Assembly)
# =====================================================================
if MODE == "embed-only-no-pca":
    print("Bypassing PCA compression: Loading full 512-dimensional embedding space...")
    X_train_combined = pd.DataFrame(data["X_train_embed"])
    X_test_combined = pd.DataFrame(data["X_test_embed"])

    full_column_names = [f"pca_embed_{i}" for i in range(X_train_combined.shape[1])]
    X_train_combined.columns = X_test_combined.columns = full_column_names

    cat_features = []
    pca_n_components = X_train_combined.shape[1]

else:
    pca = PCA(n_components=0.95, random_state=42)
    X_train_embed_pca = pd.DataFrame(pca.fit_transform(data["X_train_embed"]))
    X_test_embed_pca = pd.DataFrame(pca.transform(data["X_test_embed"]))
    print(f"PCA reduced text embeddings to {pca.n_components_} components.")

    pca_column_names = [f"pca_embed_{i}" for i in range(X_train_embed_pca.shape[1])]
    X_train_embed_pca.columns = X_test_embed_pca.columns = pca_column_names

    if MODE == "combined":
        X_train_combined = pd.concat([X_train_raw, X_train_embed_pca], axis=1)
        X_test_combined = pd.concat([X_test_raw, X_test_embed_pca], axis=1)
    elif MODE == "raw-only":
        X_train_combined = X_train_raw.copy()
        X_test_combined = X_test_raw.copy()
    elif MODE == "embed-only":
        X_train_combined = X_train_embed_pca.copy()
        X_test_combined = X_test_embed_pca.copy()
        cat_features = []

    pca_n_components = pca.n_components_

X_train_combined.columns = X_train_combined.columns.astype(str)
X_test_combined.columns = X_test_combined.columns.astype(str)

feat_total_count = X_train_combined.shape[1]

print(
    f"Final training matrix shape assembled for {MODE.upper()}: {X_train_combined.shape}"
)

n_rows, n_cols = X_train_combined.shape

# training data using 8-Fold Cross-Validation setup
if ALGO == "catboost":

    # --- start of fix ---
    # this fix prevents CatBoost from crashing by explicitly converting missing
    # float values (NaN) in categorical columns into string labels.
    #
    # without this block every TabArena dataset works fines except for:
    # - 20260624_164200_diabetes130us_combined_catboost
    # - 20260624_164205_diabetes130us_raw-only_catboost
    # - 20260624_183128_hr_analytics_job_change_of_data_scientists_combined_catboost
    # - 20260624_183132_hr_analytics_job_change_of_data_scientists_raw-only_catboost
    if cat_features:
        for col_indicator in cat_features:
            col_name = (
                col_indicator
                if isinstance(col_indicator, str)
                else X_train_combined.columns[col_indicator]
            )

            # --- Fix Training Data ---
            X_train_combined[col_name] = X_train_combined[col_name].fillna("Missing")
            X_train_combined[col_name] = X_train_combined[col_name].astype(str)
            X_train_combined[col_name] = X_train_combined[col_name].replace(
                ["nan", "NaN", "<NA>", "None"], "Missing"
            )

            # --- Fix Testing Data ---
            if col_name in X_test_combined.columns:
                X_test_combined[col_name] = X_test_combined[col_name].fillna("Missing")
                X_test_combined[col_name] = X_test_combined[col_name].astype(str)
                X_test_combined[col_name] = X_test_combined[col_name].replace(
                    ["nan", "NaN", "<NA>", "None"], "Missing"
                )
    # --- end of fix ---

    train_pool = cb.Pool(X_train_combined, label=y_train, cat_features=cat_features)
elif ALGO == "xgboost":
    dtrain = xgb.DMatrix(X_train_combined, label=y_train, enable_categorical=True)


# =====================================================================
# 4. OPTUNA TUNING ROUTER (TabArena Config Spaces)
# =====================================================================
def get_search_space(trial):
    if ALGO == "catboost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 1e-1, log=True),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bernoulli"]
            ),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["SymmetricTree", "Depthwise"]
            ),
            "depth": trial.suggest_int("depth", 4, 8),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.85, 1.0),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 5.0, log=True),
            "leaf_estimation_iterations": trial.suggest_int(
                "leaf_estimation_iterations", 1, 20, log=True
            ),
            "one_hot_max_size": trial.suggest_int("one_hot_max_size", 8, 100, log=True),
            "model_size_reg": trial.suggest_float("model_size_reg", 0.1, 1.5, log=True),
            "max_ctr_complexity": trial.suggest_int("max_ctr_complexity", 2, 5),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Plain"]),
            "max_bin": trial.suggest_categorical("max_bin", [254]),
        }
    elif ALGO == "lightgbm":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 1e-1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
            "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
            "num_leaves": trial.suggest_int("num_leaves", 2, 200, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 64, log=True),
            "extra_trees": trial.suggest_categorical("extra_trees", [False, True]),
            "min_data_per_group": trial.suggest_int(
                "min_data_per_group", 2, 100, log=True
            ),
            "cat_l2": trial.suggest_float("cat_l2", 5e-3, 2.0, log=True),
            "cat_smooth": trial.suggest_float("cat_smooth", 1e-3, 100.0, log=True),
            "max_cat_to_onehot": trial.suggest_int(
                "max_cat_to_onehot", 8, 100, log=True
            ),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 2.0),
        }
    elif ALGO == "xgboost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 1e-1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10, log=True),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1e-3, 5.0, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0),
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            ),
            "max_cat_to_onehot": trial.suggest_int(
                "max_cat_to_onehot", 8, 100, log=True
            ),
            "max_leaves": trial.suggest_int("max_leaves", 8, 1024, log=True),
        }


def objective(trial):
    # [FIXED]: Trial 0 evaluates pure native library defaults, bypassing the search space
    if trial.number != 0:
        dynamic_space = get_search_space(trial)
    else:
        dynamic_space = {}

    if ALGO == "catboost":
        if TASK == "binary":
            obj, eval_met = "Logloss", "AUC"
        elif TASK == "multiclass":
            obj, eval_met = "MultiClass", "MultiClass"
        else:
            obj, eval_met = "RMSE", "RMSE"

        # --- ISOLATE CATBOOST BASELINE HANG PROTECTOR ---
        if trial.number == 0:
            params = {
                "iterations": 100,  # [CRITICAL]: Constrain baseline trees so 8-fold CV finishes in seconds
                "loss_function": obj,
                "eval_metric": eval_met,
                "logging_level": "Silent",
                "allow_writing_files": False,
            }
        else:
            params = {
                "iterations": 1000,
                "loss_function": obj,
                "eval_metric": eval_met,
                "logging_level": "Silent",
                "allow_writing_files": False,
                **dynamic_space,
            }

        cv_res = cb.cv(
            train_pool,
            params,
            fold_count=8,
            early_stopping_rounds=100,
            stratified=(TASK != "regression"),
        )
        optimal_trees = len(cv_res)

        if TASK == "binary":
            score = 1.0 - cv_res[f"test-{eval_met}-mean"].iloc[-1]
        else:
            score = cv_res[f"test-{eval_met}-mean"].iloc[-1]

    elif ALGO == "lightgbm":
        if TASK == "binary":
            obj, eval_met = "binary", "auc"
        elif TASK == "multiclass":
            obj, eval_met = "multiclass", "multi_logloss"
        else:
            obj, eval_met = "regression", "rmse"

        params = {
            "objective": obj,
            "metric": eval_met,
            "verbosity": -1,
            "feature_pre_filter": False,
            **dynamic_space,
        }
        if TASK == "multiclass":
            params["num_class"] = num_classes

        # [FIXED]: Constructing a fresh clean Dataset explicitly per-trial avoids mutation limits
        dtrain_clean = lgb.Dataset(X_train_combined, label=y_train)

        cv_res = lgb.cv(
            params,
            dtrain_clean,
            nfold=8,
            stratified=(TASK != "regression"),
            num_boost_round=1000,
            seed=42,
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        optimal_trees = len(cv_res[f"valid {eval_met}-mean"])

        if TASK == "binary":
            score = 1.0 - cv_res[f"valid {eval_met}-mean"][-1]
        else:
            score = cv_res[f"valid {eval_met}-mean"][-1]

    elif ALGO == "xgboost":
        if TASK == "binary":
            obj, eval_met = "binary:logistic", "auc"
        elif TASK == "multiclass":
            obj, eval_met = "multi:softprob", "mlogloss"
        else:
            obj, eval_met = "reg:squarederror", "rmse"

        params = {
            "objective": obj,
            "eval_metric": eval_met,
            "tree_method": "hist",
            **dynamic_space,
        }
        if TASK == "multiclass":
            params["num_class"] = num_classes

        cv_res = xgb.cv(
            params,
            dtrain,
            nfold=8,
            stratified=(TASK != "regression"),
            num_boost_round=1000,
            early_stopping_rounds=100,
            seed=42,
            verbose_eval=False,
        )
        optimal_trees = cv_res.shape[0]

        if TASK == "binary":
            score = 1.0 - cv_res[f"test-{eval_met}-mean"].iloc[-1]
        else:
            score = cv_res[f"test-{eval_met}-mean"].iloc[-1]

    trial.set_user_attr("optimal_trees", optimal_trees)

    # Clean Manual W&B Step Logging (Ignoring Trial 0 to prevent timeline mismatch)
    if USE_WANDB and trial.number != 0:
        wandb.log(
            {"optimization_score": score, "trial_params": dynamic_space},
            step=trial.number,
        )

    return score


print(f"\nRunning {TRIALS}-Trial Optuna Study for {ALGO.upper()} ({MODE.upper()})...")

# Since we converted AUC to 1-AUC, Optuna direction is always 'minimize'
optuna_direction = "minimize"

# Route Optuna database entirely into the session sub-folder using new prefix
db_filename = f"{FILE_PREFIX}.db"
db_path = os.path.join(DBS_DIR, db_filename)

study = optuna.create_study(
    study_name=FILE_PREFIX,
    direction=optuna_direction,
    sampler=optuna.samplers.TPESampler(seed=42),
    storage=f"sqlite:///{db_path}",
    load_if_exists=True,
)

# [FIXED]: Enqueue an empty dictionary to execute native library defaults as Trial 0
study.enqueue_trial({}, skip_if_exists=True)

opt_start_time = time.time()
study.optimize(
    objective,
    n_trials=TRIALS,
    show_progress_bar=True,
    catch=(Exception,),
)
opt_duration = time.time() - opt_start_time

print(f"Optuna Search Completed in: {opt_duration:.2f} seconds")

# =====================================================================
# 5. UNIFIED FINAL MODEL STANDARDIZED EVALUATION
# =====================================================================
print("\nTraining final model on optimized parameter setup...")
optimal_trees = study.best_trial.user_attrs.get("optimal_trees", 500)

if ALGO == "catboost":
    obj = (
        "Logloss"
        if TASK == "binary"
        else ("MultiClass" if TASK == "multiclass" else "RMSE")
    )
    final_model = (
        cb.CatBoostClassifier(
            **study.best_params,
            iterations=optimal_trees,
            loss_function=obj,
            random_seed=42,
            silent=True,
        )
        if TASK in ["binary", "multiclass"]
        else cb.CatBoostRegressor(
            **study.best_params,
            iterations=optimal_trees,
            loss_function=obj,
            random_seed=42,
            silent=True,
        )
    )

    train_start_time = time.time()
    final_model.fit(train_pool)
    train_duration = time.time() - train_start_time

    infer_start_time = time.time()
    preds_raw = (
        final_model.predict_proba(X_test_combined)
        if TASK in ["binary", "multiclass"]
        else final_model.predict(X_test_combined)
    )
    infer_duration = time.time() - infer_start_time

    importance = final_model.get_feature_importance()
    feature_names = X_train_combined.columns

elif ALGO == "lightgbm":
    obj, eval_met = (
        ("binary", "auc")
        if TASK == "binary"
        else (
            ("multiclass", "multi_logloss")
            if TASK == "multiclass"
            else ("regression", "rmse")
        )
    )
    best_params = {
        **study.best_params,
        "objective": obj,
        "metric": eval_met,
        "verbosity": -1,
        "seed": 42,
    }
    if TASK == "multiclass":
        best_params["num_class"] = num_classes

    # [FIXED]: Reconstruct the final valid Dataset object for prediction
    dtrain_final = lgb.Dataset(X_train_combined, label=y_train)

    train_start_time = time.time()
    final_model = lgb.train(best_params, dtrain_final, num_boost_round=optimal_trees)
    train_duration = time.time() - train_start_time

    infer_start_time = time.time()
    preds_raw = final_model.predict(X_test_combined)
    infer_duration = time.time() - infer_start_time

    importance = final_model.feature_importance(importance_type="gain")
    feature_names = final_model.feature_name()

elif ALGO == "xgboost":
    obj, eval_met = (
        ("binary:logistic", "auc")
        if TASK == "binary"
        else (
            ("multi:softprob", "mlogloss")
            if TASK == "multiclass"
            else ("reg:squarederror", "rmse")
        )
    )
    best_params = {
        **study.best_params,
        "objective": obj,
        "eval_metric": eval_met,
        "tree_method": "hist",
        "seed": 42,
    }
    if TASK == "multiclass":
        best_params["num_class"] = num_classes

    dtest_final = xgb.DMatrix(X_test_combined, enable_categorical=True)

    train_start_time = time.time()
    final_model = xgb.train(best_params, dtrain, num_boost_round=optimal_trees)
    train_duration = time.time() - train_start_time

    infer_start_time = time.time()
    preds_raw = final_model.predict(dtest_final)
    infer_duration = time.time() - infer_start_time

    xgb_score = final_model.get_score(importance_type="total_gain")
    importance = [xgb_score.get(col, 0.0) for col in X_train_combined.columns]
    feature_names = X_train_combined.columns

# Calculate per-1K standardized metrics
train_duration_per_1k = train_duration / (max(1, len(X_train_combined)) / 1000.0)
infer_duration_per_1k = infer_duration / (max(1, len(X_test_combined)) / 1000.0)

# Dynamically route performance evaluation metrics
if TASK == "binary":
    preds_proba = preds_raw[:, 1] if ALGO == "catboost" else preds_raw
    primary_metric_name, primary_metric_value = "1-AUROC", 1.0 - roc_auc_score(
        y_test, preds_proba
    )
    secondary_metric_name, secondary_metric_value = "Accuracy", accuracy_score(
        y_test, (preds_proba > 0.5).astype(int)
    )
elif TASK == "multiclass":
    preds_proba = preds_raw
    primary_metric_name, primary_metric_value = "Log_Loss", log_loss(
        y_test, preds_proba
    )
    preds_class = np.argmax(preds_proba, axis=1)
    secondary_metric_name, secondary_metric_value = "Accuracy", accuracy_score(
        y_test, preds_class
    )
else:
    preds_values = preds_raw
    primary_metric_name, primary_metric_value = "RMSE", root_mean_squared_error(
        y_test, preds_values
    )
    secondary_metric_name, secondary_metric_value = "R2_Score", r2_score(
        y_test, preds_values
    )

# Calculate Deltas and Delta Percentages safely
if base_p is not None:
    primary_delta = primary_metric_value - base_p
    primary_delta_pct = (primary_delta / abs(base_p) * 100) if base_p != 0 else None
else:
    primary_delta = None
    primary_delta_pct = None

if base_s is not None:
    secondary_delta = secondary_metric_value - base_s
    secondary_delta_pct = (secondary_delta / abs(base_s) * 100) if base_s != 0 else None
else:
    secondary_delta = None
    secondary_delta_pct = None

p_delta_str = f"{primary_delta:+.4f}" if primary_delta is not None else "N/A"
s_delta_str = f"{secondary_delta:+.4f}" if secondary_delta is not None else "N/A"

print(f"\n[{ALGO.upper()} FINALIZED METRICS]")
print(f"Optimal Trees:         {optimal_trees}")
print(
    f"{primary_metric_name}:             {primary_metric_value:.4f} (Delta to TabPFN: {p_delta_str})"
)
print(
    f"{secondary_metric_name}:             {secondary_metric_value:.4f} (Delta to TabPFN: {s_delta_str})"
)
print(
    f"Final Train Time:      {train_duration:.2f} seconds ({train_duration_per_1k:.4f}s per 1K)"
)
print(
    f"Final Infer Time:      {infer_duration:.2f} seconds ({infer_duration_per_1k:.4f}s per 1K)"
)

# =====================================================================
# 6. FEATURE IMPORTANCE
# =====================================================================
print(f"\n--- {ALGO.upper()} Feature Importance (Raw vs. Scaled Share) ---")

raw_importance = np.array(importance, dtype=np.float64)
total_importance_sum = raw_importance.sum()

if total_importance_sum > 0:
    scaled_importance = (raw_importance / total_importance_sum) * 100
else:
    scaled_importance = np.zeros_like(raw_importance)

feature_matrix = list(zip(feature_names, raw_importance, scaled_importance))
sorted_features = sorted(feature_matrix, key=lambda x: x[1], reverse=True)

print(f"Feature | Raw gain | Share")
for name, raw_score, scaled_percentage in sorted_features:
    print(f"{name} | {raw_score:.2f} | {scaled_percentage:.2f}%")

total_embedded_share = sum(
    pct for name, _, pct in sorted_features if "pca_embed" in name
)
total_raw_share = sum(
    pct for name, _, pct in sorted_features if "pca_embed" not in name
)

total_embedded_pool = sum(1 for name in feature_names if "pca_embed" in name)
total_raw_pool = sum(1 for name in feature_names if "pca_embed" not in name)

active_embedded_count = sum(
    1 for name, raw, _ in sorted_features if "pca_embed" in name and raw > 0
)
active_raw_count = sum(
    1 for name, raw, _ in sorted_features if "pca_embed" not in name and raw > 0
)

active_embedded_pct = (
    (active_embedded_count / total_embedded_pool * 100)
    if total_embedded_pool > 0
    else 0
)
active_raw_pct = (active_raw_count / total_raw_pool * 100) if total_raw_pool > 0 else 0

cr5_score = sum(pct for _, _, pct in sorted_features[:5])
cr10_score = sum(pct for _, _, pct in sorted_features[:10])

print("\n" + "=" * 69)
print("METRIC TYPE                                  | VALUE / PROFILE")
print("=" * 69)
print(f"Embedding Share                              | {total_embedded_share:.2f}%")
print(f"Raw Feature Share                            | {total_raw_share:.2f}%")
print("-" * 69)
print(
    f"Active Text Embedding Density                | {active_embedded_pct:.2f}% ({active_embedded_count}/{total_embedded_pool})"
)
print(
    f"Active Raw Tabular Feature Density           | {active_raw_pct:.2f}% ({active_raw_count}/{total_raw_pool})"
)
print("-" * 69)
print(
    f"Top-5 Concentration Ratio (CR5)              | {cr5_score:.2f}% of total decision weight"
)
print(
    f"Top-10 Concentration Ratio (CR10)            | {cr10_score:.2f}% of total decision weight"
)
print("=" * 69)

# =====================================================================
# 7. PERMANENT ARTIFACT EXPORTER (JSON for Metrics)
# =====================================================================
total_duration = time.time() - global_start_time
print(f"\nTotal Pipeline Execution Time: {total_duration:.2f} seconds")

# --- DYNAMIC TRIALS TELEMETRY EXTRACTION ---
completed_trials = [
    t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
]
tuning_completed = len(completed_trials)

tuning_val_score_0 = (
    study.trials[0].value
    if len(study.trials) > 0 and study.trials[0].value is not None
    else np.nan
)
tuning_val_score_best = study.best_value
tuning_best_idx = study.best_trial.number

if pd.notna(tuning_val_score_0):
    tuning_overall_gain = tuning_val_score_0 - tuning_val_score_best
    tuning_gain_pct = (
        (tuning_overall_gain / abs(tuning_val_score_0)) * 100
        if tuning_val_score_0 != 0
        else 0.0
    )
else:
    tuning_overall_gain = np.nan
    tuning_gain_pct = np.nan

# Overfitting Ratio: Primary Test Value / Best Validation Score
eval_overfitting_ratio = (
    (primary_metric_value / tuning_val_score_best)
    if tuning_val_score_best != 0
    else np.nan
)
# -------------------------------------------

# Full JSON dict exactly as needed for your local orchestrator metrics mapping
performance_data = {
    "dataset": root_dataset,
    "algorithm": ALGO,
    "mode": MODE,
    "task_type": TASK,
    "dataset_samples_count": dataset_samples_count,
    "dataset_raw_features_count": dataset_raw_features_count,
    "feat_total_count": feat_total_count,
    "pca_n_components": int(pca_n_components),
    "optimal_trees": optimal_trees,
    "tuning_completed": tuning_completed,
    "tuning_best_idx": tuning_best_idx,
    "tuning_val_score_0": tuning_val_score_0,
    "tuning_val_score_best": tuning_val_score_best,
    "tuning_overall_gain": tuning_overall_gain,
    "tuning_gain_pct": tuning_gain_pct,
    "eval_overfitting_ratio": eval_overfitting_ratio,
    "primary_metric": primary_metric_name,
    "tabpfn_primary_score": base_p,
    "eval_primary_value": primary_metric_value,
    "eval_primary_delta": primary_delta,
    "eval_primary_delta_pct": primary_delta_pct,
    "secondary_metric": secondary_metric_name,
    "tabpfn_secondary_score": base_s,
    "eval_secondary_value": secondary_metric_value,
    "eval_secondary_delta": secondary_delta,
    "eval_secondary_delta_pct": secondary_delta_pct,
    "feat_share_embedded_pct": total_embedded_share,
    "feat_share_raw_pct": total_raw_share,
    "feat_active_embedded_pct": active_embedded_pct,
    "feat_active_raw_pct": active_raw_pct,
    "feat_cr5_pct": cr5_score,
    "feat_cr10_pct": cr10_score,
    "time_tuning_search_secs": opt_duration,
    "time_train_secs": train_duration,
    "time_infer_secs": infer_duration,
    "time_train_secs_per_1K": train_duration_per_1k,
    "time_infer_secs_per_1K": infer_duration_per_1k,
    "time_total_secs": total_duration,
}

# Write summary metrics cleanly to JSON using new prefix
metrics_filename = f"{FILE_PREFIX}.json"
metrics_path = os.path.join(METRICS_DIR, metrics_filename)

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(performance_data, f, indent=4)

# Write feature importance matrix to CSV using new prefix
importance_filename = f"{FILE_PREFIX}.csv"
importance_path = os.path.join(IMPORTANCE_DIR, importance_filename)

df_importance = pd.DataFrame(
    sorted_features, columns=["feature_name", "raw_gain", "percentage_share"]
)
df_importance.to_csv(importance_path, index=False)

# Write trial optimization history to CSV using new prefix
trials_filename = f"{FILE_PREFIX}.csv"
trials_path = os.path.join(TRIALS_DIR, trials_filename)

df_trials = study.trials_dataframe()
df_trials.to_csv(trials_path, index=False)

print(f"\n[SUCCESS] Permanent artifacts safely written to '{BASE_OUT_DIR}/'")

# --- W&B ARTIFACT LOGGING & CONFIG/SUMMARY SPLIT ---
if USE_WANDB:
    try:
        # 1. Inherent Metadata -> wandb.config
        wandb.config.update(
            {
                "dataset": root_dataset,
                "task_type": TASK,
                "mode": MODE,
                "algorithm": ALGO,
                "primary_metric": primary_metric_name,
                "secondary_metric": secondary_metric_name,
                "tabpfn_primary_score": base_p,
                "tabpfn_secondary_score": base_s,
                "dataset_samples_count": dataset_samples_count,
                "dataset_raw_features_count": dataset_raw_features_count,
                "feat_total_count": feat_total_count,
                "pca_n_components": int(pca_n_components),
            }
        )

        # Process Best Params to prevent dropping if Trial 0 won
        best_params_out = (
            study.best_params
            if study.best_trial.number != 0
            else {"status": "Native library defaults used (Trial 0)"}
        )

        # 2. Optimization Results -> wandb.summary
        wandb.summary.update(
            {
                "best_trial_params": best_params_out,
                "optimal_trees": optimal_trees,
                "tuning_completed": tuning_completed,
                "tuning_best_idx": tuning_best_idx,
                "tuning_val_score_0": tuning_val_score_0,
                "tuning_val_score_best": tuning_val_score_best,
                "tuning_overall_gain": tuning_overall_gain,
                "tuning_gain_pct": tuning_gain_pct,
                "eval_overfitting_ratio": eval_overfitting_ratio,
                "eval_primary_value": primary_metric_value,
                "eval_primary_delta": primary_delta,
                "eval_primary_delta_pct": primary_delta_pct,
                "eval_secondary_value": secondary_metric_value,
                "eval_secondary_delta": secondary_delta,
                "eval_secondary_delta_pct": secondary_delta_pct,
                "feat_share_embedded_pct": total_embedded_share,
                "feat_share_raw_pct": total_raw_share,
                "feat_active_embedded_pct": active_embedded_pct,
                "feat_active_raw_pct": active_raw_pct,
                "feat_cr5_pct": cr5_score,
                "feat_cr10_pct": cr10_score,
                "time_tuning_search_secs": opt_duration,
                "time_train_secs": train_duration,
                "time_infer_secs": infer_duration,
                "time_train_secs_per_1K": train_duration_per_1k,
                "time_infer_secs_per_1K": infer_duration_per_1k,
                "time_total_secs": total_duration,
            }
        )

        # 3. Feature Importance -> Native W&B Table
        fi_table = wandb.Table(columns=["feature_name", "raw_gain", "percentage_share"])
        for row in sorted_features:
            fi_table.add_data(row[0], row[1], row[2])
        wandb.log({"feature_importance": fi_table})

        # 4. Push local CSV artifacts for permanent backup
        artifact = wandb.Artifact(
            name=f"trial-results-{WANDB_RUN_NAME}",
            type="trial-results",
            description=f"Tuning results for {ALGO.upper()} in {MODE.upper()} mode on {root_dataset}",
        )
        artifact.add_file(metrics_path, name="metrics.json")
        artifact.add_file(importance_path, name="feature_importance.csv")
        artifact.add_file(trials_path, name="trials_history.csv")
        wandb.log_artifact(artifact)

        print(
            f"[W&B] Artifacts and localized metrics logged successfully for {WANDB_RUN_NAME}"
        )
        wandb.finish()
    except Exception as e:
        print(f"[W&B] Warning: Could not log artifacts: {e}")
