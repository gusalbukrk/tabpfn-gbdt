# =====================================================================
# 1. INITIAL SYSTEM IMPORTS & MATRIX CONFIGURATION
# =====================================================================
import argparse
import os
import sys
import glob
import json
import subprocess
import time
import re
import io
from datetime import datetime
import pandas as pd
import numpy as np

# --- GOOGLE DRIVE INTEGRATION IMPORTS ---
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

parser = argparse.ArgumentParser(
    description="TabArena Automated Matrix Runner Orchestrator"
)
parser.add_argument(
    "--session-id",
    type=str,
    required=True,
    help="Unique identifier to isolate this full experimental run results",
)
parser.add_argument(
    "--use-gdrive",
    action="store_true",
    help="Flag to enable Google Drive integration for fetching .npz archives",
)
parser.add_argument(
    "--use-wandb",
    action="store_true",
    help="Flag to enable Weights & Biases integration for experiment tracking",
)
args = parser.parse_args()

SESSION_ID = args.session_id
USE_GDRIVE = args.use_gdrive
USE_WANDB = args.use_wandb
# ALGORITHMS = ["lightgbm", "xgboost", "catboost"]
# MODES = ["combined", "raw-only", "embed-only"]
ALGORITHMS = ["catboost"]
# MODES = ["combined"]
MODES = ["raw-only", "combined"]
TRIALS_PER_JOB = 25

ARCHIVES_DIR = "archives"
GDRIVE_DIR = "gdrive"
BASE_OUT_DIR = os.path.join("outputs", SESSION_ID)

# if os.path.exists(BASE_OUT_DIR):
#     print(f"[FATAL ERROR] Session ID '{SESSION_ID}' already exists!")
#     sys.exit(1)

METRICS_DIR = os.path.join(BASE_OUT_DIR, "metrics")
TRIALS_DIR = os.path.join(BASE_OUT_DIR, "trials")
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- OAUTH CONFIGURATION ---
GDRIVE_FOLDER_ID = "1Dmopi2d5fl16fDKrBBRmmdslPU10_bGj"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

TOKEN_INFO = {
    "token": "",
    "refresh_token": "",
    "token_uri": "https://oauth2.googleapis.com/token",
    "client_id": "",
    "client_secret": "",
    "scopes": ["https://www.googleapis.com/auth/drive.file"],
    "universe_domain": "googleapis.com",
    "account": "",
}


def get_creds():
    creds = Credentials.from_authorized_user_info(TOKEN_INFO, SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds


print("=" * 70)
print(
    f"TABARENA AUTOMATED RUNNER DEPLOYED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
print(f"Session Identifier: {SESSION_ID}")
print(f"Benchmark matrix: {len(ALGORITHMS)} algorithms x {len(MODES)} modes")
print("=" * 70)
print()

# =====================================================================
# 2. FILE DISCOVERY & RESULTS SUMMARY
# =====================================================================
os.makedirs(ARCHIVES_DIR, exist_ok=True)
if USE_GDRIVE:
    os.makedirs(GDRIVE_DIR, exist_ok=True)

drive_service = None
target_files = []

if USE_GDRIVE:
    print("Connecting to Google Drive...")
    creds = get_creds()
    drive_service = build("drive", "v3", credentials=creds)

    query = (
        f"'{GDRIVE_FOLDER_ID}' in parents and name contains '.npz' and trashed=false"
    )
    results = (
        drive_service.files()
        .list(q=query, spaces="drive", fields="files(id, name)", pageSize=1000)
        .execute()
    )
    drive_npz_files = results.get("files", [])

    if not drive_npz_files:
        print(
            f"[FATAL ERROR] No .npz files found in Google Drive folder '{GDRIVE_FOLDER_ID}'."
        )
        sys.exit(1)

    target_files = sorted(drive_npz_files, key=lambda x: x["name"])
    print(
        f"Discovered {len(target_files)} datasets ready for benchmarking in Google Drive:"
    )
    for idx, f_info in enumerate(target_files, 1):
        print(f"  {idx}. {f_info['name']}")
else:
    npz_files = sorted(glob.glob(os.path.join(ARCHIVES_DIR, "*.npz")))

    if not npz_files:
        print(f"[FATAL ERROR] No .npz files found in the '{ARCHIVES_DIR}/' directory.")
        sys.exit(1)

    target_files = [{"name": os.path.basename(f), "id": None} for f in npz_files]
    print(f"Discovered {len(target_files)} datasets ready for benchmarking locally:")
    for idx, f_info in enumerate(target_files, 1):
        print(f"  {idx}. {f_info['name']}")

print("-" * 70)


def update_summary():
    """Scans session metric JSON files to generate a clean summary matrix."""
    metric_files = glob.glob(os.path.join(METRICS_DIR, "*.json"))

    if not metric_files:
        return

    all_rows = []
    for f in metric_files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                row_data = json.load(file)
            all_rows.append(row_data)
        except Exception:
            continue

    if all_rows:
        master_df = pd.DataFrame(all_rows)

        # 1. General numeric formatting
        time_cols_2_dec = [
            "time_total_secs",
            "time_train_secs",
            "time_infer_secs",
            "time_tuning_search_secs",
        ]
        for col in time_cols_2_dec:
            if col in master_df.columns:
                master_df[col] = pd.to_numeric(master_df[col], errors="coerce")

        time_cols_4_dec = ["time_train_secs_per_1K", "time_infer_secs_per_1K"]
        for col in time_cols_4_dec:
            if col in master_df.columns:
                master_df[col] = pd.to_numeric(master_df[col], errors="coerce")

        # Ensure native JSON telemetry columns are cleanly rounded in the final CSV
        optuna_cols_4_dec = [
            "tuning_val_score_0",
            "tuning_val_score_best",
            "tuning_overall_gain",
            "eval_overfitting_ratio",
            "eval_primary_delta",
            "eval_secondary_delta",
        ]
        for col in optuna_cols_4_dec:
            if col in master_df.columns:
                master_df[col] = pd.to_numeric(master_df[col], errors="coerce")

        pct_cols = [
            "eval_primary_delta_pct",
            "eval_secondary_delta_pct",
            "tuning_gain_pct",
            "feat_share_embedded_pct",
            "feat_share_raw_pct",
            "feat_active_embedded_pct",
            "feat_active_raw_pct",
            "feat_cr5_pct",
            "feat_cr10_pct",
        ]
        for col in pct_cols:
            if col in master_df.columns:
                master_df[col] = pd.to_numeric(master_df[col], errors="coerce")

        # 2. Apply exact target ordering using the new aligned naming scheme
        target_columns = [
            "dataset",
            "task_type",
            "algorithm",
            "mode",
            "dataset_samples_count",
            "dataset_raw_features_count",
            "pca_n_components",
            "feat_total_count",
            "primary_metric",
            "tabpfn_primary_score",
            "eval_primary_value",
            "eval_primary_delta",
            "eval_primary_delta_pct",
            "secondary_metric",
            "tabpfn_secondary_score",
            "eval_secondary_value",
            "eval_secondary_delta",
            "eval_secondary_delta_pct",
            "optimal_trees",
            "tuning_completed",
            "tuning_best_idx",
            "tuning_val_score_0",
            "tuning_val_score_best",
            "tuning_overall_gain",
            "tuning_gain_pct",
            "eval_overfitting_ratio",
            "feat_share_embedded_pct",
            "feat_share_raw_pct",
            "feat_active_embedded_pct",
            "feat_active_raw_pct",
            "feat_cr5_pct",
            "feat_cr10_pct",
            "time_train_secs",
            "time_infer_secs",
            "time_train_secs_per_1K",
            "time_infer_secs_per_1K",
            "time_tuning_search_secs",
            "time_total_secs",
        ]

        existing_columns = [col for col in target_columns if col in master_df.columns]
        summary_matrix = master_df[existing_columns].copy()

        summary_matrix = summary_matrix.sort_values(by=["dataset", "algorithm", "mode"])

        output_raw_path = os.path.join(BASE_OUT_DIR, "summary_matrix.csv")
        summary_matrix.to_csv(output_raw_path, index=False)


os.makedirs(METRICS_DIR, exist_ok=True)
update_summary()

orchestrator_start_time = time.time()
completed_jobs = 0
skipped_jobs = 0
failed_jobs = 0
total_jobs_to_run = len(target_files) * len(ALGORITHMS) * len(MODES)

# =====================================================================
# 3. MATRIX EXECUTION ENGINE WITH CHECKPOINTING
# =====================================================================
for dataset_idx, file_info in enumerate(target_files, 1):
    filename = file_info["name"]
    file_id = file_info.get("id")
    dataset_name = os.path.splitext(filename)[0]
    root_dataset = dataset_name

    current_archive_dir = GDRIVE_DIR if USE_GDRIVE else ARCHIVES_DIR
    archive_path = os.path.join(current_archive_dir, filename)

    print(
        f"\n>>> PROCESSING DATASET [{dataset_idx}/{len(target_files)}]: {dataset_name.upper()}"
    )
    print("=" * 70)

    # --- JIT GOOGLE DRIVE DOWNLOAD ---
    if USE_GDRIVE and file_id:
        if not os.path.exists(archive_path):
            print(f"[DOWNLOAD] Fetching {filename} from Google Drive...")
            request = drive_service.files().get_media(fileId=file_id)
            fh = io.FileIO(archive_path, "wb")
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            print(
                f"[DOWNLOAD] {filename} successfully saved to local {current_archive_dir}/"
            )

    for algo in ALGORITHMS:
        for mode in MODES:
            completed_jobs += 1

            # Checks for the exact new filename format output by tuning.py
            checkpoint_pattern = os.path.join(
                METRICS_DIR, f"????????_??????_{root_dataset}_{mode}_{algo}.json"
            )
            # -----------------------------------------------------------------------------
            existing_runs = glob.glob(checkpoint_pattern)

            if existing_runs and os.path.getsize(existing_runs[0]) > 50:
                print(
                    f"[JOB {completed_jobs}/{total_jobs_to_run}] [RESUMED/SKIPPED] {algo.upper()} | {mode.upper()} already exists in session."
                )
                skipped_jobs += 1
                continue

            print(
                f"\n[JOB {completed_jobs}/{total_jobs_to_run}] Launching {algo.upper()} in {mode.upper()} mode..."
            )

            command = [
                sys.executable,
                "tuning.py",
                "--archive-path",
                archive_path,
                "--algo",
                algo,
                "--mode",
                mode,
                "--trials",
                str(TRIALS_PER_JOB),
                "--session-id",
                SESSION_ID,
            ]

            # Pass W&B flag downstream if enabled
            if USE_WANDB:
                command.append("--use-wandb")

            job_start_time = time.time()

            try:
                result = subprocess.run(
                    command, stdout=None, stderr=subprocess.PIPE, text=True, check=True
                )
                print(f"[SUCCESS] Job {completed_jobs} completed cleanly.")

                update_summary()

            except subprocess.CalledProcessError as e:
                failed_jobs += 1
                print(f"\n[CRITICAL ERROR] Job {completed_jobs} FAILED.")
                print(f"Command executed: {' '.join(command)}")
                print(f"Error Traceback details:\n{e.stderr}")
                print(
                    "Skipping this specific matrix node and routing to next scheduled job...\n"
                )
                continue

# =====================================================================
# 4. COMPREHENSIVE PIPELINE SUMMARY
# =====================================================================
orchestrator_total_duration = time.time() - orchestrator_start_time
newly_successful_jobs = completed_jobs - skipped_jobs - failed_jobs

print("\n" + "=" * 70)
print("TABARENA CORE BENCHMARK MATRIX RUN COMPLETE")
print("=" * 70)
print(
    f"Total Wall-Clock Processing Time : {orchestrator_total_duration:.2f} seconds ({orchestrator_total_duration/60:.2f} minutes)"
)
print(f"Total Grid Matrix Positions      : {completed_jobs}/{total_jobs_to_run}")
print(f"  └─ Active Session Skip (Saved) : {skipped_jobs}")
print(f"  └─ Newly Optimized (Success)   : {newly_successful_jobs}")
print(f"  └─ Caught Pipeline Anomalies   : {failed_jobs}")
print(f"Files updated live at '{BASE_OUT_DIR}/summary_matrix.csv'.")
print("=" * 70)
