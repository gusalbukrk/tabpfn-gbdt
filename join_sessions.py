import os
import sys
import shutil
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
# List the session folder names you want to join
SOURCE_SESSIONS = [
    "training-1",
    "training-2",
    "training-3",
    "training-4",
    "training-5",
    "training-6",
    "training-7",
]

# The name of the new combined folder
TARGET_SESSION = "training-consolidated"

# Expected total number of runs/files per folder
EXPECTED_RUNS = 51 * 3 * 3  # datasets * modes * algorithms

# Subfolders to merge
SUBFOLDERS = ["databases", "importance", "logs", "metrics", "trials"]

# W&B Configuration for verification
WANDB_PROJECT = "tabarena-benchmark"
WANDB_ENTITY = "gusalbukrk-team"  # Usually your W&B username
# ==========================================


BASE_DIR = "outputs"


def join_sessions():
    target_dir = os.path.join(BASE_DIR, TARGET_SESSION)
    os.makedirs(target_dir, exist_ok=True)

    for sub in SUBFOLDERS:
        os.makedirs(os.path.join(target_dir, sub), exist_ok=True)

    all_csv_data = []

    print(f"Merging files into: {target_dir}...")

    for session in SOURCE_SESSIONS:
        session_path = os.path.join(BASE_DIR, session)
        if not os.path.exists(session_path):
            print(f"[SKIPPED] Source session not found: {session_path}")
            continue

        # 1. Copy files in subfolders
        for sub in SUBFOLDERS:
            src_sub = os.path.join(session_path, sub)
            dst_sub = os.path.join(target_dir, sub)

            if os.path.exists(src_sub):
                for file_name in os.listdir(src_sub):
                    src_file = os.path.join(src_sub, file_name)
                    if os.path.isfile(src_file):
                        dst_file = os.path.join(dst_sub, file_name)

                        # CRITICAL: Exit immediately if a collision is detected
                        if os.path.exists(dst_file):
                            sys.exit(
                                f"\n[CRITICAL ERROR] Data overlap detected!\n"
                                f"The file '{file_name}' from '{session}' already exists in the target folder '{sub}'.\n"
                                f"This indicates multiple instances processed the same run. Aborting merge."
                            )

                        shutil.copy2(src_file, dst_file)

        # 2. Collect summary_matrix.csv
        csv_path = os.path.join(session_path, "summary_matrix.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                all_csv_data.append(df)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")

    # 3. Merge and save the combined summary_matrix.csv
    if all_csv_data:
        merged_df = pd.concat(all_csv_data, ignore_index=True)
        merged_df = merged_df.sort_values(by="dataset", ignore_index=True)
        merged_csv_path = os.path.join(BASE_DIR, TARGET_SESSION, "summary_matrix.csv")
        merged_df.to_csv(merged_csv_path, index=False)
        print(f"Successfully created combined CSV at: {merged_csv_path}")
    else:
        print("No summary_matrix.csv files found to merge.")


def verify_runs():
    target_dir = os.path.join(BASE_DIR, TARGET_SESSION)
    print("\n==========================================")
    print("VERIFICATION")
    print("==========================================")

    # 1. Verify Subfolders
    for sub in SUBFOLDERS:
        sub_path = os.path.join(target_dir, sub)
        if os.path.exists(sub_path):
            file_count = len(
                [
                    f
                    for f in os.listdir(sub_path)
                    if os.path.isfile(os.path.join(sub_path, f))
                ]
            )
            if file_count == EXPECTED_RUNS:
                print(f"[OK] {sub}/ -> {file_count} files.")
            else:
                print(
                    f"[WARNING] {sub}/ -> Expected {EXPECTED_RUNS}, found {file_count}."
                )
        else:
            print(f"[WARNING] Directory missing: {sub_path}")

    # 2. Verify CSV Rows
    csv_path = os.path.join(target_dir, "summary_matrix.csv")
    if os.path.exists(csv_path):
        row_count = len(pd.read_csv(csv_path))
        if row_count == EXPECTED_RUNS:
            print(f"[OK] summary_matrix.csv -> {row_count} rows.")
        else:
            print(
                f"[WARNING] summary_matrix.csv -> Expected {EXPECTED_RUNS}, found {row_count} rows."
            )
    else:
        print(f"[WARNING] Missing file: {csv_path}")

    # 3. Verify Weights & Biases (Read-only)
    print("\nChecking Weights & Biases...")
    try:
        import wandb

        api = wandb.Api()
        total_wandb_runs = 0

        for session in SOURCE_SESSIONS:
            # Assumes the session name matches the `group` parameter used during wandb.init()
            runs = api.runs(
                f"{WANDB_ENTITY}/{WANDB_PROJECT}",
                filters={"group": f"session-{session}"},
            )
            total_wandb_runs += len(runs)

        if total_wandb_runs == EXPECTED_RUNS:
            print(
                f"[OK] W&B run count matches expected: {total_wandb_runs}/{EXPECTED_RUNS}"
            )
        else:
            print(
                f"[WARNING] W&B run count mismatch! Expected {EXPECTED_RUNS}, found {total_wandb_runs}."
            )

    except ImportError:
        print("[WARNING] wandb library is not installed. Skipping W&B verification.")
    except Exception as e:
        print(
            f"[WARNING] Could not verify with W&B API. Ensure you are logged in (`wandb login`). Error: {e}"
        )


if __name__ == "__main__":
    join_sessions()
    verify_runs()
