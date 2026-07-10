# ==============================================================================
# Script: inspect_npz_structure.py
# Description: Inspects the contents, keys, shapes, and data types of a specific
#              .npz file. It analyzes columns to help identify how categorical
#              features are stored and outputs to the terminal.
# ==============================================================================

import os
import numpy as np

# --- Configuration ---
# Change this path to point to any of your .npz files inside your archives folder
NPZ_FILE_PATH = "archives/mic.npz"


def inspect_npz(file_path):
    if not os.path.exists(file_path):
        error_msg = f"Error: The target .npz file was not found at: {file_path}\nPlease adjust the NPZ_FILE_PATH variable at the top of the script."
        print(error_msg)
        return

    print("=" * 80)
    print("NPZ FILE INSPECTION REPORT")
    print(f"Target File: {os.path.abspath(file_path)}")
    print("=" * 80 + "\n")

    # Load the npz file safely allowing pickle structures if present
    data = np.load(file_path, allow_pickle=True)
    keys = data.files

    print(f"Available keys/arrays inside this file: {keys}\n")

    for key in keys:
        arr = data[key]
        print("-" * 50)
        print(f"Key Name: '{key}'")
        print(f"  -> Shape: {arr.shape}")
        print(f"  -> Data Type (dtype): {arr.dtype}")

        # Check if it's a feature matrix (typically 2D)
        if arr.ndim == 2:
            print("  -> Structure Analysis (First 5 columns sample):")
            # Look at unique counts for the first few columns to detect categorical integers
            num_cols_to_check = min(5, arr.shape[1])
            for col_idx in range(num_cols_to_check):
                col_data = arr[:, col_idx]
                unique_vals = np.unique(col_data)
                num_unique = len(unique_vals)

                # Show preview of unique values if small, otherwise show min/max
                if num_unique <= 10:
                    val_preview = (
                        f"{list(unique_vals[:5])}..."
                        if num_unique > 5
                        else f"{list(unique_vals)}"
                    )
                else:
                    val_preview = f"Range: [{col_data.min()}, {col_data.max()}]"

                print(
                    f"      Column {col_idx}: {num_unique} unique values | {val_preview}"
                )
        elif arr.ndim == 1:
            print(f"  -> Sample values (First 5 elements): {arr[:5]}")
        else:
            print("  -> Higher-dimensional structure detected.")
        print("-" * 50 + "\n")

    print("Inspection complete.")


if __name__ == "__main__":
    inspect_npz(NPZ_FILE_PATH)
