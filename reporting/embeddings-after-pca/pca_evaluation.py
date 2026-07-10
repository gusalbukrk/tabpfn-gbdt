"""
==============================================================================
Script: batch_pca_metric_evaluation.py

Description:
Executes a Bi-Metric Geometric evaluation (Silhouette Score and k-NN Purity)
on TabPFN embeddings strictly AFTER applying a 95% variance PCA compression.
Includes UMAP mapping of the compressed topological space.

Features Live Checkpointing to prevent data loss on crash, and auto-generates
a macro-aggregation summary matrix upon completion.
==============================================================================
"""

import os
import time
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import warnings

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Suppress expected environmental UMAP warnings
warnings.filterwarnings(
    "ignore", message="n_jobs value 1 overridden to 1 by setting random_state"
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ARCHIVE_DIR = Path("../../archives")
# Using a distinct output directory to prevent overwriting the raw 512D baseline plots
OUTPUT_DIR = Path("outputs")
SUMMARY_CSV = OUTPUT_DIR / "pca_intrinsic_evaluation.csv"
AGGREGATED_CSV = OUTPUT_DIR / "pca_summary.csv"
RANDOM_STATE = 42
# ==============================================================================


def safe_extract_scalar(archive, key, default_value):
    """Safely extracts metadata attributes from numpy archive elements."""
    if key not in archive:
        return default_value
    element = archive[key]
    if isinstance(element, np.ndarray):
        if element.ndim == 0:
            return element.item()
        elif element.size == 1:
            return element.flat[0]
    return element


def main():
    sns.set_theme(style="white", palette="muted")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    npz_files = list(ARCHIVE_DIR.glob("*.npz"))
    if not npz_files:
        print(f"Error: No .npz files found at: {ARCHIVE_DIR.resolve()}")
        return

    print(f"Found {len(npz_files)} archives. Commencing PCA Bi-Metric profiling...")

    for idx, filepath in enumerate(npz_files, 1):
        start_time = time.time()
        print(f"\n[{idx}/{len(npz_files)}] Evaluating (PCA Space): {filepath.name}")

        try:
            archive = np.load(filepath, allow_pickle=True)
            if "X_train_embed" not in archive or "y_train" not in archive:
                print("-> Skipped: Missing critical coordinate or target matrices.")
                continue
            X_embed_raw = archive["X_train_embed"].astype(np.float32)
            y = archive["y_train"].ravel()
        except Exception as e:
            print(f"-> Skipped: Read error ({e})")
            continue

        # Extract structural metadata profile
        dataset_name = str(safe_extract_scalar(archive, "dataset_name", filepath.stem))
        task_type = safe_extract_scalar(archive, "task_type", "classification")
        is_regression = "regress" in str(task_type).lower()

        # Coerce continuous scales to discrete structural zones if executing regression tasks
        if is_regression:
            try:
                y_numeric = y.astype(float)
                y_discrete = pd.qcut(y_numeric, q=4, labels=False, duplicates="drop")
                label_suffix = " (Quantized)"
            except Exception:
                print("-> Error mapping continuous target scales. Skipping dataset.")
                continue
        else:
            y_discrete = y
            label_suffix = ""

        _, y_discrete_encoded = np.unique(y_discrete, return_inverse=True)

        # =====================================================================
        # CORE COMPRESSION: APPLY PCA (95% Variance Threshold)
        # =====================================================================
        try:
            pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
            X_embed_pca = pca.fit_transform(X_embed_raw)
            n_retained = pca.n_components_
            print(
                f"   | PCA Compression: Reduced {X_embed_raw.shape[1]}D -> {n_retained}D"
            )
        except Exception as e:
            print(f"   | PCA Transformation Failed: {e}")
            continue

        # --- METRIC 1: PCA-SPACE SILHOUETTE SCORE ---
        try:
            sil_score = float(silhouette_score(X_embed_pca, y_discrete_encoded))
            print(f"   | PCA Silhouette{label_suffix}: {sil_score:.4f}")
        except Exception:
            sil_score = np.nan

        # --- METRIC 2: PCA-SPACE k-NN TOPOLOGICAL PROBE PURITY ---
        try:
            knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean", n_jobs=-1)
            cv_strategy = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=RANDOM_STATE
            )
            knn_scores = cross_val_score(
                knn, X_embed_pca, y_discrete_encoded, cv=cv_strategy, scoring="accuracy"
            )
            knn_purity = float(knn_scores.mean())
            print(f"   | PCA k-NN Purity{label_suffix}: {knn_purity:.4f}")
        except Exception as e:
            print(f"   | k-NN Probe Failed: {e}")
            knn_purity = np.nan

        # --- VISUALIZATION LAYER: UMAP ON PCA SPACE ---
        print("   | Constructing PCA topological visualization map...")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=RANDOM_STATE,
        )
        X_umap = reducer.fit_transform(X_embed_pca)

        fig, ax = plt.subplots(figsize=(8, 6))
        if is_regression:
            sc = ax.scatter(
                X_umap[:, 0],
                X_umap[:, 1],
                c=y.astype(float),
                cmap="viridis",
                alpha=0.7,
                edgecolors="none",
                s=25,
            )
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Continuous Target Spectrum", rotation=270, labelpad=15)
        else:
            sns.scatterplot(
                x=X_umap[:, 0],
                y=X_umap[:, 1],
                hue=y,
                palette="tab10",
                alpha=0.8,
                linewidth=0,
                s=30,
                ax=ax,
            )

            # The legend is anchored outside the top-right
            ax.legend(title="Class Labels", bbox_to_anchor=(1.02, 1), loc="upper left")

        ax.set_title("")
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")

        # Compile comprehensive metrics presentation text block
        sil_str = f"{sil_score:.4f}" if not np.isnan(sil_score) else "N/A"
        knn_str = f"{knn_purity:.4f}" if not np.isnan(knn_purity) else "N/A"

        stat_box_text = (
            f"PCA Geometry Profile ({n_retained}D):\n"
            f"• PCA Silhouette: {sil_str}\n"
            f"• PCA k-NN Purity: {knn_str}"
        )

        # MOVED OUTSIDE: Anchored outside the bottom-right of the plot
        ax.text(
            1.02,
            0.0,
            stat_box_text,
            transform=ax.transAxes,
            fontsize=9,
            family="monospace",
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                alpha=0.85,
                edgecolor="gainsboro",
            ),
        )

        sns.despine()

        png_path = OUTPUT_DIR / f"{dataset_name}_pca_geometry_profile.png"

        # bbox_inches="tight" ensures the new external text boxes are not cropped out
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"   | Topology plot saved to: {png_path.name}")

        # =====================================================================
        # LIVE CHECKPOINT: OVERWRITE RECEIPT AFTER EVERY DATASET
        # =====================================================================
        current_record = pd.DataFrame(
            [
                {
                    "dataset": dataset_name,
                    "task_type": task_type,
                    "pca_dimensions": n_retained,
                    "pca_silhouette_score": sil_score,
                    "pca_knn_purity": knn_purity,
                }
            ]
        )

        if os.path.exists(SUMMARY_CSV):
            global_df = pd.read_csv(SUMMARY_CSV)
            if "dataset" in global_df.columns:
                mask = global_df["dataset"].astype(str) == str(dataset_name)
                global_df = global_df[~mask]
            global_df = pd.concat([global_df, current_record], ignore_index=True)
        else:
            global_df = current_record

        global_df.to_csv(SUMMARY_CSV, index=False)

        duration = time.time() - start_time
        print(f"   | Live CSV Receipt updated. Fold duration: {duration:.2f}s")

        # Isolate memory flush to keep RAM/VRAM pristine
        del (
            X_embed_raw,
            X_embed_pca,
            y,
            y_discrete,
            y_discrete_encoded,
            X_umap,
            current_record,
            archive,
        )
        gc.collect()

    print("\n" + "=" * 70)
    print("Execution Complete! Generating macro-aggregation summary...")

    # =====================================================================
    # GENERATE MACRO-AGGREGATION SUMMARY
    # =====================================================================
    if os.path.exists(SUMMARY_CSV):
        df_results = pd.read_csv(SUMMARY_CSV)
        df_results["task_type"] = df_results["task_type"].str.lower().str.strip()

        metrics = ["pca_silhouette_score", "pca_knn_purity"]
        stats = ["mean", "median", "std", "min", "max"]

        # Generate aggregations grouped by architectural task
        grouped_summary = df_results.groupby("task_type")[metrics].agg(stats)
        grouped_summary.columns = [
            f"{metric}_{stat}" for metric, stat in grouped_summary.columns
        ]
        grouped_summary = grouped_summary.reset_index()

        # Generate macro-aggregations across the entire dataset space
        global_dict = {"task_type": "all_combined"}
        for metric in metrics:
            global_dict[f"{metric}_mean"] = df_results[metric].mean()
            global_dict[f"{metric}_median"] = df_results[metric].median()
            global_dict[f"{metric}_std"] = df_results[metric].std()
            global_dict[f"{metric}_min"] = df_results[metric].min()
            global_dict[f"{metric}_max"] = df_results[metric].max()

        global_summary = pd.DataFrame([global_dict])

        # Interlock tables cleanly into a singular presentation matrix
        final_summary = pd.concat([grouped_summary, global_summary], ignore_index=True)
        final_summary = final_summary.round(4)

        # Export structured output matrix
        final_summary.to_csv(AGGREGATED_CSV, index=False)
        print(f"Summary aggregation successfully written to: {AGGREGATED_CSV}")

    print(f"All PCA bounds mapped and logged to: {SUMMARY_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
