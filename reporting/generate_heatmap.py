from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =====================================================================
# PATH CONFIGURATION (Relative to this script file)
# =====================================================================
SCRIPT_DIR = Path(__file__).parent
CSV_PATH = SCRIPT_DIR / "summary_matrix_refactored.csv"
OUTPUT_FILE = SCRIPT_DIR / "heatmap.png"

# Exact mapping used in the win rate matrix
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

# =====================================================================
# 1. LOAD & PREPARE DATA
# =====================================================================
if not CSV_PATH.exists():
    print(f"Error: Target CSV file not found at: {CSV_PATH}")
    exit(1)

print(f"Loading data from: {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# Rename the configurations to match the LaTeX formatting
df["mode_algorithm"] = df["mode_algorithm"].map(
    lambda x: NAME_MAPPING.get(str(x).strip(), x)
)

# =====================================================================
# 2. CALCULATION
# =====================================================================
df_norm = df.copy()
df_norm["min_val"] = df_norm.groupby("dataset")["eval_primary_value"].transform("min")
df_norm["max_val"] = df_norm.groupby("dataset")["eval_primary_value"].transform("max")
df_norm["relative_score"] = (df_norm["max_val"] - df_norm["eval_primary_value"]) / (
    df_norm["max_val"] - df_norm["min_val"]
)

# =====================================================================
# 3. PLOT: PERFORMANCE HEATMAP
# =====================================================================
print("Generating Heatmap...")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# Pivot the data
heatmap_data = df_norm.pivot(
    index="dataset", columns="mode_algorithm", values="relative_score"
)

# Sort configurations by average performance across all datasets
heatmap_data = heatmap_data[heatmap_data.mean().sort_values(ascending=False).index]

plt.figure(figsize=(14, len(heatmap_data) * 0.4))
sns.heatmap(
    heatmap_data,
    annot=False,
    cmap="YlGnBu",
    cbar_kws={"label": "Relative Performance (1=Best)"},
)

plt.title("Performance Matrix: Dataset vs Configuration")
plt.xlabel("Configuration")
plt.ylabel("Dataset")
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=300)
plt.close()

print(f"Heatmap saved successfully to: {OUTPUT_FILE}")
