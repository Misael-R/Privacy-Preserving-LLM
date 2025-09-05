import os
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def find_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the candidate paths exist: {candidates}")

def nice_ylim(values, pad=0.03):
    lo = max(0.0, float(np.nanmin(values)) - pad)
    hi = min(1.0, float(np.nanmax(values)) + pad)
    if math.isfinite(lo) and math.isfinite(hi) and lo < hi:
        return (lo, hi)
    return (0.0, 1.0)

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = find_path(["src/results", os.path.join(ROOT, "../results")])
FIG_DIR = os.path.join(ROOT, "figures_original")
ensure_dir(FIG_DIR)
BASELINE_CSV = find_path([
    os.path.join(RESULTS_DIR, "baseline_metrics_log.csv")
])
DP_CSV = find_path([
    os.path.join(RESULTS_DIR, "epsilon_metrics_log.csv")
])
baseline_df = pd.read_csv(BASELINE_CSV)
dp_df = pd.read_csv(DP_CSV)
required_baseline_cols = {"fold", "accuracy", "f1_score", "recall"}
missing = required_baseline_cols - set(baseline_df.columns)
if missing:
    raise ValueError(f"baseline_metrics_log.csv missing columns: {missing}")
required_dp_cols = {"noise_multiplier", "epoch", "epsilon", "accuracy", "f1_score", "recall"}
missing = required_dp_cols - set(dp_df.columns)
if missing:
    raise ValueError(f"epsilon_metrics_log.csv missing columns: {missing}")
baseline_df = baseline_df.copy()
dp_df = dp_df.copy()
baseline_df["fold"] = baseline_df["fold"].astype(int)
dp_df["epoch"] = dp_df["epoch"].astype(int)
dp_df["noise_multiplier"] = dp_df["noise_multiplier"].astype(float)

# Baseline metrics per fold
def plot_baseline_metrics(df: pd.DataFrame, out_path: str):
    folds = df["fold"].values
    metrics = ["accuracy", "f1_score", "recall"]
    colors = {"accuracy": "#1f77b4", "f1_score": "#7D3C98", "recall": "#7B7D7D"}
    plt.figure(figsize=(8, 5.5))
    for m in metrics:
        plt.plot(folds, df[m].values, marker="o", linestyle="-", label=m.replace("_", " ").title(), color=colors[m])
        mean_val = df[m].mean()
        std_val = df[m].std(ddof=1)
        plt.hlines(mean_val, xmin=min(folds), xmax=max(folds), colors=colors[m], linestyles="--", alpha=0.7)
        if not np.isnan(std_val) and std_val > 0:
            plt.fill_between(
                [min(folds), max(folds)],
                [mean_val - std_val, mean_val - std_val],
                [mean_val + std_val, mean_val + std_val],
                color=colors[m], alpha=0.08
            )
    all_vals = df[metrics].values.flatten()
    plt.ylim(*nice_ylim(all_vals))
    plt.xlim(min(folds) - 0.2, max(folds) + 0.2)
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.title("Baseline (Logistic Regression): Metrics per Fold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    savefig(out_path)
plot_baseline_metrics(baseline_df, os.path.join(FIG_DIR, "baseline_metrics.png"))

dp_grouped = dp_df.groupby(["noise_multiplier", "epoch"]).agg(
    epsilon_mean=("epsilon", "mean"),
    accuracy_mean=("accuracy", "mean"),
    accuracy_std=("accuracy", "std"),
    f1_mean=("f1_score", "mean"),
    f1_std=("f1_score", "std"),
    recall_mean=("recall", "mean"),
    recall_std=("recall", "std")
).reset_index()
dp_by_noise = dp_df.groupby("noise_multiplier").agg(
    epsilon_mean=("epsilon", "mean"),
    accuracy_mean=("accuracy", "mean"),
    accuracy_std=("accuracy", "std"),
    f1_mean=("f1_score", "mean"),
    f1_std=("f1_score", "std"),
    recall_mean=("recall", "mean"),
    recall_std=("recall", "std")
).reset_index().sort_values("noise_multiplier")

# DP metrics vs noise
def plot_dp_metrics_noise(dpn: pd.DataFrame, out_path: str):
    idx = np.arange(len(dpn))
    width = 0.25
    plt.figure(figsize=(9, 5.5))
    plt.bar(idx - width, dpn["accuracy_mean"], width, yerr=dpn["accuracy_std"], capsize=4, label="Accuracy", color="#27AE60", alpha=0.9)
    plt.bar(idx,         dpn["f1_mean"],       width, yerr=dpn["f1_std"],       capsize=4, label="F1-Score", color="#1ABC9C", alpha=0.9)
    plt.bar(idx + width, dpn["recall_mean"],   width, yerr=dpn["recall_std"],   capsize=4, label="Recall",   color="#117864", alpha=0.9)
    labels = [f"Noise={nm:g}\n(ε≈{e:.2f})" for nm, e in zip(dpn["noise_multiplier"], dpn["epsilon_mean"])]
    plt.xticks(idx, labels)
    all_vals = np.hstack([
        dpn["accuracy_mean"].values, dpn["f1_mean"].values, dpn["recall_mean"].values
    ])
    plt.ylim(*nice_ylim(all_vals))
    plt.ylabel("Score")
    plt.title("DP Model: Metrics by Noise Multiplier (mean ± std across folds/epochs)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    savefig(out_path)
plot_dp_metrics_noise(dp_by_noise, os.path.join(FIG_DIR, "dp_metrics_noise.png"))

# Epsilon trade-off curves
def plot_epsilon_tradeoff(dpg: pd.DataFrame, out_path: str):
    plt.figure(figsize=(8.5, 5.5))
    noises = sorted(dpg["noise_multiplier"].unique())
    colors = {"accuracy_mean": "#27AE60", "f1_mean": "#1ABC9C", "recall_mean": "#117864"}
    for nm in noises:
        sub = dpg[dpg["noise_multiplier"] == nm].sort_values("epsilon_mean")
        for m_key, label in [("accuracy_mean", "Accuracy"), ("f1_mean", "F1-Score"), ("recall_mean", "Recall")]:
            plt.plot(sub["epsilon_mean"], sub[m_key], marker="o", linestyle="-", alpha=0.9,
                     label=f"{label} (noise={nm:g})", color=colors[m_key])
    yvals = np.hstack([dpg["accuracy_mean"].values, dpg["f1_mean"].values, dpg["recall_mean"].values])
    plt.ylim(*nice_ylim(yvals))
    plt.xlabel("Privacy Budget (ε)")
    plt.ylabel("Score")
    plt.title("Privacy–Utility Trade-off: Metrics vs Epsilon")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(ncol=2)
    savefig(out_path)
plot_epsilon_tradeoff(dp_grouped, os.path.join(FIG_DIR, "epsilon_tradeoff.png"))

# Baseline vs DP
def plot_baseline_vs_dp(baseline: pd.DataFrame, dpn: pd.DataFrame, out_path: str):
    b_acc = baseline["accuracy"].mean()
    b_f1  = baseline["f1_score"].mean()
    b_rec = baseline["recall"].mean()
    labels = []
    acc_vals = []
    f1_vals = []
    rec_vals = []
    labels.append("Baseline")
    acc_vals.append(b_acc); f1_vals.append(b_f1); rec_vals.append(b_rec)
    for _, row in dpn.iterrows():
        labels.append(f"DP (noise={row['noise_multiplier']:g})")
        acc_vals.append(row["accuracy_mean"])
        f1_vals.append(row["f1_mean"])
        rec_vals.append(row["recall_mean"])
    idx = np.arange(len(labels))
    width = 0.25
    baseline_colors = ["#1f77b4", "#7D3C98", "#7B7D7D"]
    dp_colors = ["#27AE60", "#1ABC9C", "#117864"]
    acc_bar_colors = [baseline_colors[0]] + [dp_colors[0]] * (len(acc_vals) - 1)
    f1_bar_colors  = [baseline_colors[1]] + [dp_colors[1]] * (len(f1_vals) - 1)
    rec_bar_colors = [baseline_colors[2]] + [dp_colors[2]] * (len(rec_vals) - 1)
    plt.bar(idx - width, acc_vals, width, label="Accuracy", color=acc_bar_colors)
    plt.bar(idx,         f1_vals,  width, label="F1-Score", color=f1_bar_colors)
    plt.bar(idx + width, rec_vals, width, label="Recall",   color=rec_bar_colors)
    plt.xticks(idx, labels, rotation=15, ha="right")
    all_vals = np.hstack([acc_vals, f1_vals, rec_vals])
    plt.ylim(*nice_ylim(all_vals))
    plt.ylabel("Score")
    plt.title("Baseline vs DP Models (mean across folds/epochs)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    savefig(out_path)
plot_baseline_vs_dp(baseline_df, dp_by_noise, os.path.join(FIG_DIR, "baseline_vs_dp.png"))
print(f"Figures written to: {FIG_DIR}")
