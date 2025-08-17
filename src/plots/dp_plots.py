# dp_plots.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")
os.makedirs("dp_results", exist_ok=True)

# Load logs
df = pd.read_csv("../results/epsilon_metrics_log.csv")

# Average metrics per epsilon across folds and epochs
grouped = df.groupby(["noise_multiplier", "epoch"]).agg({
    "epsilon": "mean",
    "accuracy": "mean",
    "f1_score": "mean",
    "recall": "mean"
}).reset_index()

# === Plot Accuracy vs Epsilon ===
plt.figure(figsize=(8, 6))
sns.lineplot(data=grouped, x="epsilon", y="accuracy", hue="noise_multiplier", marker="o")
plt.title("Accuracy vs Privacy Budget (Epsilon)", fontsize=14)
plt.xlabel("Epsilon (ε)")
plt.ylabel("Accuracy")
plt.legend(title="Noise Multiplier")
plt.tight_layout()
plt.savefig("dp_results/plot_accuracy_vs_epsilon.png")
plt.close()

# === Plot F1-score vs Epsilon ===
plt.figure(figsize=(8, 6))
sns.lineplot(data=grouped, x="epsilon", y="f1_score", hue="noise_multiplier", marker="s")
plt.title("F1-score vs Privacy Budget (Epsilon)", fontsize=14)
plt.xlabel("Epsilon (ε)")
plt.ylabel("F1-score")
plt.legend(title="Noise Multiplier")
plt.tight_layout()
plt.savefig("dp_results/plot_f1_vs_epsilon.png")
plt.close()

# === Plot Recall vs Epsilon ===
plt.figure(figsize=(8, 6))
sns.lineplot(data=grouped, x="epsilon", y="recall", hue="noise_multiplier", marker="^")
plt.title("Recall vs Privacy Budget (Epsilon)", fontsize=14)
plt.xlabel("Epsilon (ε)")
plt.ylabel("Recall")
plt.legend(title="Noise Multiplier")
plt.tight_layout()
plt.savefig("dp_results/plot_recall_vs_epsilon.png")
plt.close()
