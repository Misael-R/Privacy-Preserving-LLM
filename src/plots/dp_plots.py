import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
palette = sns.color_palette("mako_r", 6)
sns.set(style="whitegrid")
os.makedirs("dp_results", exist_ok=True)
df = pd.read_csv("../results/epsilon_metrics_log.csv")
grouped = df.groupby(["noise_multiplier", "epoch"]).agg({
    "epsilon": "mean",
    "accuracy": "mean",
    "f1_score": "mean",
    "recall": "mean"
}).reset_index()

# Accuracy vs epsilon
plt.figure(figsize=(8, 6))
sns.lineplot(data=grouped, x="epsilon", y="accuracy", hue="noise_multiplier", marker="o", palette=palette)
plt.title("Accuracy vs Privacy Budget (Epsilon)", fontsize=14)
plt.xlabel("Epsilon (ε)")
plt.ylabel("Accuracy")
plt.legend(title="Noise Multiplier")
plt.tight_layout()
plt.savefig("dp_results/plot_accuracy_vs_epsilon.png")
plt.close()

# F1-score vs epsilon
plt.figure(figsize=(8, 6))
sns.lineplot(data=grouped, x="epsilon", y="f1_score", hue="noise_multiplier", marker="s", palette=palette)
plt.title("F1-score vs Privacy Budget (Epsilon)", fontsize=14)
plt.xlabel("Epsilon (ε)")
plt.ylabel("F1-score")
plt.legend(title="Noise Multiplier")
plt.tight_layout()
plt.savefig("dp_results/plot_f1_vs_epsilon.png")
plt.close()

# Recall vs epsilon
plt.figure(figsize=(8, 6))
sns.lineplot(data=grouped, x="epsilon", y="recall", hue="noise_multiplier", marker="^", palette=palette)
plt.title("Recall vs Privacy Budget (Epsilon)", fontsize=14)
plt.xlabel("Epsilon (ε)")
plt.ylabel("Recall")
plt.legend(title="Noise Multiplier")
plt.tight_layout()
plt.savefig("dp_results/plot_recall_vs_epsilon.png")
plt.close()
