import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
os.makedirs("baseline_results", exist_ok=True)
df = pd.read_csv("../results/baseline_metrics_log_2.csv")

# Accuracy per fold
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="fold", y="accuracy", palette="Blues_d")
plt.title("Baseline Logistic Regression - Accuracy per Fold")
plt.ylim(0.5, 1.0)
plt.ylabel("Accuracy")
plt.xlabel("Fold")
plt.tight_layout()
plt.savefig("./baseline_results/baseline_accuracy.png")
plt.close()

# F1 score per fold
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="fold", y="f1_score", palette="Greens_d")
plt.title("Baseline Logistic Regression - F1 Score per Fold")
plt.ylim(0.5, 1.0)
plt.ylabel("F1 Score")
plt.xlabel("Fold")
plt.tight_layout()
plt.savefig("./baseline_results/baseline_f1.png")
plt.close()

# Recall per fold
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="fold", y="recall", palette="Oranges_d")
plt.title("Baseline Logistic Regression - Recall per Fold")
plt.ylim(0.5, 1.0)
plt.ylabel("Recall")
plt.xlabel("Fold")
plt.tight_layout()
plt.savefig("./baseline_results/baseline_recall.png")
plt.close()
