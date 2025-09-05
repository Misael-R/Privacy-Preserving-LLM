# train_baseline.py

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import preprocess_enron

# === Load Data ===
X_train, X_val, X_test, y_train, y_val, y_test = preprocess_enron()

# Combine for cross-validation
X_all = X_train
y_all = y_train

# === Cross-Validation ===
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
logs = []
all_true = []
all_pred = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y_all), start=1):
    X_train_fold = X_all[train_idx]
    y_train_fold = y_all.iloc[train_idx]
    X_val_fold = X_all[val_idx]
    y_val_fold = y_all.iloc[val_idx]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_fold, y_train_fold)

    val_preds = clf.predict(X_val_fold)

    acc = accuracy_score(y_val_fold, val_preds)
    f1 = f1_score(y_val_fold, val_preds)
    rec = recall_score(y_val_fold, val_preds)

    logs.append({
        "fold": fold,
        "accuracy": acc,
        "f1_score": f1,
        "recall": rec
    })

    all_true.extend(y_val_fold)
    all_pred.extend(val_preds)

    print(f"[Baseline] Fold {fold} → Acc={acc:.3f}, F1={f1:.3f}, Recall={rec:.3f}")
    joblib.dump(clf, f"../models/logistic_regression_baseline_fold{fold}.pkl")

# Save logs
df_logs = pd.DataFrame(logs)
os.makedirs("../results", exist_ok=True)  # Ensure directory exists
df_logs.to_csv("../results/baseline_metrics_log.csv", index=False)

# === Confusion Matrix ===
os.makedirs("../plots/baseline_results", exist_ok=True)
cm = confusion_matrix(all_true, all_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Baseline Confusion Matrix (CV)")
plt.tight_layout()
plt.savefig("../plots/baseline_results/baseline_confusion_matrix.png")
plt.close()