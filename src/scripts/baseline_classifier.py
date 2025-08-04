# train_baseline.py

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
import os
import numpy as np

from preprocessing import preprocess_enron


# === Load Data ===
X_train, X_val, X_test, y_train, y_val, y_test = preprocess_enron()

# Combine for cross-validation
X_all = X_train
y_all = y_train

# Convert sparse matrix to array if needed (or use proper sparse matrix indexing)
# Either option 1: Convert to array (if memory allows)
# X_all = X_all.toarray()

# Or option 2: Keep as sparse but use proper indexing

# === Cross-Validation ===
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
logs = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y_all), start=1):
    # For sparse matrices, use these indexing methods instead of iloc
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

    print(f"[Baseline] Fold {fold} → Acc={acc:.3f}, F1={f1:.3f}, Recall={rec:.3f}")

# Save logs
df_logs = pd.DataFrame(logs)
os.makedirs("../results", exist_ok=True)  # Ensure directory exists
df_logs.to_csv("../results/baseline_metrics_log.csv", index=False)