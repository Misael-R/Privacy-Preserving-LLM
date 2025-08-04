# train_dp_model.py

import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from opacus import PrivacyEngine
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

from preprocessing import preprocess_enron
from torch_model import PrivacyAwareEmailClassifier

# === Configuration ===
N_FOLDS = 5
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
DELTA = 1e-5

# Try multiple noise multipliers to explore privacy/utility tradeoff
NOISE_MULTIPLIERS = [0.5, 1.0, 2.0]

# Prepare result directory
os.makedirs("results", exist_ok=True)

# Load and preprocess data
X_train_all, X_val_all, X_test_all, y_train_all, y_val_all, y_test_all = preprocess_enron()
X_all = X_train_all  # we'll do CV on the train+val split
y_all = y_train_all

# Convert entire feature set to numpy for CV indexing
X_np = X_all.toarray()
y_np = y_all.values

# Prepare to collect logs
logs = []

# Cross-validation loop
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
for noise in NOISE_MULTIPLIERS:
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y_np), start=1):
        # Build fold-specific datasets
        X_train_fold = torch.tensor(X_np[train_idx]).float()
        y_train_fold = torch.tensor(y_np[train_idx]).long()
        X_val_fold   = torch.tensor(X_np[val_idx]).float()
        y_val_fold   = torch.tensor(y_np[val_idx]).long()

        train_loader = DataLoader(
            TensorDataset(X_train_fold, y_train_fold),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        # Instantiate model & components
        model = PrivacyAwareEmailClassifier(input_dim=X_np.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        # Wrap with Opacus for DP-SGD
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise,
            max_grad_norm=1.0
        )

        # Training + logging per epoch
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            # Compute current privacy budget
            epsilon = privacy_engine.get_epsilon(delta=DELTA)

            # Evaluate on fold validation set
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_fold)
                val_preds = torch.argmax(val_logits, dim=1).numpy()
            acc  = accuracy_score(y_val_fold, val_preds)
            f1   = f1_score(y_val_fold, val_preds)
            rec  = recall_score(y_val_fold, val_preds)

            # Record log entry
            logs.append({
                "noise_multiplier": noise,
                "fold": fold,
                "epoch": epoch,
                "epsilon": epsilon,
                "delta": DELTA,
                "accuracy": acc,
                "f1_score": f1,
                "recall": rec
            })

            print(f"[Noise {noise}] Fold {fold} Epoch {epoch} → Epsilon={epsilon:.2f}, Acc={acc:.3f}, F1={f1:.3f}, Recall={rec:.3f}")

# Save consolidated log
df_logs = pd.DataFrame(logs)
df_logs.to_csv("../results/epsilon_metrics_log.csv", index=False)

print("\nAll folds & hyperparams complete. Logs saved to results/epsilon_metrics_log.csv")
