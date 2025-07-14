# train_dp_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report
import joblib
import numpy as np

from preprocessing import preprocess_enron
from torch_model import PrivacyAwareEmailClassifier

X_train, X_val, X_test, y_train, y_val, y_test = preprocess_enron()

X_train_tensor = torch.tensor(X_train.toarray()).float()
y_train_tensor = torch.tensor(y_train.values).long()
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = PrivacyAwareEmailClassifier(input_dim=2000)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)

# Training loop
for epoch in range(5):
    model.train()
    for batch in train_loader:
        x, y = batch
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    epsilon, best_alpha = privacy_engine.get_privacy_spent(delta=1e-5)
    print(f"Epoch {epoch + 1} - epsilon = {epsilon:.2f}, δ = 1e-5")

# Save model
torch.save(model.state_dict(), "models/private_model.pt")

# Evaluation
model.eval()
X_test_tensor = torch.tensor(X_test.toarray()).float()
y_test_tensor = torch.tensor(y_test.values).long()

with torch.no_grad():
    y_pred_probs = model(X_test_tensor)
    y_preds = torch.argmax(y_pred_probs, dim=1).numpy()

print("\nDP Model Evaluation on Test Set:")
print(classification_report(y_test_tensor, y_preds))

# Save metrics and epsilon
with open("../results/metrics.txt", "w") as f:
    f.write(f"Privacy Budget: epsilon = {epsilon:.2f}, δ = 1e-5\n\n")
    f.write("=== Baseline Model ===\n")
    f.write("=== DP Model ===\n")
    f.write(classification_report(y_test_tensor, y_preds))
