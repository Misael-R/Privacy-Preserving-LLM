# Step 2: Train a baseline classifier (Logistic Regression) and define a spam_prediction function

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from preprocessing import preprocess_enron

X_train, X_val, X_test, y_train, y_val, y_test = preprocess_enron()

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "../models/baseline_model.pkl")

# Evaluate on test set
y_pred = model.predict(X_test)
print("\nBaseline Evaluation on Test Set:")
print(classification_report(y_test, y_pred))

with open("../results/baseline_classifier_results.txt", "w") as f:
    f.write("\nBaseline Evaluation on Test Set:")
    f.write("=== Baseline Model ===\n")
    f.write(classification_report(y_test, y_pred))
