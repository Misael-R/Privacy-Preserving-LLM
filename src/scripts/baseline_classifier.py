# Step 2: Train a baseline classifier (Logistic Regression) and define a prediction function

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Train logistic regression
clf = LogisticRegression()
clf.fit(X_train_vec, y_train)

# Evaluate on test data
y_pred = clf.predict(X_test_vec)
report = classification_report(y_test, y_pred, output_dict=True)

# Create a prediction function
def predict_email_class(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = clf.predict(vectorized)[0]
    probability = clf.predict_proba(vectorized)[0][prediction]
    return 'Spam/Social Engineering' if prediction == 1 else 'Normal', round(probability, 3)

report, predict_email_class("Dear user, your account was flagged for suspicious activity.")
