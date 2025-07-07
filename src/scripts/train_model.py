# train_model.py (Run once to train baseline model)

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.path.insert(0, 'utils')
from preprocessing import clean_text

# Sample or full Enron dataset
df = pd.read_csv('enron_spam.csv')  # ['text', 'label']
df['clean_text'] = df['text'].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Train model
clf = LogisticRegression()
clf.fit(X, y)

# Save model and vectorizer
joblib.dump(clf, '../models/baseline_model.pkl')
joblib.dump(vectorizer, '../models/vectorizer.pkl')
