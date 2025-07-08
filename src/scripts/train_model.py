# train_model.py (Run once to train baseline model)

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import clean_text

# splits = {'train': 'train.csv', 'test': 'test.csv'}
# df = pd.read_csv("hf://datasets/adamlouly/enron_spam_data/" + splits["train"])

# Sample or full Enron dataset
df = pd.read_csv('../../assets/enronSpamDataset/enron_spam_data.csv')  # ['Message', 'Label']
df['clean_text'] = df['Message'].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['Label']

# Train model
clf = LogisticRegression()
clf.fit(X, y)

# Save model and vectorizer
joblib.dump(clf, '../models/baseline_model.pkl')
joblib.dump(vectorizer, '../models/vectorizer.pkl')
