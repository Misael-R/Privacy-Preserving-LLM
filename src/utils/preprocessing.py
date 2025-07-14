# utils/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re

def clean_text(text):
    """Lowercases and removes non-alphanumeric characters from text."""
    text = str(text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    print(text)
    return text.lower()

def clean_text2(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\W+', ' ', text)
    return text.lower()

def preprocess_enron(path="../../assets/enronSpamDataset/enron_spam_data.csv"):
    df = pd.read_csv(path)
    # df = df.dropna()
    df['Message'] = df['Message'].apply(clean_text)
    y = df['Label'].astype(int)
    X_text = df['Message']

    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(X_text)

    # Save vectorizer for Streamlit
    joblib.dump(vectorizer, "../models/vectorizer.pkl")

    # Split data: 80/10/10
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1111, stratify=y_train_val, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test