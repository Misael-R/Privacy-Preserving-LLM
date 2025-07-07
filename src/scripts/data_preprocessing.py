# Step 1: Load and preprocess a sample from the Enron dataset for baseline training

import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Simulate loading Enron dataset (simplified for demonstration)
# Assuming we have a CSV file for now: 'enron_spam.csv' with columns ['text', 'label']
# label: 0 = ham, 1 = spam/social engineering

sample_data = {
    'text': [
        "Hey John, please review the quarterly report before Friday.",
        "You've won a $1000 gift card! Click here to claim it now.",
        "Let's schedule a meeting with the legal department.",
        "URGENT: Your account has been compromised. Log in now to reset."
    ],
    'label': [0, 1, 0, 1]
}
df = pd.DataFrame(sample_data)

# Preprocess text: remove non-alphanumeric, lowercase
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()



# Example text processing
vectorizer = TfidfVectorizer(max_features=5000)
X_text = vectorizer.fit_transform(df['clean_body'])

# Add metadata features
X_meta = df[['domain_flag', 'hour', 'day_of_week']]
X = hstack([X_text, X_meta])

df['clean_text'] = df['text'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.25, random_state=42)

# TF-IDF vectorization for baseline
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

X_train.shape, X_train_vec.shape
