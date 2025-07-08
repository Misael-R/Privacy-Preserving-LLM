# utils/preprocessing.py

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
