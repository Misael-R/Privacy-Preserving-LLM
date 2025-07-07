# utils/preprocessing.py

import re

def clean_text(text):
    """Lowercases and removes non-alphanumeric characters from text."""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()
