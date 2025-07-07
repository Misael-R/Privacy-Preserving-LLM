<h1 align="center">Dissertation: Privacy-Preserving Multimodal LLM Agent for Social Engineering Detection</h1>
<h3> By Misael Rivera, MSc. Candidate </h3>
<h3> Acknowledgements: Dongzhu Liu, PhD. in Cybersecurity </h3>
---

## Project Overview

This project implements a **Privacy-Preserving Multimodal LLM Agent for Social Engineering Detection**. It provides a modular, interactive, and scalable prototype built with:

* **Streamlit** for the front-end UI
* **Logistic Regression** baseline for social engineering classification
* **TF-IDF** feature extraction
* **Opacus** for applying Differential Privacy in training
* **LangChain (Next)** for LLM-enhanced threat interpretation

> **Goal:** Enable privacy-aware AI that detects spam/social engineering attacks, while respecting user data confidentiality.

---

## Project Structure

```bash
📂 privacy_email_detector/
├── 📁 models/               # Stored models (baseline + private)
│   └── baseline_model.pkl
│   └── vectorizer.pkl
├── 📁 utils/                # Custom helper modules
│   └── preprocessing.py
├── 📄 train_model.py        # Model training script
├── 📄 app.py                # Streamlit UI entrypoint
├── 📄 requirements.txt      # All dependencies
└── 📄 README.md             # This file
```

---

## Baseline Training Pipeline

1. **Preprocessing:**

```python
# utils/preprocessing.py
import re

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()
```

2. **Train & Save Model**

```python
# train_model.py
import joblib, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.preprocessing import clean_text

df = pd.read_csv('enron_spam.csv')
df['clean_text'] = df['text'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

clf = LogisticRegression()
clf.fit(X, y)

joblib.dump(clf, 'models/baseline_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
```

---

## Streamlit UI - `main.py`

```python
import streamlit as st
import joblib
from utils.preprocessing import clean_text

clf = joblib.load('models/baseline_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

st.set_page_config(page_title="Email Threat Detector", layout="centered")
st.title("🔐 Email Threat Detector")
st.subheader("Detect Spam / Social Engineering Attempts")

email_input = st.text_area("Paste the email content here:", height=250)

if st.button("Analyze"):
    if not email_input.strip():
        st.warning("Please paste an email for analysis.")
    else:
        cleaned = clean_text(email_input)
        vect = vectorizer.transform([cleaned])
        pred = clf.predict(vect)[0]
        prob = clf.predict_proba(vect)[0][pred]

        label = "⚠️ Suspicious (Spam / Social Engineering)" if pred == 1 else "✅ Safe / Normal"
        st.markdown(f"### Prediction: {label}")
        st.markdown(f"**Confidence:** `{prob:.3f}`")
```

---

## Differential Privacy with Opacus

### Why Opacus?

Opacus adds **differentially private noise** to the gradients during training, ensuring data cannot be reverse-engineered.

### Install Required Libraries

```bash
pip install opacus torch scikit-learn
```

### Convert to PyTorch + Opacus Pipeline

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from opacus import PrivacyEngine
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.preprocessing import clean_text

# Load and preprocess dataset
df = pd.read_csv('enron_spam.csv')
df['clean_text'] = df['text'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['clean_text']).toarray()
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define basic NN
model = nn.Sequential(
    nn.Linear(2000, 256),
    nn.ReLU(),
    nn.Linear(256, 2)
)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Apply PrivacyEngine
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0
)

# Training loop (simplified)
model.train()
for epoch in range(5):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

Save the model with `torch.save(model.state_dict(), 'models/private_model.pt')`

---

## Evaluation Plan

| Metric             | Purpose                                |
| ------------------ | -------------------------------------- |
| Accuracy / F1      | Compare Baseline vs Private Model      |
| Privacy Budget (ε) | Demonstrate privacy guarantee of model |
| LLM Explanation    | (Next step) Human-readable reasoning   |

---

## Next Phase: LangChain + LLM

* Embed GPT responses for reasoning chain: "Why is this email suspicious?"
* Inject metadata (sender, domain, time) for multimodal context
* Provide explainability to user for trustworthiness

---

## Tools Used

| Tool         | Purpose                          |
| ------------ | -------------------------------- |
| Streamlit    | Interactive dashboard UI         |
| Scikit-learn | Baseline model & preprocessing   |
| Opacus       | Differential Privacy training    |
| PyTorch      | Deep Learning backend for Opacus |
| LangChain    | (Next) LLM integration           |

---

## Future Deliverables

* ✅ Baseline + UI
* ✅ Private Training with Opacus
* Multimodal Metadata
* LangChain Integration
* Deployment Option (Docker/Heroku)

---

<p align="center">
  <b> Data should be powerful, not dangerous.<br>
  This project aims to prove privacy can enhance security.</b>
</p>
