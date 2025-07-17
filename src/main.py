import streamlit as st
import torch
import joblib
import numpy as np
from torch.nn.functional import softmax

from models.torch_model import PrivacyAwareEmailClassifier
from utils.preprocessing import clean_text

# Load models and vectorizer
vectorizer = joblib.load("models/vectorizer.pkl")

# Load baseline model
baseline_model = joblib.load("models/baseline_model.pkl")

# Load privacy-preserving model
torch_model = PrivacyAwareEmailClassifier(input_dim=2000)
torch_model.load_state_dict(torch.load("models/private_model.pt", map_location=torch.device("cpu")))
torch_model.eval()

st.set_page_config(page_title="Email Threat Detector", layout="centered")
st.title("🔐 Email Threat Detector")
st.subheader("Detect Spam / Social Engineering Attempts")

model_choice = st.radio("Choose Model:", ["Baseline (Logistic Regression)", "Private (DP-PyTorch)"])

email_input = st.text_area("Paste the email content here:", height=250)

if st.button("Analyze"):
    if not email_input.strip():
        st.warning("Please paste an email for analysis.")
    else:
        cleaned = clean_text(email_input)
        vect = vectorizer.transform([cleaned]).toarray()

        if model_choice == "Baseline (Logistic Regression)":
            pred = baseline_model.predict(vect)[0]
            prob = baseline_model.predict_proba(vect)[0][pred]
        else:
            with torch.no_grad():
                input_tensor = torch.tensor(vect).float()
                logits = torch_model(input_tensor)
                probs = softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                prob = probs[0][pred].item()

        label = "⚠️ Suspicious (Spam / Social Engineering)" if pred == 1 else "✅ Safe / Normal"
        st.markdown(f"### Prediction: {label}")
        st.markdown(f"**Confidence:** `{prob:.3f}`")
