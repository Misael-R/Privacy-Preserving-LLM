import os.path

import streamlit as st
import torch
import joblib
import numpy as np
from torch.nn.functional import softmax

from models.torch_model import PrivacyAwareEmailClassifier
from utils.preprocessing import clean_text

# Load models and vectorizer
vect_path = os.path.join(os.path.dirname(__file__), "models", "vectorizer.pkl")
vectorizer = joblib.load(vect_path)

# Load baseline model
baseline_path = os.path.join(os.path.dirname(__file__), "models", "baseline_model.pkl")
baseline_model = joblib.load(baseline_path)

# Load privacy-preserving model
torch_model_path = os.path.join(os.path.dirname(__file__), "models", "private_model.pt")
torch_model = PrivacyAwareEmailClassifier(input_dim=2000)

# Fix loading for Opacus-trained models
state_dict = torch.load(torch_model_path, map_location=torch.device("cpu"))
# Remove "_module." prefix added by Opacus
new_state_dict = {k.replace("_module.", ""): v for k, v in state_dict.items()}
torch_model.load_state_dict(new_state_dict)
# torch_model.load_state_dict(torch.load(torch_model_path, map_location=torch.device("cpu")))
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
