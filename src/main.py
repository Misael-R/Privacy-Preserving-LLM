# app.py

import streamlit as st
import joblib
from utils.preprocessing import clean_text

# Load model and vectorizer
clf = joblib.load('models/baseline_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# Streamlit UI
st.set_page_config(page_title="LLM Email Threat Detector", layout="centered")
st.title("🔐 Email Threat Detector")
st.subheader("Detect Spam / Social Engineering Attempts")

# Email input
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

        if pred == 1:
            st.error("This email appears malicious. Consider reporting it.")
        else:
            st.success("This email seems safe.")
