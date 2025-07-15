# app.py
import os

import streamlit as st
import joblib

# from scripts.train_model import vectorizer
# from utils.preprocessing import clean_text

# Load model and vectorizer
clf_path = os.getcwd()
vectorizer_path = os.path.join(clf_path, 'src', 'models', 'vectorizer.pkl')
clf_path = os.path.join(clf_path, 'src', 'models', 'baseline_model.pkl')
clf = joblib.load(clf_path)
vectorizer = joblib.load(vectorizer_path)

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
        # cleaned = clean_text(email_input)
        cleaned = email_input
        vect = vectorizer.transform([cleaned])
        spam_prediction = clf.predict(vect)[0]
        prob = clf.predict_proba(vect)[0][spam_prediction]

        label = "Suspicious (Spam / Social Engineering)" if spam_prediction == 1 else "Safe / Normal"
        st.markdown(f"### Prediction: {label}")
        st.markdown(f"**Confidence:** `{prob:.3f}`")

        if spam_prediction == 1:
            st.error("This email appears malicious. Consider reporting it.")
        else:
            st.success("This email seems safe.")
