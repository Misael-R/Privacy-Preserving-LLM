<h1 align="center">Dissertation: Privacy-Preserving Multimodal LLM Agent for Social Engineering Detection</h1>
<h3 align="center">By Misael Rivera, MSc. Candidate</h3>
<h4 align="center">Acknowledgements: Dongzhu Liu, PhD. in Cybersecurity</h4>

---

## Project Overview

This project implements a **Privacy-Preserving Multimodal LLM Agent for Social Engineering Detection**. It provides a modular, interactive, and scalable prototype using:

- **Streamlit** for the front-end UI
- **Logistic Regression** as a baseline for social engineering classification
- **TF-IDF** for feature extraction
- **Opacus** for Differential Privacy in model training
- **LangChain (Next)** for LLM-based threat interpretation

> **Goal:** Empower privacy-aware AI to detect spam/social engineering attacks while preserving user data confidentiality.

---

## Motivation

Social engineering attacks are increasingly frequent and sophisticated. Because these attacks often exploit sensitive content, it's crucial to ensure detection models do not compromise user privacy. This project addresses this by integrating differential privacy into machine learning workflows for spam and social engineering detection.

---

## Project Objectives

- Train robust classifiers to detect spam and socially engineered messages using the Enron dataset.
- Apply differential privacy mechanisms with [Opacus](https://opacus.ai/) to prevent data leakage from training.
- Build an interactive web application for real-time email threat analysis.
- Log and visualize the trade-off between privacy budget (ε) and model performance.

---

## Architecture Overview

The application is modular and consists of four main components:

1. **Data Preprocessing**
   - Cleans, tokenizes, and vectorizes the Enron dataset using TF-IDF.

2. **Model Training**
   - Trains a baseline logistic regression model (scikit-learn).
   - Trains a differentially private neural network (PyTorch + Opacus).

3. **Evaluation & Logging**
   - Tracks accuracy and privacy budget (ε), saving results to CSV and visualizing trade-offs.

4. **Web Interface**
   - A Streamlit app allows users to paste email content, select models, and receive predictions and confidence metrics.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Misael-R/Privacy-Preserving-LLM.git
   cd Privacy-Preserving-LLM
   ```
2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the App Locally

1. **Preprocess the data and train the models**
   ```bash
   python src/scripts/data_preprocessing.py
   python src/scripts/baseline_classifier.py
   python src/scripts/train_dp_model.py
   ```

2. **Launch the Streamlit app**
   ```bash
   streamlit run src/main.py
   ```

3. **Open** [http://localhost:8501](http://localhost:8501) in your browser to use the app.

---

## File Structure

```
📁 Privacy-Preserving-LLM/
├── assets/
│   └── enron_spam_data/
├── src/
│   ├── models/
│   │   ├── baseline_model.pkl
│   │   ├── private_model.pt
│   │   ├── torch_model.py
│   │   └── vectorizer.pkl
│   ├── results/
│   │   ├── baseline_classifier_results.txt
│   │   ├── epsilon_accuracy_log.csv
│   │   ├── epsilon_vs_accuracy.png
│   │   ├── metrics.txt
│   │   └── results.txt
│   ├── scripts/
│   │   ├── baseline_classifier.py
│   │   ├── data_preprocessing.py
│   │   ├── train_dp_model.py
│   │   └── train_model.py
│   └── utils/
│       └── main.py
├── requirements.txt
├── LICENSE
├── README.md
└── .gitignore
```

---

## Datasets

- **Enron Email Dataset** ([Details](assets/enronSpamDataset/README.md)): Used for training and evaluating models.

---

## Acknowledgements

- Enron Email Dataset – Carnegie Mellon University
- [Opacus](https://opacus.ai/) – Differential Privacy for PyTorch
- [Streamlit](https://streamlit.io/) – Lightweight ML interface

---

## License

This project is licensed under the [APACHE License](./LICENSE).

---

<p align="center">
  <b>Data should be powerful, not dangerous.<br>
  This project aims to prove privacy can enhance security.</b>
</p>