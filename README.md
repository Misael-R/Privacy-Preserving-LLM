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


## Table of Contents

- [Motivation](#motivation)
- [Project Objectives](#project-objectives)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Running the App Locally](#running-the-app-locally)
- [File Structure](#file-structure)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Motivation

Social engineering attacks are becoming more frequent and sophisticated. As these attacks often exploit private and sensitive email content, it's vital to ensure that detection models do not compromise user privacy. This project introduces a privacy-preserving solution using differential privacy, making it suitable for real-world deployment in sensitive environments such as corporate email gateways or personal assistants.

---

## Project Objectives

- Train a robust classifier to detect spam and socially engineered messages using the Enron dataset.
- Apply differential privacy mechanisms (via [Opacus](https://opacus.ai/)) to ensure training does not leak sensitive data.
- Build an interactive web application for real-time email threat analysis.
- Log and visualize the trade-off between privacy budget (ε) and model performance.

---

## Architecture Overview

The application is modular and consists of four main components:

1. **Data Preprocessing**  
   The Enron dataset is cleaned, tokenized, and vectorized using TF-IDF.

2. **Model Training**  
   Two models are trained:
   - Baseline logistic regression (`sklearn`)
   - Differentially private neural network (`PyTorch + Opacus`)

3. **Evaluation & Logging**  
   Accuracy and privacy budget (ε) are tracked, saved to CSV, and visualized.

4. **Web Interface**  
   A `Streamlit` app lets users paste email content, select the model, and get predictions along with confidence levels.

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/Misael-R/Privacy-Preserving-LLM.git
cd privacy-preserving-llm
````

2. **Create a virtual environment and activate it**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
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

3. **Go to** [http://localhost:8501](http://localhost:8501) in your browser.

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

## Acknowledgements

* Enron Email Dataset – Carnegie Mellon University
* [Opacus](https://opacus.ai/) – Differential Privacy for PyTorch
* [Streamlit](https://streamlit.io/) – Lightweight ML interface

---

## License

This project is licensed under the [APACHE License](./LICENSE).

---

<p align="center">
  <b> Data should be powerful, not dangerous.<br>
  This project aims to prove privacy can enhance security.</b>
</p>
