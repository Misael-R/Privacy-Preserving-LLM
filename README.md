<h1 align="center">Dissertation: Privacy-Preserving Multimodal LLM Agent for Social Engineering Detection</h1>
<h3 align="center">By Misael Rivera, MSc. Candidate</h3>
<h4 align="center">Acknowledgements: Dongzhu Liu, PhD. in Cybersecurity</h4>

---


## Project Overview

This repository implements a **Privacy-Preserving Multimodal LLM Agent for Social Engineering Detection**. The system is modular, interactive, and scalable, featuring:

- **Streamlit** for the web UI
- **Logistic Regression** (scikit-learn) as a baseline classifier
- **TF-IDF** (scikit-learn) for feature extraction
- **PyTorch** neural network with **Opacus** for Differential Privacy (DP) training
- **Visualization** with matplotlib and seaborn

**Goal:** Enable privacy-aware AI to detect spam/social engineering attacks while preserving user data confidentiality.

---


## Motivation

Social engineering attacks are increasingly frequent and sophisticated. Because these attacks often exploit sensitive content, it's crucial to ensure detection models do not compromise user privacy. This project addresses this by integrating differential privacy into machine learning workflows for spam and social engineering detection, using the Enron dataset as a benchmark.

---


## Project Objectives

- Train robust classifiers to detect spam and socially engineered messages using the Enron dataset
- Apply differential privacy mechanisms with [Opacus](https://opacus.ai/) to prevent data leakage from training
- Build an interactive web application for real-time email threat analysis
- Log and visualize the trade-off between privacy budget (ε) and model performance

---


## Architecture Overview

The application consists of the following main components:

1. **Data Preprocessing**
   - Cleans and vectorizes the Enron dataset using TF-IDF (see `src/utils/preprocessing.py`).

2. **Model Training**
   - Baseline: Trains a logistic regression model (`src/scripts/baseline_classifier.py`).
   - Differential Privacy: Trains a neural network with DP-SGD using Opacus (`src/scripts/train_dp_model.py`).

3. **Evaluation & Logging**
   - Logs accuracy, F1, recall, and privacy budget (ε) to CSV files.
   - Visualizes results with matplotlib/seaborn (`src/plots/`).

4. **Web Interface**
   - Streamlit app (`src/main.py`) for real-time email threat analysis, model selection, and confidence display.

---


## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Misael-R/Privacy-Preserving-LLM.git
   cd Privacy-Preserving-LLM
   ```
2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---


## Running the App Locally

1. **Preprocess the data and train the models**
   ```bash
   # Preprocess and vectorize Enron dataset
   python src/scripts/baseline_classifier.py
   # Train the DP model (PyTorch + Opacus)
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
Privacy-Preserving-LLM/
├── assets/
│   └── enronSpamDataset/
├── src/
│   ├── main.py
│   ├── models/
│   │   ├── baseline_model.pkl
│   │   ├── private_model.pt
│   │   ├── torch_model.py
│   │   └── vectorizer.pkl
│   ├── plots/
│   │   ├── baseline_plots.py
│   │   ├── dp_plots.py
│   │   ├── make_figures.py
│   │   └── ...
│   ├── results/
│   │   ├── baseline_metrics_log.csv
│   │   ├── epsilon_metrics_log.csv
│   │   └── ...
│   ├── scripts/
│   │   ├── baseline_classifier.py
│   │   ├── train_dp_model.py
│   │   ├── train_model.py
│   │   └── ...
│   └── utils/
│       ├── preprocessing.py
│       └── ...
├── requirements.txt
├── LICENSE
├── README.md
└── .gitignore
```

---


## Datasets

- **Enron Email Dataset** ([Details](assets/enronSpamDataset/README.md)): Used for training and evaluating models. Downloaded automatically via HuggingFace Datasets (`SetFit/enron_spam`).

---


## Acknowledgements

- Enron Email Dataset – Carnegie Mellon University
- [Opacus](https://opacus.ai/) – Differential Privacy for PyTorch
- [Streamlit](https://streamlit.io/) – Lightweight ML interface
- [HuggingFace Datasets](https://huggingface.co/docs/datasets) – Data loading
- [scikit-learn](https://scikit-learn.org/) – ML models and preprocessing
- [PyTorch](https://pytorch.org/) – Deep learning framework

---


## License

This project is licensed under the [Apache License 2.0](./LICENSE).

## All project info

Please address to the following GitHub Repository to find all the material and results [Repo](https://github.com/Misael-R/Privacy-Preserving-LLM/tree/main)

---

<p align="center">
   <b>Data should be powerful, not dangerous.<br>
   This project aims to prove privacy can enhance security.</b>
</p>