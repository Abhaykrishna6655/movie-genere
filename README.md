# 🍷 Wine Quality Classification (Machine Learning Project)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning pipeline and interactive web application to classify red wines into **Premium** vs. **Standard** quality tiers based on physicochemical properties using a **Random Forest Classifier**.

---

## 📌 Section 1: Project Overview

Wine quality assessment has traditionally relied on human sensory evaluations conducted by certified sommeliers, which can be subjective, costly, and time-intensive. This project develops an automated, objective, and reproducible classification pipeline utilizing machine learning.

By leveraging physicochemical laboratory measurements (such as alcohol content, volatile acidity, sulphates, and citric acid), a supervised **Random Forest Classifier** is trained to predict whether a given red wine meets the benchmark for **Premium Quality** ($\ge 7/10$) or represents **Standard Quality** ($< 7/10$). The system includes an interactive, production-ready **Streamlit web dashboard** allowing winemakers, quality assurance analysts, and consumers to test chemical compositions and receive real-time classification probabilities.

---

## 📊 Section 2: Dataset Information

The model is trained on the benchmark Red Wine Quality dataset:

- **Dataset Name:** Wine Quality Dataset (Red Wine Subset)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality) (Cortez et al., 2009)
- **Direct Download:** `https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv`
- **Total Records:** Exactly **1,599 records** (samples)
- **Features:** Exactly **11 physicochemical input features** + 1 sensory quality rating (converted to binary target)

### Physicochemical Feature Descriptions
| # | Feature Name | Unit | Role in Classification | Dataset Median |
|---|--------------|------|------------------------|----------------|
| 1 | `fixed acidity` | g(tartaric acid)/dm³ | Background acid profile | 7.90 |
| 2 | `volatile acidity` | g(acetic acid)/dm³ | Top Predictor (Vinegar taste at high levels) | 0.52 |
| 3 | `citric acid` | g/dm³ | Top Predictor (Freshness & crispness) | 0.26 |
| 4 | `residual sugar` | g/dm³ | Sweetness balance | 2.20 |
| 5 | `chlorides` | g(NaCl)/dm³ | Salinity / salt content | 0.079 |
| 6 | `free sulfur dioxide` | mg/dm³ | Oxidation prevention | 14.0 |
| 7 | `total sulfur dioxide` | mg/dm³ | SO₂ preservative measurement | 38.0 |
| 8 | `density` | g/cm³ | Body / relationship to alcohol & sugar | 0.9968 |
| 9 | `pH` | pH scale | Overall acidity balance (0–14) | 3.31 |
| 10 | `sulphates` | g(K₂SO₄)/dm³ | Top Predictor (Antimicrobial & flavor protector) | 0.62 |
| 11 | `alcohol` | % vol. | Top Predictor (Body, warmth, structural quality) | 10.20 |

**Target Variable Definition:**
- **Class 1 (Premium):** Quality score $\ge 7$ (217 instances, 13.6% of dataset)
- **Class 0 (Standard):** Quality score $< 7$ (1,382 instances, 86.4% of dataset)

---

## 📈 Section 3: Model Performance

The supervised model was evaluated on a stratified hold-out test set (20% of data = 320 unseen wine samples).

### Primary Evaluation Metrics
- **Accuracy Score:** **94.38%** (`0.9437`)
- **ROC-AUC Score:** **0.9554** (Demonstrating excellent class separability)
- **Macro Average F1-Score:** **0.86**
- **Weighted Average Precision:** **0.94**
- **Weighted Average Recall:** **0.94**

### Confusion Matrix
```
                       Predicted Standard (0)   Predicted Premium (1)
Actual Standard (0)             275                       2
Actual Premium (1)               16                      27
```

### Detailed Classification Report
| Class | Precision | Recall | F1-Score | Support |
|:------|:---------:|:------:|:--------:|:-------:|
| **Standard (Class 0)** | 0.95 | 0.99 | 0.97 | 277 |
| **Premium (Class 1)** | 0.93 | 0.63 | 0.75 | 43 |
| **Overall Accuracy** | — | — | **0.94** | 320 |

### Top 4 Feature Importance Ranking
1. **Alcohol (`17.37%`):** Primary driver of quality tier differentiation.
2. **Sulphates (`11.13%`):** Vital preservative maintaining organoleptic freshness.
3. **Volatile Acidity (`10.24%`):** Strong negative indicator when elevated.
4. **Citric Acid (`9.29%`):** Strong positive contributor to balanced tartness.

---

## 🌐 Section 4: Streamlit App Link

You can access and interact with the deployed live web application here:

👉 **[Click here to view the live app](https://your-app-link.streamlit.app/)**

*(Placeholder for deployment on Streamlit Community Cloud or hosting platform)*

---

## 📷 Section 5: Screenshots

Below is an interface preview of the Streamlit Wine Quality Classification dashboard showing feature input sliders, real-time probability distributions, and the Premium classification badge:

![Wine Quality Classifier Web Application](screenshots/app.png)

---

## 🚀 Getting Started Locally

### 1. Clone or Open Project Directory
```bash
cd project_folder
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Random Forest Model
To preprocess data, train the classifier, print metrics, and generate `model.pkl`:
```bash
python train.py
```
*(If `winequality-red.csv` is not present, `train.py` automatically downloads it from the UCI repository.)*

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test the application interactively.
