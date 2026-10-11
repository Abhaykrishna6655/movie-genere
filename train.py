"""
Wine Quality Classification - Model Training Script
Author: Machine Learning Assignment Project
Topic: Binary Classification of Wine Quality using Random Forest
"""

import os
import urllib.request
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

DATASET_FILENAME = "winequality-red.csv"
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
MODEL_FILENAME = "model.pkl"


def load_dataset(filepath=DATASET_FILENAME):
    """
    Loads the red wine dataset. If the file is not present locally,
    downloads it directly from the UCI Machine Learning Repository.
    """
    if not os.path.exists(filepath):
        print(f"[-] '{filepath}' not found locally.")
        print(f"[*] Downloading dataset from UCI Repository: {DATASET_URL}...")
        try:
            urllib.request.urlretrieve(DATASET_URL, filepath)
            print(f"[+] Download complete: '{filepath}' saved successfully.\n")
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to auto-download dataset. Please place '{filepath}' in this directory. Error: {e}"
            )
    else:
        print(f"[+] Found dataset file: '{filepath}'\n")

    # Detect delimiter (; is standard in UCI Wine Quality, but fallback to , if modified)
    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline()
    delimiter = ";" if ";" in first_line else ","

    df = pd.read_csv(filepath, sep=delimiter)
    print(f"[+] Dataset loaded successfully with shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def preprocess_data(df):
    """
    Prepares features and converts the multiclass 'quality' score
    into a binary classification target:
        - 1: Premium Wine (quality >= 7)
        - 0: Standard Wine (quality < 7)
    """
    # Create binary target
    df["target"] = (df["quality"] >= 7).astype(int)

    # Feature matrix X and target vector y
    X = df.drop(columns=["quality", "target"])
    y = df["target"]

    print("\n[+] Target Distribution:")
    counts = y.value_counts()
    print(f"    - Standard (0, quality < 7) : {counts.get(0, 0)} ({counts.get(0, 0) / len(y) * 100:.1f}%)")
    print(f"    - Premium  (1, quality >= 7): {counts.get(1, 0)} ({counts.get(1, 0) / len(y) * 100:.1f}%)")

    return X, y


def train_and_evaluate(X, y):
    """
    Splits data, trains a Random Forest Classifier, and evaluates performance.
    """
    # 80-20 Train/Test split with stratification to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n[+] Train set size: {X_train.shape[0]} samples")
    print(f"[+] Test set size : {X_test.shape[0]} samples")

    # Initialize and train Random Forest Classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=None,
        n_jobs=-1
    )

    print("\n[*] Training Random Forest Classifier...")
    rf_classifier.fit(X_train, y_train)
    print("[+] Model training completed.")

    # Predictions
    y_pred = rf_classifier.predict(X_test)
    y_proba = rf_classifier.predict_proba(X_test)[:, 1]

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cr = classification_report(y_test, y_pred, target_names=["Standard (0)", "Premium (1)"])

    print("\n" + "=" * 50)
    print("           MODEL EVALUATION METRICS")
    print("=" * 50)
    print(f"Accuracy Score : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"ROC-AUC Score  : {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"                 Predicted 0   Predicted 1")
    print(f"  Actual 0 (Std)     {cm[0, 0]:<12} {cm[0, 1]}")
    print(f"  Actual 1 (Prem)    {cm[1, 0]:<12} {cm[1, 1]}")

    print("\nDetailed Classification Report:")
    print(cr)
    print("=" * 50)

    # Top Feature Importances
    importances = pd.Series(rf_classifier.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n[+] Top Feature Importances:")
    for feature, score in importances.items():
        print(f"    - {feature:<22}: {score:.4f} ({score * 100:.1f}%)")

    return rf_classifier


def save_model(model, filepath=MODEL_FILENAME):
    """
    Serializes and saves the trained model to disk using joblib.
    """
    joblib.dump(model, filepath)
    print(f"\n[+] Model successfully saved as '{filepath}' ({os.path.getsize(filepath) / 1024:.1f} KB).")


def main():
    print("=" * 50)
    print(" Red Wine Quality Classification - Training Pipeline")
    print("=" * 50)
    df = load_dataset(DATASET_FILENAME)
    X, y = preprocess_data(df)
    model = train_and_evaluate(X, y)
    save_model(model, MODEL_FILENAME)
    print("\n[OK] Pipeline executed successfully!")


if __name__ == "__main__":
    main()
