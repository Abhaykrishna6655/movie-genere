"""
Movie Genre Classification - Model Training Script
Author: Machine Learning University Assignment
Topic: Predicting Movie Genre from Plot Summary using TF-IDF & Multinomial Naive Bayes
"""

import os
import re
import string
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# NLTK Stopwords
try:
    import nltk
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOP_WORDS = set(ENGLISH_STOP_WORDS)

DATASET_FILE = "dataset.csv"
KAGGLE_TXT_FILE = "train_data.txt"
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"


def clean_text(text: str) -> str:
    """
    Cleans raw text by:
    1. Converting to lowercase
    2. Removing HTML tags and URLs
    3. Stripping punctuation and numeric digits
    4. Removing English stop words
    5. Normalizing multiple whitespaces
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Strip URLs & HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    
    # 3. Keep only alphabetic characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    
    # 4. Tokenize & remove stop words
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 2]
    
    return " ".join(tokens)


def load_dataset():
    """
    Loads movie plot data.
    Supports:
      - dataset.csv (standard CSV format with 'genre' and 'plot'/'description')
      - train_data.txt (Kaggle Genre Classification dataset delimiter ':::')
    """
    df = None
    
    # Check for Kaggle train_data.txt format (ID ::: TITLE ::: GENRE ::: DESCRIPTION)
    if os.path.exists(KAGGLE_TXT_FILE):
        print(f"[*] Detected Kaggle format file: '{KAGGLE_TXT_FILE}'")
        df = pd.read_csv(
            KAGGLE_TXT_FILE,
            sep=r":::",
            engine="python",
            names=["id", "title", "genre", "plot"],
            index_col=False
        )
    elif os.path.exists(DATASET_FILE):
        print(f"[*] Loading dataset from: '{DATASET_FILE}'")
        df = pd.read_csv(DATASET_FILE)
    else:
        raise FileNotFoundError(
            f"[-] Neither '{DATASET_FILE}' nor '{KAGGLE_TXT_FILE}' was found. "
            "Please ensure dataset.csv is in the project folder."
        )

    # Standardize column names
    col_mapping = {}
    for col in df.columns:
        clean_col = col.strip().lower()
        if clean_col in ["genre", "genres", "category"]:
            col_mapping[col] = "genre"
        elif clean_col in ["plot", "description", "summary", "overview", "plot_summary"]:
            col_mapping[col] = "plot"
        elif clean_col in ["title", "name", "movie_name"]:
            col_mapping[col] = "title"

    df = df.rename(columns=col_mapping)

    if "genre" not in df.columns or "plot" not in df.columns:
        raise ValueError(
            f"[-] Required columns ('genre' and 'plot'/'description') not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Clean genre labels
    df["genre"] = df["genre"].astype(str).str.strip().str.title()
    df["plot"] = df["plot"].astype(str).str.strip()

    # Drop missing or empty entries
    df = df.dropna(subset=["genre", "plot"])
    df = df[(df["genre"] != "") & (df["plot"] != "")]

    # If dataset has numerous rare classes with < 2 samples, filter them for stratified split
    genre_counts = df["genre"].value_counts()
    valid_genres = genre_counts[genre_counts >= 2].index
    df = df[df["genre"].isin(valid_genres)]

    print(f"[+] Loaded {len(df):,} movie records across {df['genre'].nunique()} distinct genres.")
    print(f"[+] Top Genres:\n{df['genre'].value_counts().head(10).to_string()}\n")
    return df


def train_model():
    """
    End-to-end training pipeline:
    1. Load dataset
    2. Clean text data
    3. Train-test split (80-20 stratified)
    4. Fit TF-IDF Vectorizer
    5. Train Multinomial Naive Bayes Classifier
    6. Evaluate and print detailed metrics
    7. Save model.pkl and vectorizer.pkl
    """
    print("=" * 70)
    print("MOVIE GENRE CLASSIFICATION - TRAINING PIPELINE")
    print("Algorithm: Multinomial Naive Bayes with TF-IDF Vectorization")
    print("=" * 70 + "\n")

    # 1. Load data
    df = load_dataset()

    # 2. Text preprocessing
    print("[*] Preprocessing movie plot summaries (cleaning, tokenizing, removing stop words)...")
    df["clean_plot"] = df["plot"].apply(clean_text)
    
    # Filter out empty cleaned plots
    df = df[df["clean_plot"].str.strip() != ""]

    X = df["clean_plot"]
    y = df["genre"]

    # 3. Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Training samples : {len(X_train):,}")
    print(f"[+] Testing samples  : {len(X_test):,}")

    # 4. Feature Extraction: TF-IDF Vectorization
    print("\n[*] Vectorizing plot summaries using TF-IDF...")
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english"
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"[+] TF-IDF vocabulary size : {len(tfidf.vocabulary_):,} features")
    print(f"[+] Feature matrix shape   : {X_train_tfidf.shape}")

    # 5. Classifier Training: Multinomial Naive Bayes
    print("\n[*] Training Multinomial Naive Bayes Classifier (alpha=0.5)...")
    clf = MultinomialNB(alpha=0.5)
    clf.fit(X_train_tfidf, y_train)
    print("[+] Model training complete.")

    # 6. Evaluation
    y_pred = clf.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS: Accuracy = {accuracy * 100:.2f}%")
    print("=" * 70)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 7. Persist Artifacts
    print("[*] Saving trained model and vectorizer...")
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(tfidf, VECTORIZER_PATH)
    print(f"[+] Model saved to       : '{MODEL_PATH}' ({os.path.getsize(MODEL_PATH) / 1024:.1f} KB)")
    print(f"[+] Vectorizer saved to  : '{VECTORIZER_PATH}' ({os.path.getsize(VECTORIZER_PATH) / 1024:.1f} KB)")
    print("\n[SUCCESS] Training pipeline completed successfully!\n")


if __name__ == "__main__":
    train_model()
