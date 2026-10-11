"""
Wine Quality Classification - Streamlit Web Application
Author: Machine Learning Assignment Project
Topic: Real-Time Wine Quality Prediction using Random Forest
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------------------------------------
# 1. Page Configuration & Custom Styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Wine Quality Classifier",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and clean typography
st.markdown("""
<style>
    /* Global Styles */
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #722F37; /* Wine Burgundy */
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    .prediction-premium {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
        color: #ffffff;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(45, 106, 79, 0.25);
        margin-top: 1rem;
    }
    .prediction-standard {
        background: linear-gradient(135deg, #343a40 0%, #495057 100%);
        color: #ffffff;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(52, 58, 64, 0.25);
        margin-top: 1rem;
    }
    .badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        font-size: 0.85rem;
        font-weight: 700;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-gold {
        background-color: #d4af37;
        color: #000000;
    }
    .badge-silver {
        background-color: #adb5bd;
        color: #212529;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. Model Loading
# ------------------------------------------------------------------------------
MODEL_PATH = "model.pkl"

@st.cache_resource
def load_model():
    """Loads the pre-trained Random Forest model from model.pkl."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}: {e}")
        return None

model = load_model()

# Median baseline values calculated from the 1,599 records in UCI winequality-red dataset
MEDIAN_DEFAULTS = {
    "fixed acidity": 7.90,
    "residual sugar": 2.20,
    "chlorides": 0.079,
    "free sulfur dioxide": 14.0,
    "total sulfur dioxide": 38.0,
    "density": 0.99675,
    "pH": 3.31
}

# Complete expected feature list in exact training order
ALL_FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]


# ------------------------------------------------------------------------------
# 3. Sidebar: Project Info & Background Parameters
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=500&auto=format&fit=crop&q=60", 
             caption="Fine Wine Chemistry")
    st.markdown("### 🍷 Model Information")
    st.markdown("""
    - **Algorithm:** Random Forest Classifier
    - **Estimators:** 100 Decision Trees
    - **Dataset:** UCI Red Wine Quality (1,599 records)
    - **Benchmark Accuracy:** ~94.4%
    - **Target:** 
      - **1 (Premium):** Quality score ≥ 7
      - **0 (Standard):** Quality score < 7
    """)

    st.markdown("---")
    st.markdown("### ⚙️ Fixed Baseline Features (Medians)")
    st.caption("Values fixed to dataset medians for non-primary features:")
    for feat, val in MEDIAN_DEFAULTS.items():
        st.text(f"• {feat.title()}: {val}")


# ------------------------------------------------------------------------------
# 4. Main Application Interface
# ------------------------------------------------------------------------------
st.markdown('<div class="main-header">🍷 Wine Quality Classification System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict whether a red wine is <b>Premium</b> (Quality ≥ 7) or <b>Standard</b> (Quality &lt; 7) using machine learning based on chemical properties.</div>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ **Model file `model.pkl` not found!**")
    st.info("Please generate the model by running: `python train.py` in your terminal.")
    st.stop()

# Layout: Split into Feature Controls (left) and Prediction Output (right)
col_controls, col_prediction = st.columns([1.1, 1], gap="large")

with col_controls:
    st.markdown("### 🧪 Top 4 Key Chemical Predictors")
    st.write("Adjust the most influential physicochemical properties identified by Random Forest feature importance:")

    # Slider 1: Alcohol
    alcohol = st.slider(
        "1. Alcohol (% by volume)",
        min_value=8.0,
        max_value=15.0,
        value=10.2,
        step=0.1,
        help="Feature Importance: ~17.4%. Higher alcohol content strongly correlates with premium quality ratings."
    )

    # Slider 2: Sulphates
    sulphates = st.slider(
        "2. Sulphates (g(potassium sulphate)/dm³)",
        min_value=0.30,
        max_value=2.00,
        value=0.62,
        step=0.01,
        help="Feature Importance: ~11.1%. Potassium sulphate acts as an antimicrobial and antioxidant agent."
    )

    # Slider 3: Volatile Acidity
    volatile_acidity = st.slider(
        "3. Volatile Acidity (g(acetic acid)/dm³)",
        min_value=0.10,
        max_value=1.60,
        value=0.52,
        step=0.01,
        help="Feature Importance: ~10.2%. High acetic acid content imparts an undesirable pungent, vinegar-like taste."
    )

    # Slider 4: Citric Acid
    citric_acid = st.slider(
        "4. Citric Acid (g/dm³)",
        min_value=0.00,
        max_value=1.00,
        value=0.26,
        step=0.01,
        help="Feature Importance: ~9.3%. Adds 'freshness' and crispness to red wine flavor profiles."
    )

    # Real-time feature summary preview
    with st.expander("📋 View Full 11-Feature Input Vector"):
        input_data = {
            "fixed acidity": [MEDIAN_DEFAULTS["fixed acidity"]],
            "volatile acidity": [volatile_acidity],
            "citric acid": [citric_acid],
            "residual sugar": [MEDIAN_DEFAULTS["residual sugar"]],
            "chlorides": [MEDIAN_DEFAULTS["chlorides"]],
            "free sulfur dioxide": [MEDIAN_DEFAULTS["free sulfur dioxide"]],
            "total sulfur dioxide": [MEDIAN_DEFAULTS["total sulfur dioxide"]],
            "density": [MEDIAN_DEFAULTS["density"]],
            "pH": [MEDIAN_DEFAULTS["pH"]],
            "sulphates": [sulphates],
            "alcohol": [alcohol]
        }
        input_df = pd.DataFrame(input_data)[ALL_FEATURES]
        st.dataframe(input_df.T.rename(columns={0: "Input Value"}))

with col_prediction:
    st.markdown("### 🎯 Classification Results")

    # Build input DataFrame
    input_data = {
        "fixed acidity": [MEDIAN_DEFAULTS["fixed acidity"]],
        "volatile acidity": [volatile_acidity],
        "citric acid": [citric_acid],
        "residual sugar": [MEDIAN_DEFAULTS["residual sugar"]],
        "chlorides": [MEDIAN_DEFAULTS["chlorides"]],
        "free sulfur dioxide": [MEDIAN_DEFAULTS["free sulfur dioxide"]],
        "total sulfur dioxide": [MEDIAN_DEFAULTS["total sulfur dioxide"]],
        "density": [MEDIAN_DEFAULTS["density"]],
        "pH": [MEDIAN_DEFAULTS["pH"]],
        "sulphates": [sulphates],
        "alcohol": [alcohol]
    }
    input_df = pd.DataFrame(input_data)[ALL_FEATURES]

    # Predict
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    prob_standard = probabilities[0] * 100
    prob_premium = probabilities[1] * 100

    # Display visually clear prediction card
    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-premium">
            <span class="badge badge-gold">★ Top Tier Selection ★</span>
            <h1 style="font-size: 2.4rem; margin: 0.6rem 0 0.2rem 0; color: #ffffff;">PREMIUM WINE</h1>
            <p style="font-size: 1.1rem; opacity: 0.95; margin-bottom: 1.2rem;">Quality Score Classification: <b>Score ≥ 7 / 10</b></p>
            <div style="background: rgba(255,255,255,0.15); border-radius: 10px; padding: 1rem; margin-top: 0.5rem;">
                <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">Model Confidence</div>
                <div class="metric-val">{prob_premium:.1f}%</div>
                <div style="font-size: 0.85rem; opacity: 0.9;">Probability of Superior Grade</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-standard">
            <span class="badge badge-silver">Standard Grade</span>
            <h1 style="font-size: 2.4rem; margin: 0.6rem 0 0.2rem 0; color: #ffffff;">STANDARD WINE</h1>
            <p style="font-size: 1.1rem; opacity: 0.95; margin-bottom: 1.2rem;">Quality Score Classification: <b>Score &lt; 7 / 10</b></p>
            <div style="background: rgba(255,255,255,0.15); border-radius: 10px; padding: 1rem; margin-top: 0.5rem;">
                <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">Model Confidence</div>
                <div class="metric-val">{prob_standard:.1f}%</div>
                <div style="font-size: 0.85rem; opacity: 0.9;">Probability of Standard Grade</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Probability Distribution")
    st.progress(int(prob_premium))
    p_col1, p_col2 = st.columns(2)
    p_col1.metric("Standard (0) Probability", f"{prob_standard:.1f}%")
    p_col2.metric("Premium (1) Probability", f"{prob_premium:.1f}%")

    st.info(
        "💡 **Interpretation Tip:** High alcohol (> 11.5%) coupled with high sulphates (> 0.70 g/dm³) "
        "and low volatile acidity (< 0.40 g/dm³) significantly increases the likelihood of achieving Premium status."
    )

st.markdown("---")
st.caption("Machine Learning Academic Project • Random Forest Classification on UCI Wine Quality Dataset • Built with Streamlit")
