"""
Movie Genre Classification - Streamlit Web Application
Author: Machine Learning University Assignment
Topic: Predicting Movie Genre from Plot Summary using TF-IDF & Multinomial Naive Bayes
"""

import os
import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Configure Streamlit page layout and appearance
st.set_page_config(
    page_title="CineGenre AI - Movie Genre Classifier",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern cinematic aesthetics
st.markdown(
    """
    <style>
    /* Global Styles & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=Cinzel:wght@600;800&display=swap');

    .cinematic-header {
        font-family: 'Cinzel', serif;
        font-weight: 800;
        letter-spacing: 1.5px;
        background: linear-gradient(135deg, #ff4b4b 0%, #ff8533 50%, #f9d423 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-hero {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 25px;
    }
    .badge-tag {
        display: inline-block;
        background: rgba(255, 75, 75, 0.12);
        color: #ff4b4b;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        border: 1px solid rgba(255, 75, 75, 0.25);
    }
    .card-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .genre-pill {
        font-size: 1.8rem;
        font-weight: 700;
        padding: 10px 24px;
        border-radius: 12px;
        text-align: center;
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.2), rgba(255, 133, 51, 0.2));
        border: 1px solid rgba(255, 75, 75, 0.4);
        margin-top: 10px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# Emojis associated with genres
GENRE_EMOJIS = {
    "Action": "💥",
    "Adventure": "🗺️",
    "Animation": "🎨",
    "Comedy": "😂",
    "Crime": "🕵️‍♂️",
    "Documentary": "📹",
    "Drama": "🎭",
    "Family": "👨‍👩‍👧‍👦",
    "Fantasy": "🧙‍♂️",
    "History": "📜",
    "Horror": "👻",
    "Music": "🎵",
    "Musical": "💃",
    "Mystery": "🔍",
    "Romance": "❤️",
    "Sci-Fi": "🚀",
    "Sport": "🏆",
    "Thriller": "⚡",
    "War": "🪖",
    "Western": "🤠",
}


@st.cache_resource(show_spinner="Loading trained ML model and TF-IDF vectorizer...")
def load_artifacts():
    """Loads and caches the model and vectorizer artifacts."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None


def clean_input_text(text: str) -> str:
    """Preprocesses input text identical to training preprocessing."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = text.split()
    return " ".join(tokens)


# Pre-defined test sample plots for one-click testing
SAMPLE_PLOTS = {
    "🚀 Sci-Fi (Deep Space Odyssey)": (
        "In a future where Earth's atmosphere is depleting, a crew of theoretical physicists "
        "and astronauts travels through an anomalous wormhole near Saturn. They race against "
        "relativistic time distortions to find a habitable exoplanet and ensure humanity's survival."
    ),
    "👻 Horror (The Cabin Curse)": (
        "Five university friends spend a weekend at an isolated woodland cabin. They unearth a "
        "cursed occult artifact in the cellar, unwittingly releasing an ancient demonic presence "
        "that stalks them one by one in absolute darkness."
    ),
    "😂 Comedy (Wedding Catastrophe)": (
        "Two irresponsible best friends are tasked with organizing an extravagant celebrity "
        "destination wedding. Their series of hilarious misunderstandings, runaway livestock, and "
        "botched speeches turn the prestigious ceremony into utter domestic chaos."
    ),
    "💥 Action (Rogue Operative)": (
        "A retired elite black-ops assassin is pulled back into the line of fire after an international "
        "mercenary cartel assassinates his mentor. Armed with tactical expertise, he wages a "
        "high-octane war across the streets of Berlin to bring them down."
    ),
    "❤️ Romance (Autumn in Paris)": (
        "An ambitious architectural illustrator and a passionate jazz pianist serendipitously "
        "meet in the rain-soaked avenues of Paris. As seasons shift, they navigate the delicate "
        "balance between personal dreams and their growing devotion to one another."
    ),
    "🔍 Mystery (The Locked Penthouse)": (
        "A celebrated forensic detective is summoned to a luxury skyscraper where a tech billionaire "
        "is discovered poisoned inside a vault locked from within. With ten suspects each concealing "
        "a motive, every clue reveals an intricate web of betrayal."
    ),
}


def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/clapperboard.png", width=90)
        st.title("Project Details")
        st.markdown(
            """
            <div class="card-box">
                <b>Topic:</b> Movie Genre Classification<br>
                <b>Classifier:</b> Multinomial Naive Bayes<br>
                <b>Features:</b> TF-IDF Vectorization<br>
                <b>Vocabulary:</b> Max 5,000 N-grams<br>
                <b>Task:</b> Multiclass NLP Prediction
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🛠️ Quick Instructions")
        st.markdown(
            """
            1. Enter or paste a movie plot summary in the text box.
            2. Or select a **Quick-Test Preset** to automatically fill an example plot.
            3. Click **Predict Genre** to execute the pipeline.
            """
        )

        st.markdown("---")
        st.markdown("### 📦 Artifact Status")
        model_exists = os.path.exists(MODEL_PATH)
        vec_exists = os.path.exists(VECTORIZER_PATH)
        if model_exists and vec_exists:
            st.success("✅ `model.pkl` & `vectorizer.pkl` Loaded")
        else:
            st.error("❌ Model artifacts missing! Run `python train.py` first.")

        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #888; text-align: center; margin-top: 30px;">
                End-to-End ML University Project<br>© 2026 Movie Genre NLP
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Main Page Header
    st.markdown('<h1 class="cinematic-header">CINEGENRE AI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-hero">Intelligent Movie Genre Classification using Natural Language Processing & Naive Bayes</p>',
        unsafe_allow_html=True,
    )

    # Badges
    st.markdown(
        """
        <div>
            <span class="badge-tag">TF-IDF Vectorizer</span>
            <span class="badge-tag">Multinomial Naive Bayes</span>
            <span class="badge-tag">Scikit-Learn</span>
            <span class="badge-tag">Streamlit UI</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    # Artifact Loading
    model, vectorizer = load_artifacts()
    if model is None or vectorizer is None:
        st.warning(
            "⚠️ **Model and Vectorizer files not found!**\n\n"
            "Please generate the model artifacts by executing the training script:\n"
            "```bash\n"
            "python train.py\n"
            "```\n"
            "Once `model.pkl` and `vectorizer.pkl` are generated, reload this page."
        )
        return

    # Sample Preset Selection
    st.markdown("### 💡 Quick-Test Example Plots")
    sample_choice = st.selectbox(
        "Choose an example plot to auto-fill the text area:",
        ["-- Select a Sample Plot --"] + list(SAMPLE_PLOTS.keys()),
        index=0,
    )

    initial_text = ""
    if sample_choice != "-- Select a Sample Plot --":
        initial_text = SAMPLE_PLOTS[sample_choice]

    # Input Section
    st.markdown("### 📝 Movie Plot Summary")
    user_input = st.text_area(
        label="Paste or type the movie synopsis below:",
        value=initial_text,
        height=180,
        placeholder="Enter a descriptive synopsis of the movie (e.g., character goals, conflicts, setting, theme)...",
        help="The richer and more descriptive the plot summary, the higher the classification confidence.",
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        predict_btn = st.button("🎬 Predict Genre", type="primary", use_container_width=True)
    with col_btn2:
        word_count = len(user_input.split()) if user_input else 0
        char_count = len(user_input) if user_input else 0
        st.caption(f"📊 **Word count:** {word_count} words | **Character count:** {char_count} chars")

    # Prediction Logic
    if predict_btn:
        if not user_input.strip():
            st.error("⚠️ Please enter a movie plot summary before predicting.")
            return

        with st.spinner("Analyzing synopsis and computing TF-IDF feature vectors..."):
            # Clean and vectorize
            cleaned_text = clean_input_text(user_input)
            features = vectorizer.transform([cleaned_text])

            # Predict class and probability
            predicted_genre = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            classes = model.classes_

            # Top predictions
            top_indices = np.argsort(probabilities)[::-1][:5]
            top_genres = [(classes[i], probabilities[i]) for i in top_indices]
            predicted_prob = dict(top_genres).get(predicted_genre, 0.0)

            emoji = GENRE_EMOJIS.get(predicted_genre, "🎬")

        # Display Results
        st.write("")
        st.markdown("## 🎯 Prediction Results")

        st.success(
            f"### {emoji} Predicted Genre: **{predicted_genre.upper()}**\n\n"
            f"Confidence Score: **{predicted_prob * 100:.2f}%**"
        )

        st.info(
            f"ℹ️ Based on the lexical patterns in the plot summary, the classifier evaluated "
            f"the synopsis as most characteristic of the **{predicted_genre}** genre."
        )

        # Confidence Breakdown
        col_chart, col_salient = st.columns([1.2, 1])

        with col_chart:
            st.markdown("#### 📊 Top Genre Probabilities")
            prob_df = pd.DataFrame(
                {
                    "Genre": [f"{GENRE_EMOJIS.get(g, '🎥')} {g}" for g, _ in top_genres],
                    "Confidence (%)": [round(p * 100, 2) for _, p in top_genres],
                }
            )

            for genre_name, prob in top_genres:
                g_emoji = GENRE_EMOJIS.get(genre_name, "🎥")
                st.write(f"**{g_emoji} {genre_name}** ({prob * 100:.1f}%)")
                st.progress(float(prob))

        with col_salient:
            st.markdown("#### 🔑 Influential Words Detected")
            # Extract non-zero TF-IDF terms from input
            feature_names = vectorizer.get_feature_names_out()
            feature_array = features.toarray()[0]
            non_zero_indices = feature_array.nonzero()[0]
            
            if len(non_zero_indices) > 0:
                word_scores = [
                    (feature_names[i], feature_array[i]) for i in non_zero_indices
                ]
                word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)[:8]

                keywords_html = "".join(
                    [
                        f"<span class='badge-tag' style='font-size: 0.85rem; margin-bottom: 8px;'>"
                        f"#{w} ({s:.2f})</span> "
                        for w, s in word_scores
                    ]
                )
                st.markdown(
                    f"<div class='card-box'>{keywords_html}</div>",
                    unsafe_allow_html=True,
                )
                st.caption("Top matching terms from the model's TF-IDF vocabulary.")
            else:
                st.caption("No significant training vocabulary terms detected in the plot.")


if __name__ == "__main__":
    main()
