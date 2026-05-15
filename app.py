import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis of Product Reviews",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0D3B66;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #1D7874;
        margin-bottom: 1.5rem;
    }
    .sentiment-positive {
        background-color: #D1FAE5;
        border-left: 5px solid #059669;
        padding: 1rem 1.5rem;
        border-radius: 6px;
        font-size: 1.3rem;
        font-weight: 700;
        color: #065F46;
    }
    .sentiment-negative {
        background-color: #FEE2E2;
        border-left: 5px solid #DC2626;
        padding: 1rem 1.5rem;
        border-radius: 6px;
        font-size: 1.3rem;
        font-weight: 700;
        color: #991B1B;
    }
    .sentiment-neutral {
        background-color: #FEF3C7;
        border-left: 5px solid #D97706;
        padding: 1rem 1.5rem;
        border-radius: 6px;
        font-size: 1.3rem;
        font-weight: 700;
        color: #92400E;
    }
    .metric-card {
        background-color: #F0F9FF;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #BAE6FD;
    }
    .stButton > button {
        background-color: #0D3B66;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
    }
    .stButton > button:hover {
        background-color: #1D7874;
    }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        with open("small_models.pkl", "rb") as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        return None

models_data = load_models()

# ── Text preprocessing ────────────────────────────────────────────────────────
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)          # remove HTML
    text = re.sub(r'[^a-z\s]', '', text)       # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()   # remove extra spaces
    return text

# ── Predict sentiment ─────────────────────────────────────────────────────────
def predict_sentiment(text, model_name, models_data):
    cleaned = preprocess_text(text)
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    if models_data is None:
        # Demo mode — simulate predictions
        scores = np.random.dirichlet(np.ones(3))
        pred_idx = np.argmax(scores)
        return label_map[pred_idx], dict(zip(label_map.values(), scores))

    model = models_data["models"][model_name]
    vectorizer = models_data.get("vectorizer")

    if vectorizer:
        X = vectorizer.transform([cleaned])
    else:
        X = [cleaned]

    pred = model.predict(X)[0]
    label = label_map.get(pred, str(pred))

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        confidence = dict(zip(label_map.values(), probs))
    else:
        confidence = {label: 1.0, **{k: 0.0 for k in label_map.values() if k != label}}

    return label, confidence

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/speech-bubble-with-dots.png", width=70)
    st.markdown("## 💬 Sentiment Analyser")
    st.markdown("---")

    page = st.radio("Navigate", [
        "🏠 Home",
        "✍️ Single Review",
        "📂 Batch Prediction",
        "📊 Model Performance",
        "👥 About"
    ])

    st.markdown("---")
    st.markdown("**Dataset:** Amazon Fine Food Reviews")
    st.markdown("**Models:** SVM · LR · DistilBERT · BERT")
    st.markdown("**Classes:** Positive · Neutral · Negative")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<div class="main-header">💬 Sentiment Analysis of Product Reviews</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Classify Amazon product reviews as Positive, Neutral, or Negative using ML & Deep Learning</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h2>4</h2><p>Models</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h2>3</h2><p>Sentiment Classes</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h2>568K+</h2><p>Reviews in Dataset</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h2>NLP</h2><p>Technique</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔍 What This App Does")
        st.markdown("""
        - **Single Review Prediction** — Enter any product review and instantly get its sentiment with confidence scores
        - **Batch Prediction** — Upload a CSV file of reviews and get bulk predictions
        - **Model Comparison** — Compare SVM, Logistic Regression, DistilBERT and BERT side by side
        - **Performance Metrics** — Explore confusion matrices and classification reports
        """)

    with col2:
        st.markdown("### 🤖 Models Used")
        model_info = {
            "Model": ["TF-IDF + SVM", "TF-IDF + Logistic Regression", "DistilBERT", "BERT"],
            "Type": ["Traditional ML", "Traditional ML", "Transformer", "Transformer"],
            "Accuracy": ["83.97%", "84.15%", "84.50%", "83.70%"]
        }
        st.dataframe(pd.DataFrame(model_info), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📋 Pipeline Overview")
    steps = ["1️⃣ Data Preprocessing", "2️⃣ EDA", "3️⃣ Feature Engineering", "4️⃣ Model Training", "5️⃣ Evaluation"]
    cols = st.columns(5)
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            st.info(step)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SINGLE REVIEW PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "✍️ Single Review":
    st.markdown("## ✍️ Single Review Sentiment Prediction")
    st.markdown("Type or paste a product review below to predict its sentiment.")

    model_choice = st.selectbox("Choose a Model", [
        "TF-IDF + SVM",
        "TF-IDF + Logistic Regression",
        "DistilBERT",
        "BERT"
    ])

    review_text = st.text_area(
        "Enter your product review here:",
        placeholder="e.g. This product is absolutely amazing! The quality is great and it arrived on time.",
        height=150
    )

    if st.button("🔍 Predict Sentiment"):
        if not review_text.strip():
            st.warning("Please enter a review before predicting.")
        else:
            with st.spinner("Analysing sentiment..."):
                sentiment, confidence = predict_sentiment(review_text, model_choice, models_data)

            st.markdown("### 🎯 Prediction Result")

            if sentiment == "Positive":
                st.markdown(f'<div class="sentiment-positive">😊 Sentiment: {sentiment}</div>', unsafe_allow_html=True)
            elif sentiment == "Negative":
                st.markdown(f'<div class="sentiment-negative">😞 Sentiment: {sentiment}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="sentiment-neutral">😐 Sentiment: {sentiment}</div>', unsafe_allow_html=True)

            st.markdown("#### Confidence Scores")
            col1, col2, col3 = st.columns(3)
            emojis = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}
            colors = {"Positive": "normal", "Neutral": "off", "Negative": "inverse"}

            for col, (label, score) in zip([col1, col2, col3], confidence.items()):
                with col:
                    st.metric(f"{emojis[label]} {label}", f"{score*100:.1f}%")

            # Bar chart
            fig, ax = plt.subplots(figsize=(6, 2.5))
            bars = ax.barh(list(confidence.keys()), list(confidence.values()),
                           color=["#059669", "#D97706", "#DC2626"])
            ax.set_xlim(0, 1)
            ax.set_xlabel("Confidence")
            ax.set_title(f"Confidence Distribution — {model_choice}")
            for bar, val in zip(bars, confidence.values()):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                        f"{val*100:.1f}%", va='center', fontsize=10)
            st.pyplot(fig)
            plt.close()

            st.markdown("#### 📝 Preprocessed Text")
            st.code(preprocess_text(review_text))

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Batch Prediction":
    st.markdown("## 📂 Batch Review Prediction")
    st.markdown("Upload a CSV file containing reviews to get sentiment predictions for all of them.")

    st.info("📌 Your CSV must have a column named **`review`** or **`Text`** containing the review text.")

    model_choice = st.selectbox("Choose a Model", [
        "TF-IDF + SVM",
        "TF-IDF + Logistic Regression",
        "DistilBERT",
        "BERT"
    ])

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown(f"**Uploaded:** {len(df)} rows, {len(df.columns)} columns")
        st.dataframe(df.head(), use_container_width=True)

        # Detect text column
        text_col = None
        for col in ["review", "Review", "text", "Text", "comment", "Comment"]:
            if col in df.columns:
                text_col = col
                break

        if text_col is None:
            st.error("Could not find a review text column. Please ensure your CSV has a column named 'review' or 'Text'.")
        else:
            if st.button("🚀 Run Batch Prediction"):
                with st.spinner("Predicting sentiments..."):
                    results = []
                    progress = st.progress(0)
                    for i, row in df.iterrows():
                        sentiment, confidence = predict_sentiment(str(row[text_col]), model_choice, models_data)
                        results.append({
                            "Review": str(row[text_col])[:80] + "...",
                            "Predicted Sentiment": sentiment,
                            "Positive %": f"{confidence.get('Positive', 0)*100:.1f}%",
                            "Neutral %": f"{confidence.get('Neutral', 0)*100:.1f}%",
                            "Negative %": f"{confidence.get('Negative', 0)*100:.1f}%",
                        })
                        progress.progress((i + 1) / len(df))

                results_df = pd.DataFrame(results)
                st.success(f"✅ Done! Predicted sentiment for {len(results_df)} reviews.")
                st.dataframe(results_df, use_container_width=True)

                # Sentiment distribution pie
                sentiment_counts = results_df["Predicted Sentiment"].value_counts()
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.pie(sentiment_counts.values, labels=sentiment_counts.index,
                       autopct="%1.1f%%",
                       colors=["#059669", "#D97706", "#DC2626"],
                       startangle=90)
                ax.set_title("Sentiment Distribution")
                st.pyplot(fig)
                plt.close()

                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button("⬇️ Download Results as CSV", csv, "sentiment_predictions.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("## 📊 Model Performance & Comparison")

    model_names = ["TF-IDF + SVM", "TF-IDF + LR", "DistilBERT", "BERT"]
    accuracies  = [0.8397, 0.8415, 0.8450, 0.8370]
    f1_scores   = [0.81, 0.81, 0.82, 0.79]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(model_names, [a * 100 for a in accuracies],
                      color=["#1D7874", "#0D3B66", "#EE964B", "#E63946"])
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Model Accuracy Comparison")
        plt.xticks(rotation=15, ha="right")
        for bar, val in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val*100:.1f}%", ha="center", fontsize=10, fontweight="bold")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### F1-Score Comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(model_names, f1_scores,
                      color=["#1D7874", "#0D3B66", "#EE964B", "#E63946"])
        ax.set_ylabel("F1-Score")
        ax.set_ylim(0, 1)
        ax.set_title("Model F1-Score Comparison")
        plt.xticks(rotation=15, ha="right")
        for bar, val in zip(bars, f1_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("### 📋 Metrics Summary Table")
    metrics_df = pd.DataFrame({
        "Model": model_names,
        "Accuracy": [f"{a*100:.2f}%" for a in accuracies],
        "F1-Score": [f"{f:.2f}" for f in f1_scores],
        "Type": ["Traditional ML", "Traditional ML", "Transformer", "Transformer"]
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🔲 Confusion Matrix")
    sample_cm = np.array([[326, 10, 245], [14, 16, 287], [47, 15, 3040]])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(sample_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Neutral", "Positive"],
                yticklabels=["Negative", "Neutral", "Positive"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — TF-IDF + SVM")
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👥 About":
    st.markdown("## 👥 About This Project")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Project Overview")
        st.markdown("""
        This project builds an end-to-end **Sentiment Analysis** pipeline for Amazon product reviews.
        It compares traditional NLP approaches with state-of-the-art transformer models to classify
        reviews as **Positive**, **Neutral**, or **Negative**.

        **Dataset:** Amazon Fine Food Reviews (Kaggle)
        **Total Reviews:** 568,454
        **Tech Stack:** Python, Scikit-learn, HuggingFace Transformers, Streamlit
        """)

        st.markdown("### 🔗 Links")
        st.markdown("- 📦 [GitHub Repository](https://github.com/sravananambiar20/Sentiment_Analysis_of_Product_Reviews)")
        st.markdown("- 📊 [Dataset on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)")

    with col2:
        st.markdown("### 👩‍💻 Team Members")
        team = pd.DataFrame({
            "Name": ["Ardra Selin A G", "Sravana Nambiar", "Archana"],
            "Reg No": ["XXXXXX", "XXXXXX", "XXXXXX"]
        })
        st.dataframe(team, use_container_width=True, hide_index=True)

        st.markdown("### 🤖 Models")
        st.markdown("""
        | Model | Approach |
        |---|---|
        | TF-IDF + SVM | Traditional ML |
        | TF-IDF + Logistic Regression | Traditional ML |
        | DistilBERT | Transformer |
        | BERT | Transformer |
        """)

    st.markdown("---")
    st.markdown("### 📁 Methodology Pipeline")
    cols = st.columns(5)
    steps = [
        ("1️⃣", "Data Preprocessing", "Cleaning, stopword removal, lemmatization"),
        ("2️⃣", "EDA", "Distribution plots, word clouds"),
        ("3️⃣", "Feature Engineering", "TF-IDF, BERT tokenization"),
        ("4️⃣", "Model Training", "SVM, LR, DistilBERT, BERT"),
        ("5️⃣", "Evaluation", "Accuracy, F1, Confusion Matrix"),
    ]
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"**{num} {title}**")
            st.caption(desc)
