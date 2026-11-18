
import streamlit as st
import joblib
import re
import nltk
import pandas as pd
from datetime import datetime
from nltk.corpus import stopwords

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Fake News Detector | AI Dashboard",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2. LOAD NLP RESOURCES
# --------------------------------------------------

@st.cache_resource
def load_stopwords():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    return set(stopwords.words("english"))


@st.cache_resource
def load_models():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


try:
    stop_words = load_stopwords()
    model, vectorizer = load_models()
except Exception as e:
    st.error("Unable to load the model or NLP resources.")
    st.code(str(e))
    st.stop()


# --------------------------------------------------
# 3. TEXT PREPROCESSING
# --------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# --------------------------------------------------
# 4. SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0


# --------------------------------------------------
# 5. CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #F4F7FB;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background-color: #102A43;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    color: #163A5F;
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    color: #64748B;
    font-size: 15px;
    margin-bottom: 25px;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 3px 12px rgba(0,0,0,0.04);
}

.metric-label {
    color: #64748B;
    font-size: 14px;
}

.metric-value {
    color: #163A5F;
    font-size: 30px;
    font-weight: 800;
    margin-top: 7px;
}

.section-title {
    color: #163A5F;
    font-size: 22px;
    font-weight: 700;
}

.result-real {
    background: #DCFCE7;
    border-left: 6px solid #16A34A;
    padding: 22px;
    border-radius: 12px;
    color: #166534;
}

.result-fake {
    background: #FEE2E2;
    border-left: 6px solid #DC2626;
    padding: 22px;
    border-radius: 12px;
    color: #991B1B;
}

.info-card {
    background: white;
    border: 1px solid #E2E8F0;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 12px;
}

.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 10px 22px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 6. PREDICTION FUNCTION
# --------------------------------------------------

def predict_news(news_text):

    cleaned = clean_text(news_text)

    if not cleaned.strip():
        raise ValueError("The text contains no usable words after cleaning.")

    vectorized = vectorizer.transform([cleaned])

    prediction = int(model.predict(vectorized)[0])

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectorized)[0]

        class_positions = list(model.classes_)
        predicted_position = class_positions.index(prediction)

        confidence = float(probabilities[predicted_position]) * 100

    return prediction, confidence


# --------------------------------------------------
# 7. SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:

    st.markdown("# 📰 NewsAI")
    st.caption("Machine Learning Based Detection")

    st.markdown("---")

    page = st.radio(
        "MAIN MENU",
        [
            "Dashboard",
            "Analyze News",
            "Prediction History",
            "Analytics",
            "About Project"
        ]
    )

    st.markdown("---")

    st.markdown("### 🤖 Model Information")
    st.caption("Algorithm: Logistic Regression")
    st.caption("Feature Extraction: TF-IDF")
    st.caption("Language: Python")

    st.markdown("---")
    st.caption("B.Tech IT Project")


# --------------------------------------------------
# 8. DASHBOARD PAGE
# --------------------------------------------------

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">Dashboard Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Welcome to your AI-powered Fake News Detection System.</div>',
        unsafe_allow_html=True
    )

    total = len(st.session_state.history)

    real_count = sum(
        1 for item in st.session_state.history
        if item["result"] == "REAL"
    )

    fake_count = sum(
        1 for item in st.session_state.history
        if item["result"] == "FAKE"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Analyses</div>
            <div class="metric-value">{total}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Real Predictions</div>
            <div class="metric-value" style="color:#16A34A">{real_count}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Fake Predictions</div>
            <div class="metric-value" style="color:#DC2626">{fake_count}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(
        '<div class="section-title">System Overview</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:
        st.markdown("""
        <div class="info-card">
            <h3 style="color:#163A5F;">🔍 Intelligent News Analysis</h3>
            <p style="color:#64748B;">
            Analyse news articles using Natural Language Processing
            and Machine Learning techniques.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="info-card">
            <h3 style="color:#163A5F;">⚡ Fast Predictions</h3>
            <p style="color:#64748B;">
            Get a model-generated classification of news content
            as likely real or likely fake.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Recent Predictions")

    if st.session_state.history:

        recent = pd.DataFrame(
            st.session_state.history[-5:][::-1]
        )

        st.dataframe(
            recent,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("No news articles have been analysed yet. Open Analyze News to begin.")


# --------------------------------------------------
# 9. ANALYZE NEWS PAGE
# --------------------------------------------------

elif page == "Analyze News":

    st.markdown(
        '<div class="main-title">News Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Enter a news article to analyse its content.</div>',
        unsafe_allow_html=True
    )

    st.markdown("### 📝 Enter News Article")

    user_input = st.text_area(
        "Paste your news article below:",
        height=230,
        placeholder="Paste the complete news article here..."
    )

    st.caption("For better results, provide a meaningful article with sufficient text.")

    if st.button("🔍 Analyse News", use_container_width=True):

        if not user_input.strip():

            st.warning("Please enter a news article first.")

        else:

            try:
                with st.spinner("Analysing news article..."):

                    prediction, confidence = predict_news(user_input)

                result = "REAL" if prediction == 1 else "FAKE"

                record = {
                    "date": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                    "result": result,
                    "confidence": (
                        round(confidence, 2)
                        if confidence is not None else None
                    ),
                    "article": user_input[:150]
                }

                st.session_state.history.append(record)

                st.markdown("---")
                st.markdown("### Analysis Result")

                if prediction == 1:

                    st.markdown("""
                    <div class="result-real">
                        <h2>✅ Likely REAL News</h2>
                        <p>The model classified this article as real.</p>
                    </div>
                    """, unsafe_allow_html=True)

                else:

                    st.markdown("""
                    <div class="result-fake">
                        <h2>❌ Likely FAKE News</h2>
                        <p>The model classified this article as fake.</p>
                    </div>
                    """, unsafe_allow_html=True)

                if confidence is not None:

                    st.markdown("### Model Confidence")

                    st.progress(
                        min(max(int(confidence), 0), 100)
                    )

                    st.metric(
                        "Prediction confidence",
                        f"{confidence:.2f}%"
                    )

                st.caption(
                    "Disclaimer: This is a machine-learning prediction, "
                    "not independent verification of the article's truth."
                )

            except Exception as e:

                st.error("An error occurred during prediction.")
                st.code(str(e))


# --------------------------------------------------
# 10. PREDICTION HISTORY PAGE
# --------------------------------------------------

elif page == "Prediction History":

    st.markdown(
        '<div class="main-title">Prediction History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Review previously analysed news articles in this session.</div>',
        unsafe_allow_html=True
    )

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history[::-1]
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download History as CSV",
            data=csv,
            file_name="news_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Prediction History"):

            st.session_state.history = []
            st.rerun()

    else:

        st.info("Your prediction history is currently empty.")


# --------------------------------------------------
# 11. ANALYTICS PAGE
# --------------------------------------------------

elif page == "Analytics":

    st.markdown(
        '<div class="main-title">Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Visual overview of your prediction activity.</div>',
        unsafe_allow_html=True
    )

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        counts = history_df["result"].value_counts()

        chart_data = pd.DataFrame({
            "Prediction": ["REAL", "FAKE"],
            "Count": [
                int(counts.get("REAL", 0)),
                int(counts.get("FAKE", 0))
            ]
        })

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Prediction Distribution")
            st.bar_chart(
                chart_data.set_index("Prediction")
            )

        with col2:
            st.markdown("### Prediction Summary")
            st.dataframe(
                chart_data,
                use_container_width=True,
                hide_index=True
            )

        st.metric(
            "Total analysed articles",
            len(history_df)
        )

    else:

        st.info(
            "No analytics available yet. Analyse some news articles first."
        )


# --------------------------------------------------
# 12. ABOUT PROJECT PAGE
# --------------------------------------------------

elif page == "About Project":

    st.markdown(
        '<div class="main-title">About the Project</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Fake News Detection System using Machine Learning.</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ### 🎯 Project Objective

    The objective of this project is to develop a machine-learning
    application that analyses news article text and classifies it
    as likely real or likely fake.

    ### ⚙️ Technologies Used

    | Technology | Purpose |
    |---|---|
    | Python | Programming language |
    | Pandas | Data processing |
    | NLTK | Text preprocessing |
    | TF-IDF | Text feature extraction |
    | Logistic Regression | Classification |
    | Streamlit | Web application |
    | Joblib | Model loading |

    ### 🧠 Working Methodology

    1. News article is entered by the user.
    2. Text preprocessing is performed.
    3. TF-IDF converts text into numerical features.
    4. Logistic Regression predicts the class.
    5. The application displays the prediction.

    ### ⚠️ Disclaimer

    This system is an academic machine-learning project.
    Its predictions are based on patterns learned from its
    training dataset. It does not independently verify news
    or guarantee factual accuracy.
    """)


# --------------------------------------------------
# 13. FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#64748B; padding:10px;">
        <b>NewsAI - Fake News Detection System</b><br>
        Built with Python, Machine Learning and Streamlit
        <br><br>
        B.Tech Information Technology Project
    </div>
    """,
    unsafe_allow_html=True
)
