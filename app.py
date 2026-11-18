import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = stopwords.words("english")

# Load saved model
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection App")
st.write("Enter a news article and check whether it is **REAL** or **FAKE**")

user_input = st.text_area("Paste News Text Here")

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)

        if prediction[0] == 1:
            st.success("✅ This looks like REAL news")
        else:
            st.error("❌ This looks like FAKE news")

st.markdown("---")
st.markdown("Built with ❤️ using Machine Learning")
