import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


with open("models/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("models/best_model.pkl", "rb") as f:   # SUBJECT MODEL
    subject_model = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    subject_le = pickle.load(f)

with open("models/best_model_topic.pkl", "rb") as f:
    topic_model = pickle.load(f)

with open("models/label_encoder_topic.pkl", "rb") as f:
    topic_le = pickle.load(f)

with open("models/best_model_difficulty.pkl", "rb") as f:
    diff_model = pickle.load(f)

with open("models/label_encoder_difficulty.pkl", "rb") as f:
    diff_le = pickle.load(f)


nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)


st.set_page_config(page_title="Smart Study Assistant", page_icon="📚", layout="centered")

st.title("📚 Smart Study Assistant (AI Tutor)")
st.write("Enter your exam-style question to get subject, topic, difficulty and study tips.")

user_input = st.text_area("✍️ Enter your question here:")

if st.button("Analyze Question"):
    if user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        cleaned = clean_text(user_input)
        final = preprocess(cleaned)
        vec = tfidf.transform([final])

        sub_pred = subject_model.predict(vec)
        subject = subject_le.inverse_transform(sub_pred)[0]

        topic_pred = topic_model.predict(vec)
        topic = topic_le.inverse_transform(topic_pred)[0]

        diff_pred = diff_model.predict(vec)
        difficulty = diff_le.inverse_transform(diff_pred)[0]

        st.subheader("📊 Prediction Results")

        col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📘 Subject")
        st.success(subject)

    with col2:
        st.markdown("### 📚 Topic")
        st.info(topic)   

    with col3:
        st.markdown("### 🎯 Difficulty")
        st.warning(difficulty)


st.markdown("---")
st.caption("Built by Aaradhya Jain | Smart Study Assistant using NLP & Machine Learning")