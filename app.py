import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ========== Load Models ==========

with open("models/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ========== NLP Setup ==========

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

# ========== UI ==========

st.set_page_config(page_title="Smart Study Assistant", page_icon="📚")

st.title("📚 Smart Study Assistant (Subject Predictor)")
st.write("Enter your exam-style question and I will predict the subject.")

user_input = st.text_area("Enter your question:")

if st.button("Predict Subject"):
    if user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        cleaned = clean_text(user_input)
        final = preprocess(cleaned)
        vec = tfidf.transform([final])
        pred = model.predict(vec)
        subject = le.inverse_transform(pred)[0]

        st.success(f"📘 Predicted Subject: **{subject}**")
