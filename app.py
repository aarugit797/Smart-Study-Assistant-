import streamlit as st
import pickle
import re
import nltk
import os
from typing import TypedDict
from nltk.corpus import stopwords
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


@st.cache_resource
def load_all_assets():
    with open("models/tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("models/best_model.pkl", "rb") as f:
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
    return tfidf, subject_model, subject_le, topic_model, topic_le, diff_model, diff_le



tfidf, subject_model, subject_le, topic_model, topic_le, diff_model, diff_le = load_all_assets()


nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess(text):
    tokens = text.split()
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)




class AgentState(TypedDict):
    question: str
    cleaned_text: str
    subject: str
    topic: str
    difficulty: str
    answer: str

def ml_classification_node(state: AgentState):
    """Your existing ML logic as a Graph Node"""
    cleaned = clean_text(state['question'])
    processed = preprocess(cleaned)
    vec = tfidf.transform([processed])


    sub_pred = subject_model.predict(vec)
    subject = subject_le.inverse_transform(sub_pred)[0]

    topic_pred = topic_model.predict(vec)
    topic = topic_le.inverse_transform(topic_pred)[0]

    diff_pred = diff_model.predict(vec)
    difficulty = diff_le.inverse_transform(diff_pred)[0]

    return {
        "subject": subject, 
        "topic": topic, 
        "difficulty": difficulty, 
        "cleaned_text": processed
    }

def llm_answer_node(state: AgentState):
    """Generates the final answer using LangChain"""
   # Use an environment variable instead of hardcoding
    os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "PASTE_KEY_HERE_FOR_LOCAL_ONLY")
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    
    prompt = ChatPromptTemplate.from_template(
        """You are a Smart Study Assistant. 
        A student has asked a question classified as:
        Subject: {subject}
        Topic: {topic}
        Difficulty: {difficulty}

        Question: {question}

        Provide a detailed, helpful academic explanation. 
        Adjust your tone to match the {difficulty} level (if hard, be more thorough).
        """
    )
    
    chain = prompt | llm
    response = chain.invoke({
        "subject": state['subject'],
        "topic": state['topic'],
        "difficulty": state['difficulty'],
        "question": state['question']
    })
    
    return {"answer": response.content}


workflow = StateGraph(AgentState)
workflow.add_node("classify", ml_classification_node)
workflow.add_node("answer", llm_answer_node)

workflow.set_entry_point("classify")
workflow.add_edge("classify", "answer")
workflow.add_edge("answer", END)

app_graph = workflow.compile()



st.set_page_config(page_title="Smart Study Assistant", page_icon="📚", layout="centered")
st.title("📚 Smart Study Assistant (AI Tutor)")
st.write("Enter your exam-style question to get analysis and an AI-generated answer.")

user_input = st.text_area("✍️ Enter your question here:")

if st.button("Analyze & Answer"):
    if user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing through LangGraph..."):
         
            result = app_graph.invoke({"question": user_input})

            st.subheader("📊 Prediction Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("### 📘 Subject")
                st.success(result['subject'])
            with col2:
                st.markdown("### 📚 Topic")
                st.info(result['topic'])
            with col3:
                st.markdown("### 🎯 Difficulty")
                st.warning(result['difficulty'])

            st.markdown("---")

            st.subheader("🤖 AI Tutor Explanation")
            st.write(result['answer'])

st.markdown("---")
st.caption("Built by Aaradhya Jain | Smart Study Assistant using NLP, Machine Learning & LangGraph")