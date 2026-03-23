import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# Title
st.title("📰 Fake News Detector")
st.write("Check whether a news is Real or Fake using AI")

# Load data
@st.cache_data
def load_data():
    fake = pd.read_csv("Fake.csv", encoding='latin1', on_bad_lines='skip', engine='python')
    real = pd.read_csv("True.csv", encoding='latin1', on_bad_lines='skip', engine='python')

    fake["label"] = 0
    real["label"] = 1

    data = pd.concat([fake, real])
    data.columns = data.columns.str.strip()

    return data

data = load_data()

# Train model
@st.cache_resource
def train_model(data):
    X = data["text"]
    y = data["label"]

    vectorizer = TfidfVectorizer(stop_words="english")
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vec, y)

    return model, vectorizer

model, vectorizer = train_model(data)

# Input box
user_input = st.text_area("✍️ Enter News Text:", height=150)

# Button
if st.button("🔍 Check News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        vec = vectorizer.transform([user_input])
        result = model.predict(vec)

        if result[0] == 1:
            st.success("✅ This is REAL news")
        else:
            st.error("❌ This is FAKE news")

# Footer
st.markdown("---")
st.caption("Built using Python, Machine Learning & Streamlit")