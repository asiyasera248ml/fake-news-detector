import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.title("📰 Fake News Detector")

# Load dataset from internet
@st.cache_data
def load_data():
    fake_url = "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/all_train.txt"
real_url = "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/all_val.txt"

    fake = pd.read_csv(fake_url)
    real = pd.read_csv(real_url)

    fake["label"] = 0
    real["label"] = 1

    data = pd.concat([fake, real])
    data.columns = data.columns.str.strip()

    return data

data = load_data()

# Train model
X = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

# UI
user_input = st.text_area("Enter news text:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter text")
    else:
        vec = vectorizer.transform([user_input])
        result = model.predict(vec)

        if result[0] == 1:
            st.success("✅ Real News")
        else:
            st.error("❌ Fake News")
