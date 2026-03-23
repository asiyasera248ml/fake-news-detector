import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.title("📰 Fake News Detector")

# Small but better dataset (realistic)
data = {
    "text": [
        "The government has launched a new policy to improve education in rural areas",
        "Scientists discovered a new planet similar to Earth in a distant galaxy",
        "The stock market saw a significant rise due to strong earnings reports",
        "Drinking hot water every hour cures all diseases instantly",
        "Aliens have landed in New York and met government officials secretly",
        "A man claims he can become invisible by using a special chemical formula"
    ],
    "label": [1, 1, 1, 0, 0, 0]
}

df = pd.DataFrame(data)

# Train model
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

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
