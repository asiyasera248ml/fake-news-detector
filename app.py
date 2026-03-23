import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
fake = pd.read_csv("Fake.csv", encoding='latin1', on_bad_lines='skip', engine='python')
real = pd.read_csv("True.csv", encoding='latin1', on_bad_lines='skip', engine='python')

print(fake.columns)
print(real.columns)

# Add labels
fake["label"] = 0
real["label"] = 1

# Combine
data = pd.concat([fake, real])

# Split
X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert text
vectorizer = TfidfVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test input
def predict_news(text):
    vec = vectorizer.transform([text])
    result = model.predict(vec)
    return "Real News" if result[0] == 1 else "Fake News"

print(predict_news("Breaking news: something happened"))