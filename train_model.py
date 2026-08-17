import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load datasets
fake_news = pd.read_csv("dataset/Fake.csv")
real_news = pd.read_csv("dataset/True.csv")


# Add labels
fake_news["label"] = 0
real_news["label"] = 1


# Combine datasets
data = pd.concat([fake_news, real_news], ignore_index=True)


# Keep required columns
data = data[["title", "text", "label"]]


# Combine title and article text
data["content"] = data["title"] + " " + data["text"]


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Clean the content
data["content"] = data["content"].apply(clean_text)


# Input and output
X = data["content"]
y = data["label"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=50000,
    stop_words="english"
)


X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)

print("Training model...")

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# Make predictions
y_pred = model.predict(X_test_tfidf)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)


# Detailed evaluation
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Fake News", "Real News"]
))


# Save model and vectorizer
joblib.dump(model, "models/fake_news_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")


print("\nModel saved successfully!")
print("Vectorizer saved successfully!")