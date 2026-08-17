import re
import joblib


# Load trained model
model = joblib.load("models/fake_news_model.pkl")

# Load TF-IDF vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Get article from user
print("====================================")
print("     AI FAKE NEWS DETECTOR")
print("====================================")

article = input("\nEnter your news article:\n")


# Clean article
cleaned_article = clean_text(article)


# Convert article to TF-IDF
article_tfidf = vectorizer.transform([cleaned_article])


# Make prediction
prediction = model.predict(article_tfidf)[0]

# Get probability
probability = model.predict_proba(article_tfidf)[0]


# Display result
print("\n====================================")

if prediction == 0:
    confidence = probability[0] * 100
    print("🔴 RESULT: LIKELY FAKE NEWS")
    print(f"Confidence: {confidence:.2f}%")

else:
    confidence = probability[1] * 100
    print("🟢 RESULT: LIKELY REAL NEWS")
    print(f"Confidence: {confidence:.2f}%")

print("====================================")