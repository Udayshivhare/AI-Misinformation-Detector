import re
import joblib
from pypdf import PdfReader
from docx import Document


# Load trained model
model = joblib.load("models/fake_news_model.pkl")

# Load TF-IDF vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Extract text from PDF
def extract_pdf(file_path):
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + " "

    return text


# Extract text from DOCX
def extract_docx(file_path):
    document = Document(file_path)

    text = ""

    for paragraph in document.paragraphs:
        text += paragraph.text + " "

    return text


# Read document
def read_document(file_path):

    if file_path.endswith(".pdf"):
        return extract_pdf(file_path)

    elif file_path.endswith(".docx"):
        return extract_docx(file_path)

    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    else:
        print("Unsupported file format!")
        return ""


# Main program
print("======================================")
print("     AI DOCUMENT FAKE NEWS DETECTOR")
print("======================================")

file_path = input("\nEnter document path: ")

text = read_document(file_path)

if not text:
    print("No text could be extracted.")
    exit()


# Clean extracted text
cleaned_text = clean_text(text)


# Convert text into TF-IDF
text_tfidf = vectorizer.transform([cleaned_text])


# Prediction
prediction = model.predict(text_tfidf)[0]

probability = model.predict_proba(text_tfidf)[0]


# Display result
print("\n======================================")

if prediction == 0:

    confidence = probability[0] * 100

    print("🔴 RESULT: LIKELY FAKE NEWS")
    print(f"Confidence: {confidence:.2f}%")

else:

    confidence = probability[1] * 100

    print("🟢 RESULT: LIKELY REAL NEWS")
    print(f"Confidence: {confidence:.2f}%")

print("======================================")