import streamlit as st
import re
import joblib
import cv2
import numpy as np

from PIL import Image
from pypdf import PdfReader
from docx import Document


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Misinformation Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #0b0f19;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263244;
    }

    /* Main content */
    .main-title {
        font-size: 46px;
        font-weight: 800;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 17px;
        margin-bottom: 30px;
    }

    /* Hero */
    .hero {
        background: linear-gradient(
            135deg,
            #111827,
            #172033
        );
        border: 1px solid #263244;
        border-radius: 18px;
        padding: 28px;
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 30px;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .hero-text {
        color: #9ca3af;
        font-size: 15px;
    }

    /* Cards */
    .info-card {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 16px;
        padding: 22px;
        min-height: 150px;
        margin-bottom: 15px;
    }

    .card-icon {
        font-size: 28px;
        margin-bottom: 8px;
    }

    .card-title {
        font-size: 19px;
        font-weight: 700;
    }

    .card-text {
        color: #9ca3af;
        font-size: 14px;
        margin-top: 5px;
    }

    /* Result cards */
    .result-card {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 18px;
        padding: 28px;
        margin-top: 25px;
    }

    .result-title {
        font-size: 25px;
        font-weight: 750;
    }

    .confidence-label {
        color: #9ca3af;
        font-size: 14px;
        margin-top: 15px;
    }

    /* Status */
    .status {
        display: inline-block;
        padding: 7px 13px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
        background: #162033;
        border: 1px solid #263244;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 45px;
        font-weight: 700;
        border: 1px solid #334155;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #263244;
        padding: 18px;
        border-radius: 14px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 12px;
        margin-top: 45px;
        padding: 20px;
        border-top: 1px solid #1f2937;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODELS
# ============================================================

news_model = joblib.load("models/fake_news_model.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
image_model = joblib.load("models/image_model.pkl")


# ============================================================
# HOG
# ============================================================

hog = cv2.HOGDescriptor(
    (32, 32),
    (16, 16),
    (8, 8),
    (8, 8),
    9
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# NEWS PREDICTION
# ============================================================
def predict_news(text):

    cleaned_text = clean_text(text)

    vector = tfidf_vectorizer.transform(
        [cleaned_text]
    )

    prediction = news_model.predict(vector)[0]

    probability = news_model.predict_proba(vector)[0]

    if prediction == 0:

        confidence = probability[0] * 100

        if confidence < 70:
            return "UNCERTAIN", confidence

        return "LIKELY MISLEADING", confidence

    else:

        confidence = probability[1] * 100

        if confidence < 70:
            return "UNCERTAIN", confidence

        return "LIKELY AUTHENTIC", confidence


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + " "

    return text


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_docx(uploaded_file):

    document = Document(uploaded_file)

    text = ""

    for paragraph in document.paragraphs:

        text += paragraph.text + " "

    return text


# ============================================================
# IMAGE PREDICTION
# ============================================================

def predict_image(uploaded_file):

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_array = np.array(image)

    image_array = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    image_array = cv2.resize(
        image_array,
        (32, 32)
    )

    features = hog.compute(
        image_array
    )

    features = features.reshape(
        1, -1
    )

    prediction = image_model.predict(
        features
    )[0]

    probability = image_model.predict_proba(
        features
    )[0]

    if prediction == 0:

        confidence = probability[0] * 100

        if confidence < 70:
            return "UNCERTAIN", confidence

        return "LIKELY AUTHENTIC IMAGE", confidence

    else:

        confidence = probability[1] * 100

        if confidence < 70:
            return "UNCERTAIN", confidence

        return "LIKELY AI-GENERATED IMAGE", confidence


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_history" not in st.session_state:

    st.session_state.analysis_history = []


if "mode" not in st.session_state:

    st.session_state.mode = "Article"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🛡️ AI Misinformation Detector"
    )

    st.caption(
        "Multimodal misinformation analysis"
    )

    st.divider()

    st.markdown("### Analysis")

    mode = st.radio(
        "Select content type",
        [
            "Article",
            "Document",
            "Image"
        ],
        index=[
            "Article",
            "Document",
            "Image"
        ].index(st.session_state.mode),
        label_visibility="collapsed"
    )

    st.session_state.mode = mode

    st.divider()

  # ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown("### 📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "News Accuracy",
        "99.03%"
    )

with col2:
    st.metric(
        "Image Accuracy",
        "69.68%"
    )

st.caption(
    "News: Precision 0.99 • Recall 0.99 • F1 0.99"
)

st.caption(
    "Image: Precision 0.69–0.70 • Recall 0.68–0.72 • F1 0.69–0.70"
)

st.divider()

st.caption(
    "Python • NLP • Machine Learning • Computer Vision"
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """<div class="hero">
<div class="hero-title">🛡️ AI Misinformation Detector</div>
<div class="hero-text">Multimodal AI system for analyzing news, documents and images for potentially misleading or AI-generated content.</div>
</div>""",
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

col1, col2 = st.columns(
    [5, 1]
)

with col2:

    st.markdown(
        '<div class="status">● SYSTEM READY</div>',
        unsafe_allow_html=True
    )


# ============================================================
# CONTENT TYPE CARDS
# ============================================================

st.markdown("### What would you like to analyze?")

card1, card2, card3 = st.columns(3)

with card1:

    st.markdown(
        """
        <div class="info-card">
            <div class="card-icon">📰</div>
            <div class="card-title">News & Article</div>
            <div class="card-text">
                Analyze written news content using NLP
                and machine learning.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with card2:

    st.markdown(
        """
        <div class="info-card">
            <div class="card-icon">📄</div>
            <div class="card-title">Documents</div>
            <div class="card-text">
                Analyze PDF, DOCX and TXT documents
                after extracting their text.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with card3:

    st.markdown(
        """
        <div class="info-card">
            <div class="card-icon">🖼️</div>
            <div class="card-title">Images</div>
            <div class="card-text">
                Analyze images using computer-vision
                features and a trained ML model.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()


# ============================================================
# ARTICLE
# ============================================================

if mode == "Article":

    st.markdown("## 📰 News Authenticity Analysis")

    article = st.text_area(
        "Paste article content",
        height=260,
        placeholder=(
            "Paste the complete news article here..."
        )
    )

    if st.button(
        "🔍 Analyze Article",
        type="primary"
    ):

        if not article.strip():

            st.error(
                "Please enter an article first."
            )

        else:

            result, confidence = predict_news(
                article
            )

            st.markdown(
                """
                <div class="result-card">
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="result-title">{result}</div>',
                unsafe_allow_html=True
            )

            st.progress(
                int(confidence)
            )

            st.caption(
                f"Model confidence: {confidence:.2f}%"
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            st.session_state.analysis_history.append(
                {
                    "Type": "Article",
                    "Result": result,
                    "Confidence": f"{confidence:.2f}%"
                }
            )


# ============================================================
# DOCUMENT
# ============================================================

elif mode == "Document":

    st.markdown("## 📄 Document Authenticity Analysis")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "txt"]
    )

    if uploaded_file:

        st.info(
            f"Selected: {uploaded_file.name}"
        )

        if st.button(
            "🔍 Analyze Document",
            type="primary"
        ):

            extension = (
                uploaded_file.name
                .split(".")[-1]
                .lower()
            )

            try:

                if extension == "pdf":

                    text = extract_pdf(
                        uploaded_file
                    )

                elif extension == "docx":

                    text = extract_docx(
                        uploaded_file
                    )

                else:

                    text = uploaded_file.read().decode(
                        "utf-8",
                        errors="ignore"
                    )

                if not text.strip():

                    st.error(
                        "No readable text was found."
                    )

                else:

                    result, confidence = predict_news(
                        text
                    )

                    if result == "LIKELY AUTHENTIC":

                        st.success(
                            f"🟢 {result}"
                        )

                    elif result == "UNCERTAIN":

                        st.warning(
                            f"🟡 {result}"
                        )

                    else:

                        st.error(
                            f"🔴 {result}"
                        )

                    st.progress(
                        int(confidence)
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )

                    st.session_state.analysis_history.append(
                        {
                            "Type": "Document",
                            "Result": result,
                            "Confidence": f"{confidence:.2f}%"
                        }
                    )

                    with st.expander(
                        "View extracted text"
                    ):

                        st.write(
                            text[:5000]
                        )

            except Exception as e:

                st.error(
                    f"Analysis failed: {e}"
                )


# ============================================================
# IMAGE
# ============================================================

elif mode == "Image":

    st.markdown("## 🖼️ Image Authenticity Analysis")

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        image = Image.open(
            uploaded_image
        )

        st.image(
            image,
            caption="Uploaded Image",
            width=500
        )

        if st.button(
            "🔍 Analyze Image",
            type="primary"
        ):

            try:

                result, confidence = predict_image(
                    uploaded_image
                )

                if result == "LIKELY AUTHENTIC IMAGE":

                    st.success(
                        f"🟢 {result}"
                    )

                elif result == "UNCERTAIN":

                    st.warning(
                        f"🟡 {result}"
                    )

                else:

                    st.error(
                        f"🔴 {result}"
                    )

                st.progress(
                    int(confidence)
                )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                st.session_state.analysis_history.append(
                    {
                        "Type": "Image",
                        "Result": result,
                        "Confidence": f"{confidence:.2f}%"
                    }
                )

            except Exception as e:

                st.error(
                    f"Image analysis failed: {e}"
                )
# ============================================================
# NOTE
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align: center;
        padding: 20px;
        color: #9ca3af;
        font-size: 13px;
    ">
        <strong>Note:</strong>
        Results are AI-generated predictions and should be
        independently verified before making decisions.
        <br><br>
        Python • NLP • Machine Learning • Computer Vision
    </div>
    """,
    unsafe_allow_html=True
)