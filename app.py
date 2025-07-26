import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Download NLTK data
nltk.download('stopwords')

# Set page configuration for better UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Custom CSS for Tailwind-like styling
st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css');
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f2937;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.25rem;
        color: #4b5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .input-box {
        width: 100%;
        padding: 0.75rem;
        border-radius: 0.375rem;
        border: 1px solid #d1d5db;
        margin-bottom: 1rem;
        font-size: 1rem;
    }
    .predict-button {
        background-color: #3b82f6;
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 0.375rem;
        font-weight: bold;
        width: 100%;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .predict-button:hover {
        background-color: #2563eb;
    }
    .result-box {
        margin-top: 2rem;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .fake {
        background-color: #fee2e2;
        color: #b91c1c;
    }
    .real {
        background-color: #d1fae5;
        color: #065f46;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_fake_news_model():
    return load_model('model.h5')

model = load_fake_news_model()

# Preprocessing function
def preprocess_text(text):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    return review

# Tokenization and padding
vocab_size = 5000
sent_length = 20

def prepare_input(text):
    corpus = preprocess_text(text)
    one_hot_repr = [one_hot(corpus, vocab_size)]
    padded = pad_sequences(one_hot_repr, padding='pre', maxlen=sent_length)
    return np.array(padded)

# Streamlit UI
st.markdown('<div class="title">Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter a news title to check if it\'s real or fake!</div>', unsafe_allow_html=True)

# Input form
with st.form(key='news_form'):
    news_title = st.text_input("News Title", placeholder="Enter the news title here...", key="news_input")
    submit_button = st.form_submit_button(label="Predict", help="Click to classify the news")

# Prediction logic
if submit_button and news_title:
    try:
        # Preprocess and predict
        processed_input = prepare_input(news_title)
        prediction = model.predict(processed_input)[0][0]
        confidence = prediction if prediction > 0.5 else 1 - prediction
        label = "Fake" if prediction > 0.5 else "Real"
        color_class = "fake" if label == "Fake" else "real"

        # Display result
        st.markdown(f"""
            <div class="result-box {color_class}">
                <h3>Prediction: {label}</h3>
                <p>Confidence: {confidence:.2%}</p>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error processing input: {str(e)}")
elif submit_button and not news_title:
    st.warning("Please enter a news title to classify.")

# Instructions for sharing
st.markdown("""
    <div style='margin-top: 2rem; text-align: center; color: #4b5563;'>
        <p>Share this app by deploying it on Streamlit Cloud or a similar platform.</p>
        <p>Simply share the generated URL with others to let them access this Fake News Detector!</p>
    </div>
""", unsafe_allow_html=True)
