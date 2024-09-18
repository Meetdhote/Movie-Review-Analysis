import pickle
import streamlit as st

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    with open('sentiment_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        tfidf = pickle.load(vectorizer_file)
    return model, tfidf

# Load emotion classifier model
@st.cache_resource
def load_emotion_model():
    from transformers import pipeline
    return pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', return_all_scores=True)
