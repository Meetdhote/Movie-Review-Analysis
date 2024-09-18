import streamlit as st
from model_loader import load_model_and_vectorizer, load_emotion_model
from prediction import predict_sentiment_and_extras

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Load the model and vectorizer
model, tfidf = load_model_and_vectorizer()

# Load the emotion classifier
emotion_classifier = load_emotion_model()

# Streamlit app
st.title('Movie Review Analysis')

user_input = st.text_area("Enter a movie review:")

if st.button('Predict'):
    if user_input:
        # Pass emotion_classifier to predict_sentiment_and_extras
        sentiment, score, emotion, rating = predict_sentiment_and_extras(model, tfidf, user_input, emotion_classifier)

        st.write(f'The sentiment of the review is: **{sentiment}**')
        st.write(f'Sentiment score: **{score:.2f}**')
        st.write(f'Predicted Emotion: **{emotion}**')
        st.write(f'Predicted Rating (1-5): **{rating}**')

        # Add to history
        st.session_state.history.append({
            'Review': user_input,
            'Sentiment': sentiment,
            'Score': score,
            'Emotion': emotion,
            'Rating': rating
        })
    else:
        st.write('Please enter a review to get a prediction.')

# Display review history
if st.session_state.history:
    st.subheader('Review History:')
    for i, record in enumerate(st.session_state.history):
        st.write(f"{i + 1}. Review: {record['Review']}")
        st.write(f"   Sentiment: {record['Sentiment']} (Score: {record['Score']:.2f})")
        st.write(f"   Emotion: {record['Emotion']}")
        st.write(f"   Rating: {record['Rating']} / 5")
