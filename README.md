# Movie-Review-Analysis

This project is a Streamlit-based application that performs sentiment analysis, emotion classification, and rating prediction for movie reviews. It uses a machine learning model for sentiment analysis and a pre-trained model from Hugging Face for emotion classification.

## Features
Sentiment Analysis: Classifies the sentiment of the review as Positive or Negative.
Emotion Classification: Identifies the predominant emotion in the review (e.g., Joy, Sadness, Anger, etc.).
Rating Prediction: Provides a rating between 1 and 5 based on the sentiment probability.
Review History: Stores and displays a history of all processed reviews.


## Technologies Used
Streamlit: For building the interactive web application.
NLTK: For text preprocessing (e.g., tokenization, stopword removal).
Transformers: For using the pre-trained emotion classification model.
Pickle: For loading the pre-trained sentiment analysis model and TF-IDF vectorizer.

## Requirements
Python 3.8 or higher
Streamlit
NLTK
Transformers
Scikit-learn
Pickle

You can install the required packages using pip:
pip install streamlit nltk transformers scikit-learn

## Setup
Download Pre-trained Models:
Ensure that you have the pre-trained models saved as sentiment_model.pkl and tfidf_vectorizer.pkl. These files are used to load the sentiment analysis model and TF-IDF vectorizer.

Pre-trained Emotion Model:
The application uses the Hugging Face model j-hartmann/emotion-english-distilroberta-base. Ensure you have internet access to download the model when running the app for the first time.

Download NLTK Data:
The application requires NLTK data for tokenization and stopwords. It is automatically downloaded when the script runs.

## Running the Application
To run the Streamlit application, navigate to the directory containing your app.py file and run the following command:
streamlit run app.py

## This will start the Streamlit server, and you can view the application in your web browser at http://localhost:8501.

## Usage
Enter a Review: Type your movie review into the text area provided.
Predict: Click the "Predict" button to analyze the review. The application will display:
The sentiment of the review (Positive or Negative).
The sentiment score (probability of the review being positive).
The predicted emotion based on the review.
The predicted rating (1-5) based on the sentiment score.
Review History: The history of all reviews analyzed during the current session will be displayed below the prediction results.

## Notes
Ensure that your environment has internet access to download the Hugging Face model the first time you run the application.
If the application doesn't work as expected, check the console output for any error messages.
Troubleshooting
Model Loading Error: If you encounter issues with loading the Hugging Face model, ensure that the model identifier is correct and you have internet access.
Missing Files: Ensure that sentiment_model.pkl and tfidf_vectorizer.pkl are located in the same directory as app.py.


## Dataset 
link -->  https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews/input
