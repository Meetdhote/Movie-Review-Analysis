from preprocessing import preprocess_text

# Predict sentiment, emotion, and rating
def predict_sentiment_and_extras(model, tfidf, user_input, emotion_classifier):
    cleaned_input = preprocess_text(user_input)
    input_vector = tfidf.transform([cleaned_input]).toarray()

    prediction = model.predict(input_vector)[0]
    prediction_proba = model.predict_proba(input_vector)[0]

    sentiment = 'Positive' if prediction == 1 else 'Negative'
    positive_prob = prediction_proba[1]

    # Pass the emotion_classifier to the predict_emotion function
    emotion = predict_emotion(user_input, emotion_classifier)
    rating = predict_rating(positive_prob)

    return sentiment, positive_prob, emotion, rating

# Predict rating based on sentiment score
def predict_rating(positive_prob):
    if positive_prob >= 0.8:
        return 5
    elif 0.6 <= positive_prob < 0.8:
        return 4
    elif 0.4 <= positive_prob < 0.6:
        return 3
    elif 0.2 <= positive_prob < 0.4:
        return 2
    else:
        return 1

# Predict emotion
def predict_emotion(user_input, emotion_classifier):
    emotion_scores = emotion_classifier(user_input)[0]
    predicted_emotion = max(emotion_scores, key=lambda x: x['score'])
    return predicted_emotion['label']
