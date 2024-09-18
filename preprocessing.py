import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text)
    cleaned_text = ' '.join([word.lower() for word in tokens if word.lower() not in stop_words and word not in string.punctuation])
    return cleaned_text
