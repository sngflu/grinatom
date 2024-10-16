import os
import re
import pickle
import joblib
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")

# Initialize models and tokenizers
model = None
tokenizer = None
pos_model = None
neg_model = None
vectorizer = None

# Dictionary for expanding contractions in text
contractions_dict = {
    "aren't": "are not",
    "can't": "can not",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "might've": "might have",
    "must've": "must have",
    "mustn't": "must not",
    "should've": "should have",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you've": "you have",
}

# Function to expand contractions in a text
def expand_contractions(text, contractions_dict):
    for contraction, expanded in contractions_dict.items():
        text = re.sub(r"\b{}\b".format(contraction), expanded, text)
    return text

# Function to remove HTML tags from a text
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, ' ', text)

# Function to clean the input text
def clean_text(text):
    # Remove HTML tags
    text = remove_html_tags(text)
    
    # Convert text to lowercase
    text = text.lower()
    
    # Expand contractions
    text = expand_contractions(text, contractions_dict)
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Tokenize and lemmatize using spaCy
    doc = nlp(text)
    
    # Remove stop words and punctuation, and keep words longer than 1 character
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    
    # Keep tokens with more than 1 character or digits
    tokens = [token for token in tokens if len(token) > 1 or token.isdigit()]
    
    # Join the tokens back into a cleaned text
    clean_text = ' '.join(tokens)
    
    # Remove extra whitespace and strip leading/trailing spaces
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text

# Function to load the machine learning models and tokenizer
def load_model_and_tokenizer():
    global model, tokenizer, pos_model, neg_model, vectorizer
    
    # Load the LSTM model
    if model is None:
        MODEL_PATH = os.path.join(settings.BASE_DIR, 'mainapp', 'artifacts', 'LSTM_model.h5')
        model = load_model(MODEL_PATH)
    
    # Load the tokenizer
    if tokenizer is None:
        TOKENIZER_PATH = os.path.join(settings.BASE_DIR, 'mainapp', 'artifacts', 'tokenizer.pkl')
        if os.path.exists(TOKENIZER_PATH):
            with open(TOKENIZER_PATH, 'rb') as handle:
                tokenizer = pickle.load(handle)
        else:
            raise FileNotFoundError("Токенизатор не найден. Пожалуйста, сохраните токенизатор после обучения.")
    
    # Load the positive sentiment model
    if pos_model is None:
        POS_MODEL_PATH = os.path.join(settings.BASE_DIR, 'mainapp', 'artifacts', 'LR4_pos.joblib')
        if os.path.exists(POS_MODEL_PATH):
            pos_model = joblib.load(POS_MODEL_PATH)
        else:
            raise FileNotFoundError("Позитивная модель не найдена. Убедитесь, что файл 'LR4_pos.joblib' существует в папке 'artifacts'.")
    
    # Load the negative sentiment model
    if neg_model is None:
        NEG_MODEL_PATH = os.path.join(settings.BASE_DIR, 'mainapp', 'artifacts', 'LR4_neg.joblib')
        if os.path.exists(NEG_MODEL_PATH):
            neg_model = joblib.load(NEG_MODEL_PATH)
        else:
            raise FileNotFoundError("Негативная модель не найдена. Убедитесь, что файл 'LR4_neg.joblib' существует в папке 'artifacts'.")
    
    # Load the vectorizer
    if vectorizer is None:
        VECTORIZER_PATH = os.path.join(settings.BASE_DIR, 'mainapp', 'artifacts', 'tfidf_vectorizer.pkl')
        if os.path.exists(VECTORIZER_PATH):
            with open(VECTORIZER_PATH, 'rb') as handle:
                vectorizer = pickle.load(handle)
        else:
            raise FileNotFoundError("Векторизатор не найден. Пожалуйста, сохраните векторизатор после обучения.")

# Function to predict sentiment (1 for positive, 0 for negative)
def predict_sentiment(text):
    if model is None or tokenizer is None or pos_model is None or neg_model is None or vectorizer is None:
        load_model_and_tokenizer()
    
    # Clean the input text
    cleaned_text = clean_text(text)
    
    # Convert text to sequences using the tokenizer
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    
    # Pad the sequences to the same length
    max_length = 128
    padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    
    # Predict the sentiment using the LSTM model
    prediction = model.predict(padded)
    
    # Round the prediction to 0 or 1
    sentiment = int(round(prediction[0][0]))
    
    return sentiment

# Function to predict a score based on sentiment and the text
def predict_score(text, sentiment):
    if model is None or tokenizer is None or pos_model is None or neg_model is None or vectorizer is None:
        load_model_and_tokenizer()
    
    # Clean the input text
    cleaned_text = clean_text(text)
    
    # For positive sentiment (1), use the positive model
    if sentiment == 1:
        features = vectorizer.transform([cleaned_text])
        prediction = pos_model.predict(features)
        if hasattr(pos_model, "predict_proba"):
            score_class = pos_model.predict(features)[0]
        else:
            score_class = prediction[0]
        score = score_class
    # For negative sentiment (0), use the negative model
    else:
        features = vectorizer.transform([cleaned_text])
        prediction = neg_model.predict(features)
        if hasattr(neg_model, "predict_proba"):
            score_class = neg_model.predict(features)[0]
        else:
            score_class = prediction[0]
        score = score_class
    
    # Round the score and clamp it to the appropriate range
    if sentiment == 1:
        score = int(round(score))
        score = min(max(score, 7), 10)
    else:
        score = int(round(score))
        score = min(max(score, 1), 4)
    
    return score
