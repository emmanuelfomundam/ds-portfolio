import joblib
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import clean_text

LABEL_INV = {0: 'negative', 1: 'neutral', 2: 'positive'}


def load_pipeline(model_path='models/sentiment_model.pkl',
                  vec_path='models/tfidf_vectorizer.pkl'):
    model = joblib.load(model_path)
    vec   = joblib.load(vec_path)
    return model, vec


def predict_sentiment(text: str, model, vectorizer) -> dict:
    cleaned = clean_text(text)
    vec     = vectorizer.transform([cleaned])
    label   = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]
    return {
        'text':       text,
        'sentiment':  LABEL_INV[label],
        'confidence': round(float(proba.max()) * 100, 2)
    }
