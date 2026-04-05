import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
from predict import load_pipeline, predict_sentiment

st.set_page_config(page_title='Sentiment Analyzer', page_icon='💬')
st.title('💬 Text Sentiment Analyzer')
st.markdown('Enter a product review and get an instant sentiment prediction.')

model, vectorizer = load_pipeline(
    model_path=os.path.join(os.path.dirname(__file__), '..', 'models', 'sentiment_model.pkl'),
    vec_path=os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_vectorizer.pkl')
)

user_input = st.text_area('Enter review text:', height=150,
                           placeholder='e.g. This product was absolutely amazing...')

if st.button('Analyze'):
    if user_input.strip():
        result = predict_sentiment(user_input, model, vectorizer)
        emoji  = {'positive': '🟢', 'neutral': '🟡', 'negative': '🔴'}
        sentiment = result['sentiment']
        st.markdown(f'### {emoji[sentiment]} Sentiment: **{sentiment.upper()}**')
        st.markdown(f'**Confidence:** {result["confidence"]}%')
    else:
        st.warning('Please enter some text first.')
