import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))


def load_and_sample(path: str, n=100000, random_state=42) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[['Score', 'Text']].dropna()
    df['sentiment'] = df['Score'].apply(
        lambda s: 'negative' if s <= 2 else ('neutral' if s == 3 else 'positive')
    )
    df = df.sample(n=min(n, len(df)), random_state=random_state).reset_index(drop=True)
    print(f'Loaded {len(df):,} reviews.')
    print(df['sentiment'].value_counts())
    return df


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return ' '.join(tokens)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['clean_text'] = df['Text'].apply(clean_text)
    df = df[df['clean_text'].str.strip() != ''].reset_index(drop=True)
    print(f'Preprocessing complete. Shape: {df.shape}')
    return df
