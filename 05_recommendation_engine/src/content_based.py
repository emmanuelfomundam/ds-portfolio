import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_movies(movies_path: str) -> pd.DataFrame:
    df = pd.read_csv(movies_path, sep='::',
                     names=['movieId', 'title', 'genres'],
                     engine='python', encoding='latin-1')
    df['genres_clean'] = df['genres'].str.replace('|', ' ', regex=False)
    return df


def build_content_matrix(movies: pd.DataFrame):
    vec   = TfidfVectorizer(max_features=500)
    tfidf = vec.fit_transform(movies['genres_clean'])
    sim   = cosine_similarity(tfidf)
    sim_df = pd.DataFrame(sim, index=movies['movieId'], columns=movies['movieId'])
    print(f'Content similarity matrix: {sim_df.shape}')
    return sim_df


def content_based_recs(movie_id: int, movies: pd.DataFrame,
                        sim_df: pd.DataFrame, top_n=10) -> list:
    if movie_id not in sim_df.index:
        print(f'Movie {movie_id} not found.')
        return []
    similar = (
        sim_df[movie_id]
        .drop(index=movie_id)
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    return similar
