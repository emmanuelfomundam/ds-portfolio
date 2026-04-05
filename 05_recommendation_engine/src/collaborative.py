import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_ratings(ratings_path: str) -> pd.DataFrame:
    df = pd.read_csv(ratings_path, sep='::',
                     names=['userId', 'movieId', 'rating', 'timestamp'],
                     engine='python')
    print(f'Loaded {len(df):,} ratings | {df["userId"].nunique():,} users | {df["movieId"].nunique():,} movies')
    return df


def build_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    matrix = df.pivot_table(index='userId', columns='movieId', values='rating')
    sparsity = 1 - matrix.notna().sum().sum() / matrix.size
    print(f'Matrix shape: {matrix.shape} | Sparsity: {sparsity:.2%}')
    return matrix


def user_based_cf(matrix: pd.DataFrame, user_id: int, top_n=10) -> list:
    filled = matrix.fillna(0)
    sim = cosine_similarity(filled)
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
    if user_id not in sim_df.index:
        print(f'User {user_id} not found.')
        return []
    similar_users = (
        sim_df[user_id]
        .drop(index=user_id)
        .sort_values(ascending=False)
        .head(20)
        .index.tolist()
    )
    seen      = matrix.loc[user_id].dropna().index.tolist()
    candidate = matrix.loc[similar_users].mean(axis=0)
    candidate = candidate.drop(index=seen, errors='ignore')
    recs      = candidate.sort_values(ascending=False).head(top_n).index.tolist()
    return recs


def item_based_cf(matrix: pd.DataFrame, movie_id: int, top_n=10) -> list:
    filled = matrix.fillna(0).T
    sim = cosine_similarity(filled)
    sim_df = pd.DataFrame(sim, index=matrix.columns, columns=matrix.columns)
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
