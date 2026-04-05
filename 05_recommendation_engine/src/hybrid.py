import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate


def train_svd(df: pd.DataFrame):
    reader = Reader(rating_scale=(1, 5))
    data   = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    print('Running 5-fold cross-validation on SVD...')
    cross_validate(SVD(n_factors=50, random_state=42),
                   data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    trainset = data.build_full_trainset()
    model    = SVD(n_factors=50, random_state=42)
    model.fit(trainset)
    print('SVD model trained on full dataset.')
    return model, data


def svd_recommend(model, user_id: int, all_movie_ids: list,
                   seen_ids: list, top_n=10) -> list:
    unseen = [m for m in all_movie_ids if m not in seen_ids]
    preds  = [(m, model.predict(user_id, m).est) for m in unseen]
    preds.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in preds[:top_n]]


def hybrid_recommend(user_id: int, ratings_df: pd.DataFrame,
                      svd_model, content_sim_df: pd.DataFrame,
                      all_movie_ids: list, top_n=10,
                      svd_weight=0.7, content_weight=0.3) -> list:
    seen   = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    unseen = [m for m in all_movie_ids if m not in seen]

    svd_scores = {m: svd_model.predict(user_id, m).est for m in unseen}

    top_rated = (
        ratings_df[ratings_df['userId'] == user_id]
        .sort_values('rating', ascending=False)
        .head(10)['movieId'].tolist()
    )
    content_scores = {}
    for m in unseen:
        if m in content_sim_df.index:
            scores = [
                content_sim_df.loc[m, r]
                for r in top_rated if r in content_sim_df.columns
            ]
            content_scores[m] = np.mean(scores) if scores else 0
        else:
            content_scores[m] = 0

    def normalize(d):
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        return {k: (v - mn) / (mx - mn + 1e-9) for k, v in d.items()} if mx > mn else d

    svd_n     = normalize(svd_scores)
    content_n = normalize(content_scores)

    hybrid = {
        m: svd_weight * svd_n.get(m, 0) + content_weight * content_n.get(m, 0)
        for m in unseen
    }
    top = sorted(hybrid, key=hybrid.get, reverse=True)[:top_n]
    return top


def popularity_fallback(ratings_df: pd.DataFrame, top_n=10) -> list:
    popular = (
        ratings_df.groupby('movieId')['rating']
        .agg(['count', 'mean'])
        .query('count >= 50')
        .sort_values('mean', ascending=False)
        .head(top_n)
        .index.tolist()
    )
    return popular
