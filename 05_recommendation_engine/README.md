# 05 — Recommendation Engine

## Overview

A four-mode hybrid recommendation system built on MovieLens data. Implements user-based CF, item-based CF, TF-IDF content-based filtering, and SVD matrix factorization — fused into a weighted hybrid recommender with a cold-start popularity fallback.

**SVD 5-fold CV: RMSE 0.8783 | MAE 0.6647**

## Dataset

**MovieLens 1M**
- 32M ratings across 2,000 users and 17,338 movies
- User-item matrix sparsity: **99.08%**
- Source: [GroupLens](https://grouplens.org/datasets/movielens/)

> Download and place `ratings.dat`, `movies.dat`, and `users.dat` in `data/raw/`.

## Project Structure

```
05_recommendation_engine/
├── data/
│   ├── raw/                            # Raw .dat files (not tracked by git)
│   └── processed/                      # Matrices and similarity files (not tracked by git)
├── notebooks/
│   └── recommendation_engine.ipynb     # Full walkthrough notebook
├── src/
│   ├── collaborative.py                # load_ratings(), build_user_item_matrix(),
│   │                                   # user_based_cf(), item_based_cf()
│   ├── content_based.py                # load_movies(), build_content_matrix(),
│   │                                   # content_based_recs()
│   └── hybrid.py                       # train_svd(), svd_recommend(),
│                                       # hybrid_recommend(), popularity_fallback()
├── outputs/
│   └── plots/
├── requirements.txt
└── README.md
```

## Four Recommendation Modes

### 1. User-Based Collaborative Filtering
- Builds a user-item rating matrix and computes cosine similarity between users
- Recommends movies rated highly by the 20 most similar users that the target user hasn't seen

### 2. Item-Based Collaborative Filtering
- Transposes the matrix and computes cosine similarity between items
- Returns the top-N most similar movies to a given title

### 3. Content-Based Filtering
- Vectorizes movie genres with TF-IDF (500 features)
- Recommends movies with the highest genre-based cosine similarity to a seed movie

### 4. SVD Matrix Factorization (Surprise)
- 50 latent factors, trained on full dataset after cross-validation
- 5-fold CV results:

| Metric | Score |
|--------|-------|
| RMSE | **0.8783** |
| MAE | **0.6647** |

### Hybrid Recommender
Fuses SVD and content-based scores with min-max normalization:

```
hybrid_score = 0.7 × SVD_score + 0.3 × content_score
```

### Cold-Start Fallback
For new users with no rating history — surfaces movies with 50+ ratings sorted by mean score.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/recommendation_engine.ipynb
```

## Requirements

```
pandas
numpy
scikit-learn
scikit-surprise
matplotlib
seaborn
jupyter
```
