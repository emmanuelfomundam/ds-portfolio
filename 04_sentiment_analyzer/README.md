# 04 — Text Sentiment Analyzer

## Overview

A 3-class sentiment classifier (positive / neutral / negative) trained on Amazon product reviews, with a deployed Streamlit web app for real-time inference. Achieved **82% overall accuracy** and **0.90 F1** on the dominant positive class.

## Dataset

**Amazon Fine Food Reviews**
- 568,454 total reviews; 100,000 sampled for training
- Star rating mapped: 1–2 → negative, 3 → neutral, 4–5 → positive
- Source: [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

> Download `Reviews.csv` and place it in `data/raw/`.

## Project Structure

```
04_sentiment_analyzer/
├── data/
│   ├── raw/                        # Raw CSV (not tracked by git)
│   └── processed/                  # Cleaned text (not tracked by git)
├── notebooks/
│   └── sentiment_analysis.ipynb    # Full walkthrough notebook
├── src/
│   ├── preprocess.py               # load_and_sample(), clean_text(), preprocess()
│   ├── train.py                    # split_data(), build_tfidf(), train_baselines(),
│   │                               # plot_wordclouds(), save_pipeline()
│   └── predict.py                  # load_pipeline(), predict_sentiment()
├── app/
│   └── app.py                      # Streamlit web app
├── models/                         # Serialized model + vectorizer (not tracked by git)
├── outputs/
│   ├── confusion_matrix_baseline.png
│   └── wordclouds.png
├── requirements.txt
└── README.md
```

## Methodology

### Preprocessing
- Lowercased, stripped HTML tags, removed punctuation via regex
- Filtered NLTK English stopwords and tokens under 3 characters

### Vectorization
- TF-IDF: **30,000 features**, unigram + bigram range `(1, 2)`

### Models
| Model | Accuracy | Positive F1 |
|-------|----------|-------------|
| Naive Bayes | ~77% | 0.86 |
| **Logistic Regression** | **82%** | **0.90** |

### Class Distribution (100K sample)
| Sentiment | Count |
|-----------|-------|
| Positive | ~78,000 |
| Negative | ~14,000 |
| Neutral | ~8,000 |

## Streamlit App

The app loads the serialized `sentiment_model.pkl` and `tfidf_vectorizer.pkl`, accepts free-text review input, and returns:
- Predicted sentiment label (🟢 Positive / 🟡 Neutral / 🔴 Negative)
- Confidence score (%)

```bash
# Run the app
streamlit run app/app.py
```

## How to Run (Notebook)

```bash
pip install -r requirements.txt
jupyter notebook notebooks/sentiment_analysis.ipynb
```

## Requirements

```
pandas
numpy
scikit-learn
nltk
streamlit
matplotlib
seaborn
wordcloud
joblib
jupyter
```
