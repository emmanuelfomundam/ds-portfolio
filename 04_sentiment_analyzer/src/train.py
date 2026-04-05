import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from wordcloud import WordCloud


LABEL_MAP = {'negative': 0, 'neutral': 1, 'positive': 2}


def split_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    X = df['clean_text']
    y = df['sentiment'].map(LABEL_MAP)
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=random_state)


def build_tfidf(X_train, X_test, max_features=30000):
    vec  = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)
    return X_tr, X_te, vec


def train_baselines(X_tr, y_tr, X_te, y_te, output_dir='outputs'):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000,
                                                   class_weight='balanced',
                                                   random_state=42),
        'Naive Bayes': MultinomialNB()
    }
    trained = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        trained[name] = m
        print(f'\n-- {name} --')
        print(classification_report(y_te, y_pred,
              target_names=['negative', 'neutral', 'positive']))

    best  = trained['Logistic Regression']
    y_pred = best.predict(X_te)
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_te, y_pred,
        display_labels=['negative', 'neutral', 'positive'],
        ax=ax, cmap='Blues', colorbar=False
    )
    ax.set_title('Logistic Regression - Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix_baseline.png', dpi=150)
    plt.show()
    return trained


def plot_wordclouds(df: pd.DataFrame, output_dir='outputs'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = ['negative', 'neutral', 'positive']
    colors = ['Reds', 'Greys', 'Greens']
    for ax, label, cmap in zip(axes, labels, colors):
        text = ' '.join(df[df['sentiment'] == label]['clean_text'])
        wc   = WordCloud(width=600, height=400,
                         background_color='white', colormap=cmap).generate(text)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(label.capitalize(), fontweight='bold', fontsize=13)
    plt.suptitle('Most Common Words by Sentiment', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/wordclouds.png', dpi=150)
    plt.show()


def save_pipeline(model, vectorizer,
                  model_path='models/sentiment_model.pkl',
                  vec_path='models/tfidf_vectorizer.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f'Model saved to {model_path}')
    print(f'Vectorizer saved to {vec_path}')
