"""Train and save Naive Bayes and Logistic Regression models on the processed dataset."""
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset_processed.csv")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")


def train_and_save():
    df = pd.read_csv(DATASET_PATH).dropna(subset=["Kalimat Dinormalisasi", "Level Kata Kasar"])
    X = df["Kalimat Dinormalisasi"].astype(str)
    y = df["Level Kata Kasar"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # TF-IDF for LR
    lr_tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train_lr = lr_tfidf.fit_transform(X_train)
    X_test_lr = lr_tfidf.transform(X_test)

    lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
    lr.fit(X_train_lr, y_train)
    print("\n=== Logistic Regression ===")
    print(classification_report(y_test, lr.predict(X_test_lr)))

    # TF-IDF for NB (separate to allow different settings)
    nb_tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train_nb = nb_tfidf.fit_transform(X_train)
    X_test_nb = nb_tfidf.transform(X_test)

    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train_nb, y_train)
    print("\n=== Naive Bayes ===")
    print(classification_report(y_test, nb.predict(X_test_nb)))

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "lr_model.pkl"), "wb") as f:
        pickle.dump(lr, f)
    with open(os.path.join(MODELS_DIR, "lr_tfidf.pkl"), "wb") as f:
        pickle.dump(lr_tfidf, f)
    with open(os.path.join(MODELS_DIR, "nb_model.pkl"), "wb") as f:
        pickle.dump(nb, f)
    with open(os.path.join(MODELS_DIR, "nb_tfidf.pkl"), "wb") as f:
        pickle.dump(nb_tfidf, f)

    print("\nModels saved to", MODELS_DIR)


if __name__ == "__main__":
    train_and_save()
