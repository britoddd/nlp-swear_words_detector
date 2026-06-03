"""
Train Logistic Regression, Naive Bayes, and SVM classifiers.

Usage:
    python train_classical.py

Reads:  dataset_processed.csv  (produced by preprocess_pipeline.py)
Writes: saved_models/lr_model.pkl, lr_tfidf.pkl, nb_model.pkl,
        nb_tfidf.pkl, svm_model.pkl
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, classification_report
from sklearn.utils import resample

from config import (
    OUTPUT_PATH, MODELS_DIR, FORCE_RETRAIN,
    LR_MODEL_PATH, LR_TFIDF_PATH, NB_MODEL_PATH, NB_TFIDF_PATH, SVM_MODEL_PATH,
    TEST_SIZE, RANDOM_STATE,
)


def random_oversample(X_ser: pd.Series, y_ser: pd.Series,
                      random_state: int = 42) -> tuple:
    max_count = y_ser.value_counts().max()
    parts_X, parts_y = [], []
    for label in sorted(y_ser.unique()):
        mask = (y_ser == label).values
        Xs = X_ser[mask].reset_index(drop=True)
        ys = y_ser[mask].reset_index(drop=True)
        if len(ys) < max_count:
            idx = resample(range(len(ys)), n_samples=max_count,
                           random_state=random_state, replace=True)
            Xs, ys = Xs.iloc[idx], ys.iloc[idx]
        parts_X.append(Xs)
        parts_y.append(ys)
    return (pd.concat(parts_X).reset_index(drop=True),
            pd.concat(parts_y).reset_index(drop=True))


def load_splits():
    df = pd.read_csv(OUTPUT_PATH)
    df = df.dropna(subset=['Kalimat Dinormalisasi'])
    df = df[df['Kalimat Dinormalisasi'].str.strip() != ''].reset_index(drop=True)

    X = df['Kalimat Dinormalisasi']
    y = df['Level Kata Kasar']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, y_train = random_oversample(X_train, y_train, random_state=RANDOM_STATE)
    print(f'Train: {len(y_train):,} (after oversampling) | Test: {len(y_test):,}')
    return X_train, X_test, y_train, y_test


def build_tfidf_features(X_train, X_test):
    tfidf_lr = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5),
                               max_features=50_000, min_df=2, sublinear_tf=True)
    tfidf_nb = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),
                               max_features=50_000, min_df=2, sublinear_tf=True)

    X_train_lr = tfidf_lr.fit_transform(X_train)
    X_test_lr  = tfidf_lr.transform(X_test)
    X_train_nb = tfidf_nb.fit_transform(X_train)
    X_test_nb  = tfidf_nb.transform(X_test)

    return tfidf_lr, tfidf_nb, X_train_lr, X_test_lr, X_train_nb, X_test_nb


def train_lr(X_train, y_train, X_test, y_test):
    if not FORCE_RETRAIN and os.path.exists(LR_MODEL_PATH):
        print('LR: loading from disk')
        with open(LR_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    else:
        print('LR: training...')
        model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                                   random_state=RANDOM_STATE, class_weight='balanced')
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f'LR  F1 (macro): {f1:.4f}')
    print(classification_report(y_test, y_pred,
          target_names=['Level 0', 'Level 1', 'Level 2', 'Level 3']))
    return model, y_pred


def train_nb(X_train, y_train, X_test, y_test):
    if not FORCE_RETRAIN and os.path.exists(NB_MODEL_PATH):
        print('NB: loading from disk')
        with open(NB_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    else:
        print('NB: training...')
        model = ComplementNB(alpha=0.1)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f'NB  F1 (macro): {f1:.4f}')
    print(classification_report(y_test, y_pred,
          target_names=['Level 0', 'Level 1', 'Level 2', 'Level 3']))
    return model, y_pred


def train_svm(X_train, y_train, X_test, y_test):
    if not FORCE_RETRAIN and os.path.exists(SVM_MODEL_PATH):
        print('SVM: loading from disk')
        with open(SVM_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    else:
        print('SVM: training...')
        svm_base = LinearSVC(C=1.0, max_iter=2000,
                             class_weight='balanced', random_state=RANDOM_STATE)
        model = CalibratedClassifierCV(svm_base, cv=5)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f'SVM F1 (macro): {f1:.4f}')
    print(classification_report(y_test, y_pred,
          target_names=['Level 0', 'Level 1', 'Level 2', 'Level 3']))
    return model, y_pred


def save_models(lr_model, tfidf_lr, nb_model, tfidf_nb, svm_model):
    os.makedirs(MODELS_DIR, exist_ok=True)
    for path, obj in [
        (LR_MODEL_PATH,  lr_model),
        (LR_TFIDF_PATH,  tfidf_lr),
        (NB_MODEL_PATH,  nb_model),
        (NB_TFIDF_PATH,  tfidf_nb),
        (SVM_MODEL_PATH, svm_model),
    ]:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    print(f'Models saved to {MODELS_DIR}/')


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_splits()
    tfidf_lr, tfidf_nb, X_train_lr, X_test_lr, X_train_nb, X_test_nb = \
        build_tfidf_features(X_train, X_test)

    lr_model,  y_pred_lr  = train_lr(X_train_lr,  y_train, X_test_lr,  y_test)
    nb_model,  y_pred_nb  = train_nb(X_train_nb,  y_train, X_test_nb,  y_test)
    svm_model, y_pred_svm = train_svm(X_train_lr, y_train, X_test_lr,  y_test)

    save_models(lr_model, tfidf_lr, nb_model, tfidf_nb, svm_model)
