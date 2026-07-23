"""
Strategy: Random oversampling on raw text before TF-IDF.
Duplicates minority-class sentences to match the majority class size.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from pipeline.libs._train_core import (
    setup_logging, build_tfidf_features,
    train_lr, train_nb, train_svm, save_models, logger,
)
from pipeline.libs.config import OUTPUT_PATH, TEST_SIZE, RANDOM_STATE


def load_splits_with_oversample():
    df = pd.read_csv(OUTPUT_PATH)
    df = df.dropna(subset=['Kalimat Dinormalisasi'])
    df = df[df['Kalimat Dinormalisasi'].str.strip() != ''].reset_index(drop=True)
    X = df['Kalimat Dinormalisasi']
    y = df['Level Kata Kasar']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    max_count  = y_train.value_counts().max()
    parts_X, parts_y = [], []
    for label in sorted(y_train.unique()):
        mask = (y_train == label).values
        Xs   = X_train[mask].reset_index(drop=True)
        ys   = y_train[mask].reset_index(drop=True)
        if len(ys) < max_count:
            idx  = resample(range(len(ys)), n_samples=max_count,
                            random_state=RANDOM_STATE, replace=True)
            Xs, ys = Xs.iloc[idx], ys.iloc[idx]
        parts_X.append(Xs)
        parts_y.append(ys)

    X_train = pd.concat(parts_X).reset_index(drop=True)
    y_train = pd.concat(parts_y).reset_index(drop=True)
    logger.info(f'Data loaded | train={len(y_train):,} (after oversampling) | test={len(y_test):,}')
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    log_path = setup_logging('classical_random_oversample.log')
    logger.info('=' * 60)
    logger.info('Strategy: random oversampling on raw text')

    X_train, X_test, y_train, y_test = load_splits_with_oversample()
    tfidf_lr, tfidf_nb, X_train_lr, X_test_lr, X_train_nb, X_test_nb = \
        build_tfidf_features(X_train, X_test)

    lr_model,  _ = train_lr(X_train_lr,  y_train, X_test_lr,  y_test)
    nb_model,  _ = train_nb(X_train_nb,  y_train, X_test_nb,  y_test)
    svm_model, _ = train_svm(X_train_lr, y_train, X_test_lr,  y_test)

    save_models(lr_model, tfidf_lr, nb_model, tfidf_nb, svm_model)
    logger.info('=' * 60)
    logger.info(f'Complete. Log: {log_path}')
