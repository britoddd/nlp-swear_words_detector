"""
Strategy: SMOTE on TF-IDF features.
Synthesizes new minority-class samples by interpolating in TF-IDF feature space.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from imblearn.over_sampling import SMOTE

from _train_core import (
    setup_logging, load_splits, build_tfidf_features,
    train_lr, train_nb, train_svm, save_models, logger,
)
from config import RANDOM_STATE


def apply_smote(X, y):
    X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)
    logger.info(f'SMOTE applied | total={len(y_res):,}')
    return X_res, y_res


if __name__ == '__main__':
    log_path = setup_logging('classical_smote.log')
    logger.info('=' * 60)
    logger.info('Strategy: SMOTE on TF-IDF features')

    X_train, X_test, y_train, y_test = load_splits()
    tfidf_lr, tfidf_nb, X_train_lr, X_test_lr, X_train_nb, X_test_nb = \
        build_tfidf_features(X_train, X_test)

    X_train_lr_s, y_train_lr_s = apply_smote(X_train_lr, y_train)
    X_train_nb_s, y_train_nb_s = apply_smote(X_train_nb, y_train)

    lr_model,  _ = train_lr(X_train_lr_s,  y_train_lr_s, X_test_lr,  y_test)
    nb_model,  _ = train_nb(X_train_nb_s,  y_train_nb_s, X_test_nb,  y_test)
    svm_model, _ = train_svm(X_train_lr_s, y_train_lr_s, X_test_lr,  y_test)

    save_models(lr_model, tfidf_lr, nb_model, tfidf_nb, svm_model)
    logger.info('=' * 60)
    logger.info(f'Complete. Log: {log_path}')
