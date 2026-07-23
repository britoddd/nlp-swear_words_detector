"""
Strategy: SMOTEENN on raw TF-IDF features (no SVD compression).
Warning: ENN nearest-neighbour search on 50k sparse features is very slow.
Use train_classical.py (SVD+SMOTEENN) for a faster equivalent.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours

from pipeline.libs._train_core import (
    setup_logging, load_splits, build_tfidf_features,
    train_lr, train_nb, train_svm, save_models, logger,
)
from pipeline.libs.config import RANDOM_STATE


def apply_smoteenn(X, y):
    enn     = EditedNearestNeighbours(n_jobs=-1)
    X_res, y_res = SMOTEENN(random_state=RANDOM_STATE, enn=enn).fit_resample(X, y)
    logger.info(f'SMOTEENN applied | total={len(y_res):,}')
    return X_res, y_res


if __name__ == '__main__':
    log_path = setup_logging('classical_smoteenn_raw.log')
    logger.info('=' * 60)
    logger.info('Strategy: SMOTEENN on raw TF-IDF (50k features) — expect slow ENN step')

    X_train, X_test, y_train, y_test = load_splits()
    tfidf_lr, tfidf_nb, X_train_lr, X_test_lr, X_train_nb, X_test_nb = \
        build_tfidf_features(X_train, X_test)

    X_train_lr_s, y_train_lr_s = apply_smoteenn(X_train_lr, y_train)
    X_train_nb_s, y_train_nb_s = apply_smoteenn(X_train_nb, y_train)

    lr_model,  _ = train_lr(X_train_lr_s,  y_train_lr_s, X_test_lr,  y_test)
    nb_model,  _ = train_nb(X_train_nb_s,  y_train_nb_s, X_test_nb,  y_test)
    svm_model, _ = train_svm(X_train_lr_s, y_train_lr_s, X_test_lr,  y_test)

    save_models(lr_model, tfidf_lr, nb_model, tfidf_nb, svm_model)
    logger.info('=' * 60)
    logger.info(f'Complete. Log: {log_path}')
