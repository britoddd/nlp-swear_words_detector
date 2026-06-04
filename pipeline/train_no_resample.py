"""
Strategy: No resampling.
Class imbalance handled solely by class_weight='balanced' in LR and SVM.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _train_core import (
    setup_logging, load_splits, build_tfidf_features,
    train_lr, train_nb, train_svm, save_models, logger,
)
from config import RANDOM_STATE

if __name__ == '__main__':
    log_path = setup_logging('classical_no_resample.log')
    logger.info('=' * 60)
    logger.info('Strategy: no resampling')

    X_train, X_test, y_train, y_test = load_splits()
    tfidf_lr, tfidf_nb, X_train_lr, X_test_lr, X_train_nb, X_test_nb = \
        build_tfidf_features(X_train, X_test)

    lr_model,  _ = train_lr(X_train_lr,  y_train, X_test_lr,  y_test)
    nb_model,  _ = train_nb(X_train_nb,  y_train, X_test_nb,  y_test)
    svm_model, _ = train_svm(X_train_lr, y_train, X_test_lr,  y_test)

    save_models(lr_model, tfidf_lr, nb_model, tfidf_nb, svm_model)
    logger.info('=' * 60)
    logger.info(f'Complete. Log: {log_path}')
