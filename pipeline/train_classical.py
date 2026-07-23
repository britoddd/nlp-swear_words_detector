"""
Train Logistic Regression, Naive Bayes, and SVM classifiers (main classical
trainer — no special resampling; imbalance handled by class_weight='balanced').

All training logic lives in _train_core.py, which is shared with the
resampling-strategy comparison scripts (train_smote.py, train_undersample.py,
…). This file only orchestrates the no-resampling run and uses a larger Optuna
budget (config.N_TRIALS_CLASSICAL) than the quick strategy comparisons.

Usage:
    python train_classical.py

Reads:  dataset_processed.csv  (produced by preprocess_pipeline.py)
Writes: saved_models/lr_model.pkl, lr_tfidf.pkl, nb_model.pkl,
        nb_tfidf.pkl, svm_model.pkl
        logs/classical_training.log
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.libs._train_core import (
    setup_logging, load_splits, build_tfidf_features,
    train_lr, train_nb, train_svm, save_models, logger,
)
from pipeline.libs.config import N_TRIALS_CLASSICAL

if __name__ == '__main__':
    log_path = setup_logging('classical_training.log')
    logger.info('=' * 60)
    logger.info(f'Classical training session started ({N_TRIALS_CLASSICAL} trials)')

    X_train, X_test, y_train, y_test = load_splits()
    tfidf_lr, tfidf_nb, X_train_lr, X_test_lr, X_train_nb, X_test_nb = \
        build_tfidf_features(X_train, X_test)

    lr_model,  _ = train_lr(X_train_lr,  y_train, X_test_lr, y_test, n_trials=N_TRIALS_CLASSICAL)
    nb_model,  _ = train_nb(X_train_nb,  y_train, X_test_nb, y_test, n_trials=N_TRIALS_CLASSICAL)
    svm_model, _ = train_svm(X_train_lr, y_train, X_test_lr, y_test, n_trials=N_TRIALS_CLASSICAL)

    save_models(lr_model, tfidf_lr, nb_model, tfidf_nb, svm_model)
    logger.info('=' * 60)
    logger.info(f'Classical training session complete. Log: {log_path}')
