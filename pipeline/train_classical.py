"""
Train Logistic Regression, Naive Bayes, and SVM classifiers.

Usage:
    python train_classical.py

Reads:  dataset_processed.csv  (produced by preprocess_pipeline.py)
Writes: saved_models/lr_model.pkl, lr_tfidf.pkl, nb_model.pkl,
        nb_tfidf.pkl, svm_model.pkl
        logs/classical_training.log
"""

import logging
import os
import pickle
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, classification_report

from config import (
    OUTPUT_PATH, MODELS_DIR, FORCE_RETRAIN,
    LR_MODEL_PATH, LR_TFIDF_PATH, NB_MODEL_PATH, NB_TFIDF_PATH, SVM_MODEL_PATH,
    TEST_SIZE, RANDOM_STATE, N_TRIALS,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOGS_DIR = os.path.join(_ROOT, 'logs')
os.makedirs(_LOGS_DIR, exist_ok=True)

_LOG_PATH = os.path.join(_LOGS_DIR, 'classical_training.log')

logger = logging.getLogger('classical_training')
logger.setLevel(logging.INFO)
if not logger.handlers:
    _fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                             datefmt='%Y-%m-%d %H:%M:%S')
    _fh = logging.FileHandler(_LOG_PATH, encoding='utf-8')
    _fh.setFormatter(_fmt)
    _ch = logging.StreamHandler()
    _ch.setFormatter(_fmt)
    logger.addHandler(_fh)
    logger.addHandler(_ch)

_TARGET_NAMES = ['Level 0', 'Level 1', 'Level 2', 'Level 3']


def load_splits():
    df = pd.read_csv(OUTPUT_PATH)
    df = df.dropna(subset=['Kalimat Dinormalisasi'])
    df = df[df['Kalimat Dinormalisasi'].str.strip() != ''].reset_index(drop=True)

    X = df['Kalimat Dinormalisasi']
    y = df['Level Kata Kasar']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f'Data loaded | train={len(y_train):,} | test={len(y_test):,}')
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


# ── Hyperparameter tuning ──────────────────────────────────────────────────────

def tune_lr(X_train, y_train) -> dict:
    logger.info(f'LR: starting hyperparameter tuning ({N_TRIALS} trials)')

    def objective(trial):
        C      = trial.suggest_float('C', 1e-3, 100.0, log=True)
        solver = trial.suggest_categorical('solver', ['lbfgs', 'saga'])
        model  = LogisticRegression(C=C, solver=solver, max_iter=1000,
                                    random_state=RANDOM_STATE, class_weight='balanced')
        val_f1 = cross_val_score(model, X_train, y_train,
                                 cv=3, scoring='f1_macro', n_jobs=-1).mean()
        model.fit(X_train, y_train)
        train_f1 = f1_score(y_train, model.predict(X_train), average='macro')
        logger.info(f'LR | trial {trial.number:>3d} | params={trial.params} '
                    f'| train_F1={train_f1:.4f} | val_CV_F1={val_f1:.4f}')
        return val_f1

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    logger.info(f'LR BEST | params={study.best_params} | val_CV_F1={study.best_value:.4f}')
    return study.best_params


def tune_nb(X_train, y_train) -> dict:
    logger.info(f'NB: starting hyperparameter tuning ({N_TRIALS} trials)')

    def objective(trial):
        alpha = trial.suggest_float('alpha', 1e-3, 10.0, log=True)
        norm  = trial.suggest_categorical('norm', [True, False])
        model = ComplementNB(alpha=alpha, norm=norm)
        val_f1 = cross_val_score(model, X_train, y_train,
                                 cv=3, scoring='f1_macro', n_jobs=-1).mean()
        model.fit(X_train, y_train)
        train_f1 = f1_score(y_train, model.predict(X_train), average='macro')
        logger.info(f'NB | trial {trial.number:>3d} | params={trial.params} '
                    f'| train_F1={train_f1:.4f} | val_CV_F1={val_f1:.4f}')
        return val_f1

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    logger.info(f'NB BEST | params={study.best_params} | val_CV_F1={study.best_value:.4f}')
    return study.best_params


def tune_svm(X_train, y_train) -> dict:
    logger.info(f'SVM: starting hyperparameter tuning ({N_TRIALS} trials)')

    def objective(trial):
        C     = trial.suggest_float('C', 1e-3, 100.0, log=True)
        model = CalibratedClassifierCV(
            LinearSVC(C=C, max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE),
            cv=3,
        )
        val_f1 = cross_val_score(model, X_train, y_train,
                                 cv=3, scoring='f1_macro', n_jobs=-1).mean()
        model.fit(X_train, y_train)
        train_f1 = f1_score(y_train, model.predict(X_train), average='macro')
        logger.info(f'SVM | trial {trial.number:>3d} | params={trial.params} '
                    f'| train_F1={train_f1:.4f} | val_CV_F1={val_f1:.4f}')
        return val_f1

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    logger.info(f'SVM BEST | params={study.best_params} | val_CV_F1={study.best_value:.4f}')
    return study.best_params


# ── Model training ─────────────────────────────────────────────────────────────

def _log_metrics(model_name: str, split: str, y_true, y_pred):
    f1     = f1_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred, target_names=_TARGET_NAMES)
    logger.info(f'{model_name} | {split} | macro_F1={f1:.4f}')
    logger.info(f'{model_name} | {split} | classification report:\n{report}')
    return f1


def train_lr(X_train, y_train, X_test, y_test):
    logger.info('=' * 60)
    logger.info('LR: training start')
    if not FORCE_RETRAIN and os.path.exists(LR_MODEL_PATH):
        logger.info('LR: loading from disk')
        with open(LR_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        best = None
    else:
        best  = tune_lr(X_train, y_train)
        logger.info(f'LR: training final model with best params={best}')
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                   class_weight='balanced', **best)
        model.fit(X_train, y_train)

    _log_metrics('LR', 'TRAIN', y_train, model.predict(X_train))
    y_pred = model.predict(X_test)
    _log_metrics('LR', 'TEST', y_test, y_pred)
    if best:
        logger.info(f'LR SUMMARY | best_params={best}')
    return model, y_pred


def train_nb(X_train, y_train, X_test, y_test):
    logger.info('=' * 60)
    logger.info('NB: training start')
    if not FORCE_RETRAIN and os.path.exists(NB_MODEL_PATH):
        logger.info('NB: loading from disk')
        with open(NB_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        best = None
    else:
        best  = tune_nb(X_train, y_train)
        logger.info(f'NB: training final model with best params={best}')
        model = ComplementNB(**best)
        model.fit(X_train, y_train)

    _log_metrics('NB', 'TRAIN', y_train, model.predict(X_train))
    y_pred = model.predict(X_test)
    _log_metrics('NB', 'TEST', y_test, y_pred)
    if best:
        logger.info(f'NB SUMMARY | best_params={best}')
    return model, y_pred


def train_svm(X_train, y_train, X_test, y_test):
    logger.info('=' * 60)
    logger.info('SVM: training start')
    if not FORCE_RETRAIN and os.path.exists(SVM_MODEL_PATH):
        logger.info('SVM: loading from disk')
        with open(SVM_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        best = None
    else:
        best     = tune_svm(X_train, y_train)
        logger.info(f'SVM: training final model with best params={best}')
        svm_base = LinearSVC(max_iter=2000, class_weight='balanced',
                             random_state=RANDOM_STATE, **best)
        model    = CalibratedClassifierCV(svm_base, cv=5)
        model.fit(X_train, y_train)

    _log_metrics('SVM', 'TRAIN', y_train, model.predict(X_train))
    y_pred = model.predict(X_test)
    _log_metrics('SVM', 'TEST', y_test, y_pred)
    if best:
        logger.info(f'SVM SUMMARY | best_params={best}')
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
    logger.info(f'Models saved to {MODELS_DIR}/')


if __name__ == '__main__':
    logger.info('=' * 60)
    logger.info('Classical training session started')
    X_train, X_test, y_train, y_test = load_splits()
    tfidf_lr, tfidf_nb, X_train_lr, X_test_lr, X_train_nb, X_test_nb = \
        build_tfidf_features(X_train, X_test)

    lr_model,  _ = train_lr(X_train_lr,  y_train, X_test_lr,  y_test)
    nb_model,  _ = train_nb(X_train_nb,  y_train, X_test_nb,  y_test)
    svm_model, _ = train_svm(X_train_lr, y_train, X_test_lr,  y_test)

    save_models(lr_model, tfidf_lr, nb_model, tfidf_nb, svm_model)
    logger.info('=' * 60)
    logger.info(f'Classical training session complete. Log: {_LOG_PATH}')
