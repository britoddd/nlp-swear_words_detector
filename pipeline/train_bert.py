"""
Fine-tune IndoBERTweet with Optuna hyperparameter tuning.

Split: 80% train / 10% val / 10% test (stratified).
Oversampling applied to training split only.

Tuning phase  : N_TRIALS trials, each training TRIAL_EPOCHS epochs,
                evaluated on val F1. Test set is never seen.
Final training : best hyperparameters, full EPOCHS, train loss + val F1
                 reported each epoch.
Test results  : printed only after final training is complete.

Usage:
    python train_bert.py

Reads:  dataset_processed.csv
Writes: saved_models/indobert_finetuned/
        outputs/hparam_tuning.csv
        outputs/bert_training_history.png
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup, logging as hf_logging,
)
from sklearn.metrics import f1_score, classification_report
from sklearn.utils import resample
from tqdm.auto import tqdm

from config import (
    OUTPUT_PATH, BERT_SAVE_PATH, INDOBERT_MODEL, FORCE_RETRAIN, OUTPUTS_DIR,
    MODELS_DIR, MAX_LEN, BATCH_SIZE, EPOCHS, RANDOM_STATE,
    N_TRIALS, TRIAL_EPOCHS, device,
)

# Local snapshot of the base model — downloaded once, reused every Optuna trial
# so trials never depend on network access or HuggingFace cache availability.
_BASE_MODEL_CACHE = os.path.join(MODELS_DIR, 'indobert_base_cached')
from data_loader import make_splits

hf_logging.set_verbosity_error()
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Logging ────────────────────────────────────────────────────
# Mirrors the classical trainers' setup_logging(): every trial's parameters and
# each epoch's train-loss / val-F1 are written to both the console and a file so
# a full record survives even if the run later crashes.
LOG_PATH = os.path.join(OUTPUTS_DIR, 'bert_training_log.txt')
logger = logging.getLogger('bert_training')
logger.setLevel(logging.INFO)


def setup_logging() -> str:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(LOG_PATH, encoding='utf-8')
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return LOG_PATH

# Conda sometimes sets SSL_CERT_FILE to a path that no longer exists, which
# causes httpx (used by huggingface_hub) to crash when building an SSL context.
_ssl_cert = os.environ.get('SSL_CERT_FILE')
if _ssl_cert and not os.path.isfile(_ssl_cert):
    del os.environ['SSL_CERT_FILE']


# ── Dataset ────────────────────────────────────────────────────

class ProfanityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts     = list(texts)
        self.labels    = list(labels)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Data helpers ───────────────────────────────────────────────

def random_oversample_bert(X_bert: pd.Series, y: pd.Series,
                            random_state: int = RANDOM_STATE) -> tuple:
    max_count = y.value_counts().max()
    parts_X, parts_y = [], []
    for label in sorted(y.unique()):
        mask = (y == label).values
        Xb = X_bert[mask].reset_index(drop=True)
        ys = y[mask].reset_index(drop=True)
        if len(ys) < max_count:
            idx = resample(range(len(ys)), n_samples=max_count,
                           random_state=random_state, replace=True)
            Xb, ys = Xb.iloc[idx], ys.iloc[idx]
        parts_X.append(Xb)
        parts_y.append(ys)
    return (pd.concat(parts_X).reset_index(drop=True),
            pd.concat(parts_y).reset_index(drop=True))


def load_splits():
    from preprocess_pipeline import preprocess

    df = pd.read_csv(OUTPUT_PATH)
    df = df.dropna(subset=['Kalimat Asli', 'Kalimat Dinormalisasi'])
    df = df[df['Kalimat Dinormalisasi'].str.strip() != ''].reset_index(drop=True)

    tqdm.pandas(desc='Regenerating BERT input')
    df['Kalimat Bert'] = df['Kalimat Asli'].progress_apply(
        lambda x: preprocess(x, for_bert=True)
    )
    df = df[df['Kalimat Bert'].str.strip() != ''].reset_index(drop=True)

    X_ml   = df['Kalimat Dinormalisasi']
    X_bert = df['Kalimat Bert']
    y      = df['Level Kata Kasar']

    (_, _, _,
     X_train_bert, X_val_bert, X_test_bert,
     y_train, y_val, y_test) = make_splits(X_ml, y, X_bert)

    X_train_bert, y_train = random_oversample_bert(X_train_bert, y_train)

    print(f'Train: {len(y_train):,} (after oversampling) | Val: {len(y_val):,} | Test: {len(y_test):,}')
    return X_train_bert, X_val_bert, X_test_bert, y_train, y_val, y_test


def _make_loader(texts, labels, tokenizer, shuffle: bool, batch_size: int = BATCH_SIZE):
    ds = ProfanityDataset(texts, labels.reset_index(drop=True), tokenizer)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ── Training / eval primitives ─────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc='  Training', leave=False):
        optimizer.zero_grad()
        out = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            labels=batch['labels'].to(device),
        )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += out.loss.item()
    return total_loss / len(loader)


def run_eval(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='  Evaluating', leave=False):
            out = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
            )
            preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            trues.extend(batch['labels'].numpy())
    return np.array(preds), np.array(trues)


def _build_optimizer_scheduler(model, loader_len, n_epochs, lr, weight_decay, warmup_ratio):
    optimizer    = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps  = loader_len * n_epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    return optimizer, scheduler


# ── Optuna tuning ──────────────────────────────────────────────

def _objective(trial, X_train, X_val, y_train, y_val, tokenizer):
    """One Optuna trial: train TRIAL_EPOCHS, return val F1."""
    lr           = trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True)
    weight_decay = trial.suggest_float('weight_decay',  0.0,  0.1)
    warmup_ratio = trial.suggest_float('warmup_ratio',  0.0,  0.2)
    batch_size   = trial.suggest_categorical('batch_size', [16, 32])

    logger.info(
        'Trial %d START | learning_rate=%.3e weight_decay=%.4f '
        'warmup_ratio=%.4f batch_size=%d',
        trial.number, lr, weight_decay, warmup_ratio, batch_size,
    )

    train_loader = _make_loader(X_train, y_train, tokenizer, shuffle=True,  batch_size=batch_size)
    val_loader   = _make_loader(X_val,   y_val,   tokenizer, shuffle=False, batch_size=batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        _BASE_MODEL_CACHE, num_labels=4, ignore_mismatched_sizes=True,
    ).to(device)

    optimizer, scheduler = _build_optimizer_scheduler(
        model, len(train_loader), TRIAL_EPOCHS, lr, weight_decay, warmup_ratio
    )

    val_f1 = 0.0
    for epoch in range(TRIAL_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
        val_pred, val_true = run_eval(model, val_loader)
        val_f1 = f1_score(val_true, val_pred, average='macro')
        logger.info(
            'Trial %d | epoch %d/%d | train_loss=%.4f | val_f1=%.4f',
            trial.number, epoch + 1, TRIAL_EPOCHS, train_loss, val_f1,
        )
        trial.report(val_f1, epoch)
        if trial.should_prune():
            logger.info('Trial %d PRUNED at epoch %d (val_f1=%.4f)',
                        trial.number, epoch + 1, val_f1)
            del model
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()

    logger.info('Trial %d COMPLETE | final val_f1=%.4f', trial.number, val_f1)
    del model
    torch.cuda.empty_cache()
    return val_f1


def run_tuning(X_train, X_val, y_train, y_val, tokenizer) -> tuple[dict, optuna.Study]:
    logger.info('=' * 55)
    logger.info('Hyperparameter Tuning — %d trials × %d epochs each',
                N_TRIALS, TRIAL_EPOCHS)
    logger.info('Search space: learning_rate, weight_decay, warmup_ratio, batch_size')
    logger.info('=' * 55)

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study  = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=pruner,
    )
    study.optimize(
        lambda trial: _objective(trial, X_train, X_val, y_train, y_val, tokenizer),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    # Trial summary — 'intermediate_values' adds one column per epoch
    # (intermediate_value_0, _1, …) so the per-epoch val-F1 of every trial is
    # persisted, not just each trial's final score.
    trials_df = study.trials_dataframe(
        attrs=('number', 'value', 'params', 'state', 'intermediate_values')
    )
    logger.info('All trials:\n%s', trials_df.to_string(index=False))

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    tuning_path = os.path.join(OUTPUTS_DIR, 'hparam_tuning.csv')
    trials_df.to_csv(tuning_path, index=False)
    logger.info('Saved → %s', tuning_path)

    best = study.best_params
    logger.info('Best trial #%d  Val F1 = %.4f',
                study.best_trial.number, study.best_value)
    for k, v in best.items():
        logger.info('  %s: %s', k, v)

    return best, study


# ── Final training with best hyperparameters ───────────────────

def train_final(best_params, X_train, X_val, X_test, y_train, y_val, y_test, tokenizer):
    lr           = best_params['learning_rate']
    weight_decay = best_params['weight_decay']
    warmup_ratio = best_params['warmup_ratio']
    batch_size   = best_params['batch_size']

    logger.info('=' * 55)
    logger.info('Final training — %d epochs on %s', EPOCHS, device)
    logger.info('  learning_rate = %.2e  weight_decay = %.4f', lr, weight_decay)
    logger.info('  warmup_ratio  = %.4f  batch_size = %d', warmup_ratio, batch_size)
    logger.info('=' * 55)

    train_loader = _make_loader(X_train, y_train, tokenizer, shuffle=True,  batch_size=batch_size)
    val_loader   = _make_loader(X_val,   y_val,   tokenizer, shuffle=False, batch_size=batch_size)
    test_loader  = _make_loader(X_test,  y_test,  tokenizer, shuffle=False, batch_size=batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        _BASE_MODEL_CACHE, num_labels=4, ignore_mismatched_sizes=True,
    ).to(device)

    optimizer, scheduler = _build_optimizer_scheduler(
        model, len(train_loader), EPOCHS, lr, weight_decay, warmup_ratio
    )

    logger.info('%5s  %10s  %8s', 'Epoch', 'Train Loss', 'Val F1')
    history = []
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
        val_pred, val_true = run_eval(model, val_loader)
        val_f1 = f1_score(val_true, val_pred, average='macro')
        history.append({'epoch': epoch, 'loss': train_loss, 'val_f1': val_f1})
        logger.info('%5d  %10.4f  %8.4f', epoch, train_loss, val_f1)

    # Save training curve + the underlying per-epoch numbers
    hist_df = pd.DataFrame(history)
    hist_path = os.path.join(OUTPUTS_DIR, 'bert_final_history.csv')
    hist_df.to_csv(hist_path, index=False)
    logger.info('Saved → %s', hist_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(hist_df['epoch'], hist_df['loss'],   marker='o', color='#e74c3c')
    ax1.set_title('Train Loss');     ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax2.plot(hist_df['epoch'], hist_df['val_f1'], marker='o', color='#3498db')
    ax2.set_title('Val F1 (macro)'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('F1')
    plt.tight_layout()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    curve_path = os.path.join(OUTPUTS_DIR, 'bert_training_history.png')
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('Saved → %s', curve_path)

    # Save model
    os.makedirs(BERT_SAVE_PATH, exist_ok=True)
    model.save_pretrained(BERT_SAVE_PATH)
    tokenizer.save_pretrained(BERT_SAVE_PATH)
    logger.info('Saved → %s', BERT_SAVE_PATH)

    # ── Test evaluation — only after training is complete ─────────
    logger.info('=== TEST SET RESULTS (held-out) ===')
    y_pred, y_true = run_eval(model, test_loader)
    f1 = f1_score(y_true, y_pred, average='macro')
    logger.info('IndoBERT  F1 (macro) = %.4f', f1)
    logger.info('\n%s', classification_report(y_true, y_pred,
          target_names=['Level 0', 'Level 1', 'Level 2', 'Level 3']))


# ── Entry point ────────────────────────────────────────────────

def main():
    log_path = setup_logging()
    logger.info('Logging to %s', log_path)

    tokenizer = AutoTokenizer.from_pretrained(
        BERT_SAVE_PATH if os.path.exists(BERT_SAVE_PATH) else INDOBERT_MODEL
    )

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    if not FORCE_RETRAIN and os.path.exists(BERT_SAVE_PATH):
        logger.info('Loading fine-tuned weights from %s', BERT_SAVE_PATH)
        logger.info('Set FORCE_RETRAIN=True in config.py to retrain.')
        # Already-fine-tuned 4-label weights — surface a mismatch instead of
        # silently reinitialising the classifier head.
        model = AutoModelForSequenceClassification.from_pretrained(
            BERT_SAVE_PATH, num_labels=4,
        ).to(device)
        test_loader = _make_loader(X_test, y_test, tokenizer, shuffle=False)
        logger.info('=== TEST SET RESULTS (held-out) ===')
        y_pred, y_true = run_eval(model, test_loader)
        f1 = f1_score(y_true, y_pred, average='macro')
        logger.info('IndoBERT  F1 (macro) = %.4f', f1)
        logger.info('\n%s', classification_report(y_true, y_pred,
              target_names=['Level 0', 'Level 1', 'Level 2', 'Level 3']))
        return

    # Download and snapshot the base model once so every Optuna trial loads
    # from local disk and never fails due to network or cache issues.
    if not os.path.isfile(os.path.join(_BASE_MODEL_CACHE, 'config.json')):
        logger.info('Caching base model to %s ...', _BASE_MODEL_CACHE)
        os.makedirs(_BASE_MODEL_CACHE, exist_ok=True)
        _base = AutoModelForSequenceClassification.from_pretrained(
            INDOBERT_MODEL, num_labels=4, ignore_mismatched_sizes=True,
        )
        _base.save_pretrained(_BASE_MODEL_CACHE)
        tokenizer.save_pretrained(_BASE_MODEL_CACHE)
        del _base
        torch.cuda.empty_cache()

    best_params, _ = run_tuning(X_train, X_val, y_train, y_val, tokenizer)
    train_final(best_params, X_train, X_val, X_test, y_train, y_val, y_test, tokenizer)


if __name__ == '__main__':
    # Guard the whole run so a fatal error (OOM, CUDA abort, etc.) is recorded
    # in the log file with a full traceback instead of vanishing on stderr.
    setup_logging()
    try:
        main()
    except Exception:
        logger.exception('BERT training aborted with an unhandled exception')
        raise
