"""
Load all saved models and produce evaluation outputs:
  - Per-model classification reports
  - confusion_matrices.png
  - model_comparison.png

Usage:
    python evaluate.py

Reads:  dataset_processed.csv, saved_models/
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
from tqdm.auto import tqdm

from config import (
    OUTPUT_PATH, BERT_SAVE_PATH, OUTPUTS_DIR,
    LR_MODEL_PATH, LR_TFIDF_PATH, NB_MODEL_PATH, NB_TFIDF_PATH, SVM_MODEL_PATH,
    MAX_LEN, BATCH_SIZE, device,
)
from data_loader import make_splits
from train_bert import ProfanityDataset

hf_logging.set_verbosity_error()


def load_test_split():
    from preprocess_pipeline import preprocess
    df = pd.read_csv(OUTPUT_PATH)
    df = df.dropna(subset=['Kalimat Asli', 'Kalimat Dinormalisasi'])
    df = df[df['Kalimat Dinormalisasi'].str.strip() != ''].reset_index(drop=True)

    # Kalimat Bert is not stored in the CSV — regenerate it (no stemming, no slang norm)
    tqdm.pandas(desc='Regenerating BERT input')
    df['Kalimat Bert'] = df['Kalimat Asli'].progress_apply(
        lambda x: preprocess(x, for_bert=True)
    )
    df = df[df['Kalimat Bert'].str.strip() != ''].reset_index(drop=True)

    X_ml   = df['Kalimat Dinormalisasi']
    X_bert = df['Kalimat Bert']
    y      = df['Level Kata Kasar']

    # Use same make_splits() as training scripts — guarantees identical test set
    _, _, X_test_ml, _, _, X_test_bert, _, _, y_test = make_splits(X_ml, y, X_bert)
    return X_test_ml, X_test_bert, y_test


def predict_classical(X_test_ml):
    with open(LR_TFIDF_PATH,  'rb') as f: tfidf_lr  = pickle.load(f)
    with open(NB_TFIDF_PATH,  'rb') as f: tfidf_nb  = pickle.load(f)
    with open(LR_MODEL_PATH,  'rb') as f: lr_model  = pickle.load(f)
    with open(NB_MODEL_PATH,  'rb') as f: nb_model  = pickle.load(f)
    with open(SVM_MODEL_PATH, 'rb') as f: svm_model = pickle.load(f)

    X_lr = tfidf_lr.transform(X_test_ml)
    X_nb = tfidf_nb.transform(X_test_ml)

    return (lr_model.predict(X_lr),
            nb_model.predict(X_nb),
            svm_model.predict(X_lr))


def predict_bert(X_test_bert, y_test):
    tokenizer = AutoTokenizer.from_pretrained(BERT_SAVE_PATH)
    # These are already-fine-tuned 4-label weights; a size mismatch here means
    # something is wrong, so let it raise rather than silently reinit the head.
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_SAVE_PATH, num_labels=4,
    ).to(device)
    model.eval()

    test_ds = ProfanityDataset(X_test_bert, y_test, tokenizer, max_len=MAX_LEN)
    loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='BERT inference'):
            out = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
            )
            preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
    return np.array(preds)


def plot_confusion_matrices(y_test, y_pred_lr, y_pred_nb, y_pred_svm, y_pred_bert):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    specs = [
        (y_test, y_pred_lr,   f'Logistic Regression'),
        (y_test, y_pred_nb,   f'Naive Bayes'),
        (y_test, y_pred_svm,  f'SVM (LinearSVC)'),
        (y_test, y_pred_bert, f'IndoBERT'),
    ]
    for ax, (yt, yp, title) in zip(axes, specs):
        cm = confusion_matrix(yt, yp, labels=[0, 1, 2, 3])
        # Draw colours with seaborn but annotate manually: seaborn's own
        # annot=True path silently drops all but the first row under some
        # matplotlib / numpy-2.x backend combinations.
        sns.heatmap(cm, annot=False, cmap='Blues', ax=ax, cbar=True,
                    xticklabels=['L0', 'L1', 'L2', 'L3'],
                    yticklabels=['L0', 'L1', 'L2', 'L3'])
        thresh = cm.max() / 2 if cm.max() else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j + 0.5, i + 0.5, format(cm[i, j], 'd'),
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')
    plt.suptitle('Confusion Matrices – Perbandingan Model',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUTS_DIR, 'confusion_matrices.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved → {out_path}')
    plt.show()


def plot_model_comparison(results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(results))
    w = 0.25
    bars = [
        ax.bar(x - w, results['Precision (macro)'], w, label='Precision', color='#3498db'),
        ax.bar(x,     results['Recall (macro)'],    w, label='Recall',    color='#2ecc71'),
        ax.bar(x + w, results['F1 (macro)'],        w, label='F1 (macro)',color='#e74c3c'),
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(results.index)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Perbandingan Performa Model', fontweight='bold')
    ax.legend()
    for bar_group in bars:
        for bar in bar_group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, 'model_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved → {out_path}')
    plt.show()


if __name__ == '__main__':
    X_test_ml, X_test_bert, y_test = load_test_split()

    y_pred_lr, y_pred_nb, y_pred_svm = predict_classical(X_test_ml)
    y_pred_bert = predict_bert(X_test_bert, y_test)

    f1_lr   = f1_score(y_test, y_pred_lr,   average='macro')
    f1_nb   = f1_score(y_test, y_pred_nb,   average='macro')
    f1_svm  = f1_score(y_test, y_pred_svm,  average='macro')
    f1_bert = f1_score(y_test, y_pred_bert, average='macro')

    print('\n=== Classification Reports ===')
    for name, yt, yp in [
        ('Logistic Regression', y_test, y_pred_lr),
        ('Naive Bayes',          y_test, y_pred_nb),
        ('SVM (LinearSVC)',      y_test, y_pred_svm),
        ('IndoBERT',             y_test, y_pred_bert),
    ]:
        print(f'\n--- {name} ---')
        print(classification_report(yt, yp,
              target_names=['Level 0', 'Level 1', 'Level 2', 'Level 3']))

    def get_metrics(yt, yp):
        return {
            'Precision (macro)': precision_score(yt, yp, average='macro', zero_division=0),
            'Recall (macro)':    recall_score(yt, yp,    average='macro', zero_division=0),
            'F1 (macro)':        f1_score(yt, yp,        average='macro', zero_division=0),
        }

    results = pd.DataFrame({
        'Logistic Regression': get_metrics(y_test, y_pred_lr),
        'Naive Bayes':         get_metrics(y_test, y_pred_nb),
        'SVM (LinearSVC)':     get_metrics(y_test, y_pred_svm),
        'IndoBERT':            get_metrics(y_test, y_pred_bert),
    }).T.round(4)

    print('\n=== Model Comparison ===')
    print(results.to_string())

    plot_confusion_matrices(y_test, y_pred_lr, y_pred_nb, y_pred_svm, y_pred_bert)
    plot_model_comparison(results)
