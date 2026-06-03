"""
Fine-tune IndoBERTweet for 4-class profanity classification.

Usage:
    python train_bert.py

Reads:  dataset_processed.csv  (produced by preprocess_pipeline.py)
Writes: saved_models/indobert_finetuned/
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup, logging as hf_logging,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.utils import resample
from tqdm.auto import tqdm

from config import (
    OUTPUT_PATH, BERT_SAVE_PATH, INDOBERT_MODEL, FORCE_RETRAIN, OUTPUTS_DIR,
    MAX_LEN, BATCH_SIZE, EPOCHS, LEARNING_RATE, RANDOM_STATE, TEST_SIZE, device,
)

hf_logging.set_verbosity_error()


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


def random_oversample_bert(X_ml, X_bert, y, random_state=42):
    max_count = y.value_counts().max()
    parts_ml, parts_bert, parts_y = [], [], []
    for label in sorted(y.unique()):
        mask = (y == label).values
        Xm = X_ml[mask].reset_index(drop=True)
        Xb = X_bert[mask].reset_index(drop=True)
        ys = y[mask].reset_index(drop=True)
        if len(ys) < max_count:
            idx = resample(range(len(ys)), n_samples=max_count,
                           random_state=random_state, replace=True)
            Xm, Xb, ys = Xm.iloc[idx], Xb.iloc[idx], ys.iloc[idx]
        parts_ml.append(Xm)
        parts_bert.append(Xb)
        parts_y.append(ys)
    return (pd.concat(parts_ml).reset_index(drop=True),
            pd.concat(parts_bert).reset_index(drop=True),
            pd.concat(parts_y).reset_index(drop=True))


def load_splits():
    df = pd.read_csv(OUTPUT_PATH)
    df = df.dropna(subset=['Kalimat Dinormalisasi', 'Kalimat Bert'])
    df = df[(df['Kalimat Dinormalisasi'].str.strip() != '') &
            (df['Kalimat Bert'].str.strip() != '')].reset_index(drop=True)

    X_ml   = df['Kalimat Dinormalisasi']
    X_bert = df['Kalimat Bert']
    y      = df['Level Kata Kasar']

    (X_train_ml, X_test_ml,
     X_train_bert, X_test_bert,
     y_train, y_test) = train_test_split(
        X_ml, X_bert, y,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    X_train_ml, X_train_bert, y_train = random_oversample_bert(
        X_train_ml, X_train_bert, y_train, random_state=RANDOM_STATE
    )
    print(f'Train: {len(y_train):,} (after oversampling) | Test: {len(y_test):,}')
    return X_train_bert, X_test_bert, y_train, y_test


def build_loaders(X_train, X_test, y_train, y_test, tokenizer):
    train_ds = ProfanityDataset(X_train, y_train.reset_index(drop=True), tokenizer)
    test_ds  = ProfanityDataset(X_test,  y_test.reset_index(drop=True),  tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader


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


def evaluate(model, loader):
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


def fine_tune():
    already_tuned = os.path.exists(BERT_SAVE_PATH) and not FORCE_RETRAIN
    load_path = BERT_SAVE_PATH if already_tuned else INDOBERT_MODEL

    print(f'Loading BERT from: {load_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        BERT_SAVE_PATH if os.path.exists(BERT_SAVE_PATH) else INDOBERT_MODEL
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        load_path, num_labels=4, ignore_mismatched_sizes=True,
    ).to(device)

    X_train, X_test, y_train, y_test = load_splits()
    train_loader, test_loader = build_loaders(X_train, X_test, y_train, y_test, tokenizer)

    if already_tuned:
        print('Fine-tuned weights already loaded. Set FORCE_RETRAIN=True to retrain.')
    else:
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, total_steps // 10),
            num_training_steps=total_steps,
        )

        import matplotlib.pyplot as plt
        import pandas as pd

        print(f'Fine-tuning IndoBERT — {EPOCHS} epoch(s) on {device}')
        history = []
        for epoch in range(1, EPOCHS + 1):
            loss = train_one_epoch(model, train_loader, optimizer, scheduler)
            y_pred_ep, y_true_ep = evaluate(model, test_loader)
            ep_f1 = f1_score(y_true_ep, y_pred_ep, average='macro')
            history.append({'epoch': epoch, 'loss': loss, 'f1': ep_f1})
            print(f'Epoch {epoch}/{EPOCHS}  loss={loss:.4f}  F1={ep_f1:.4f}')

        hist_df = pd.DataFrame(history)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(hist_df['epoch'], hist_df['loss'], marker='o', color='#e74c3c')
        ax1.set_title('IndoBERT Training Loss'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax2.plot(hist_df['epoch'], hist_df['f1'],  marker='o', color='#3498db')
        ax2.set_title('IndoBERT F1 per Epoch');  ax2.set_xlabel('Epoch'); ax2.set_ylabel('F1 (macro)')
        plt.tight_layout()
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUTS_DIR, 'bert_training_history.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved → {out_path}')

        os.makedirs(BERT_SAVE_PATH, exist_ok=True)
        model.save_pretrained(BERT_SAVE_PATH)
        tokenizer.save_pretrained(BERT_SAVE_PATH)
        print(f'Saved → {BERT_SAVE_PATH}')

    y_pred, y_true = evaluate(model, test_loader)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f'\nIndoBERT F1 (macro): {f1:.4f}')
    print(classification_report(y_true, y_pred,
          target_names=['Level 0', 'Level 1', 'Level 2', 'Level 3']))


if __name__ == '__main__':
    fine_tune()
