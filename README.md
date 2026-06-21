---
title: Detektor Kata Kasar Indonesia
emoji: 🔍
colorFrom: red
colorTo: indigo
sdk: streamlit
sdk_version: 1.39.0
python_version: "3.11"
app_file: app/app.py
pinned: false
---

# 🔍 Indonesian Swear-Word & Hate-Speech Detector

A multi-model NLP pipeline that classifies Indonesian text into **four levels of
profanity / abusiveness**, comparing three classical machine-learning models
against a fine-tuned IndoBERTweet transformer. Ships with an interactive
Streamlit web app for live analysis, preprocessing visualisation, and lexicon-based
censoring.

> NLP course project — Group 10.

---

## Levels

| Level | Label (ID)       | Label (EN)           | Mapped from source labels         |
|:-----:|------------------|----------------------|-----------------------------------|
| 0     | Bersih           | Clean                | not hate speech and not abusive   |
| 1     | Kasar Ringan     | Mildly Abusive       | weak hate speech or abusive       |
| 2     | Kasar Sedang     | Moderately Abusive   | moderate hate speech              |
| 3     | Kasar Berat      | Strongly Abusive     | strong hate speech                |

## Models

| Model                 | Features                                | Notes                                       |
|-----------------------|-----------------------------------------|---------------------------------------------|
| Naive Bayes           | TF-IDF                                   | classical baseline                          |
| Logistic Regression   | TF-IDF                                   | `class_weight='balanced'`                   |
| SVM (LinearSVC)        | TF-IDF (shared with LR)                  | calibrated for probability output           |
| IndoBERTweet          | transformer tokeniser (`max_len=128`)    | fine-tuned `indolem/indobertweet-base-uncased` |

Classical models and BERT are tuned with **Optuna** and trained on a deterministic
stratified 80 / 10 / 10 split. The Streamlit app also reports a **consensus vote**
across all four models.

---

## Project structure

```
.
├── app/                        # Streamlit web app
│   ├── app.py                  #   UI entry point
│   ├── predictor.py            #   loads saved models, runs inference
│   └── preprocessing.py        #   serving-time preprocessing (train/serve parity)
├── pipeline/                   # Training & evaluation pipeline
│   ├── config.py               #   paths, hyperparameters, runtime flags
│   ├── data_loader.py          #   dataset loading, label mapping, splits
│   ├── preprocess_pipeline.py  #   7-step text preprocessing
│   ├── lexicon.py              #   Indonesian swear lexicon + fuzzy matching
│   ├── train_classical.py      #   train NB / LR / SVM (main trainer)
│   ├── train_bert.py           #   fine-tune IndoBERTweet
│   ├── train_*.py              #   resampling-strategy comparison variants
│   ├── _train_core.py          #   shared classical-training logic
│   └── evaluate.py             #   reports + confusion-matrix / comparison plots
├── archive/                    # Source data & lexicons (Ibrohim & Budi, 2019)
├── saved_models/               # Trained artefacts (gitignored)
├── outputs/                    # Generated plots & reports (gitignored)
├── dataset_processed.csv       # Preprocessed dataset
└── requirements.txt
```

---

## Preprocessing

Text is normalised through a multi-step pipeline (`pipeline/preprocess_pipeline.py`)
designed to defeat common obfuscation:

1. Lowercasing
2. Collapsing spaced-out characters (`a n j i n g` → `anjing`)
3. Stripping intra-word punctuation
4. Leet-speak substitution (`4` → `a`, `@` → `a`, `1` → `i`, …)
5. Removing non-alphabetic characters

For **classical models**, output is additionally tokenised, slang-normalised
(via *kamusalay*), and stemmed (Sastrawi). For **BERT**, the cleaned text is passed
straight to the transformer's own tokeniser. This split keeps the serving path
(`app/preprocessing.py`) in parity with training.

---

## Setup

```bash
# 1. Create & activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

> A CUDA-capable GPU is recommended for IndoBERTweet training but not required —
> the pipeline falls back to CPU automatically.

---

## Usage

### 1. Preprocess the dataset

```bash
cd pipeline
python preprocess_pipeline.py     # → dataset_processed.csv
```

### 2. Train the models

```bash
python train_classical.py         # → saved_models/{lr,nb,svm}_*.pkl
python train_bert.py              # → saved_models/indobert_finetuned/
```

### 3. Evaluate

```bash
python evaluate.py                # classification reports + plots in outputs/
```

Produces per-model classification reports, `confusion_matrices.png`, and
`model_comparison.png`.

### 4. Run the web app

```bash
streamlit run app/app.py
```

Enter Indonesian text (or pick an example) to see per-model predictions with
confidence scores, the full preprocessing trace, a model consensus, and
lexicon-based censoring of detected abusive words.

---

## Configuration

Key knobs live in `pipeline/config.py`:

| Setting              | Default | Purpose                                          |
|----------------------|---------|--------------------------------------------------|
| `MAX_LEN`            | 128     | BERT token sequence length                       |
| `BATCH_SIZE`         | 16      | training / inference batch size                  |
| `EPOCHS`             | 10      | final BERT training epochs                       |
| `N_TRIALS`           | 10      | Optuna trials (BERT, resampling comparisons)     |
| `N_TRIALS_CLASSICAL` | 100     | Optuna trials for the main classical trainer     |
| `MAX_SAMPLES`        | `None`  | set to an int for faster smoke tests             |
| `FORCE_RETRAIN`      | `True`  | retrain even if saved artefacts exist            |
| `RANDOM_STATE`       | 42      | reproducibility                                  |

---

## Dataset

Trained on the multi-label Indonesian Twitter hate-speech and abusive-language
dataset from:

> Ibrohim, M. O. & Budi, I. (2019). *Multi-label Hate Speech and Abusive Language
> Detection in Indonesian Twitter.* In Proceedings of the Third Workshop on Abusive
> Language Online (ACL). [W19-3506](https://www.aclweb.org/anthology/W19-3506.pdf)

The accompanying *abusive* lexicon and *kamusalay* slang dictionary (in `archive/`)
are used for slang normalisation and lexicon-based censoring.
