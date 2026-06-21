<div align="center">

# Identifikasi Tingkat Kata Kasar Berbahasa Indonesia dalam Platform Media Sosial Berbasis Teks
## (Indonesian Swear-Word & Hate-Speech Level Detection in Text-Based Social Media)

**Final Project Report**

---

**Project Title:** Multi-Model NLP Pipeline for Four-Level Indonesian Profanity / Abusiveness Classification

**Team:** Group 10 — Class LA01

| Role | Name | Student ID |
|------|------|------------|
| Member 1 | _[Team Member 1]_ | _[NIM]_ |
| Member 2 | _[Team Member 2]_ | _[NIM]_ |
| Member 3 | _[Team Member 3]_ | _[NIM]_ |
| Member 4 | _[Team Member 4]_ | _[NIM]_ |

**Course Code:** _[Course Code — e.g. COMP6178 Natural Language Processing]_

**Semester:** _[Semester — e.g. Even Semester 2025/2026]_

**Date:** 18 June 2026

</div>

<div style="page-break-after: always;"></div>

---

## Abstract

The proliferation of abusive language and hate speech on Indonesian social media creates a need for automated moderation tools that go beyond a simple toxic / non-toxic decision. This project presents an end-to-end Natural Language Processing (NLP) pipeline that classifies Indonesian text into **four graded levels of profanity / abusiveness**: *Bersih* (Clean), *Kasar Ringan* (Mildly Abusive), *Kasar Sedang* (Moderately Abusive), and *Kasar Berat* (Strongly Abusive). We map the multi-label hate-speech and abusive-language annotations of the Ibrohim & Budi (2019) Indonesian Twitter corpus onto this ordinal four-level scheme, and benchmark three classical machine-learning models (Multinomial Naive Bayes, Logistic Regression, and a linear SVM over TF-IDF features) against a fine-tuned **IndoBERTweet** transformer. A purpose-built preprocessing pipeline is designed to defeat common obfuscation tactics (spaced-out characters, leet-speak, intra-word punctuation) that are prevalent in real abusive text. On a held-out, stratified 10 % test split (1,317 samples), the fine-tuned IndoBERTweet model achieves the best macro-F1 of **0.777** (accuracy 0.83), outperforming the best classical model, Logistic Regression, at **0.736** macro-F1 (accuracy 0.78). We deploy the trained models in an interactive Streamlit web application that reports per-model predictions, confidence scores, a multi-model consensus vote, the full preprocessing trace, and lexicon-based censoring of detected abusive words. We discuss the central challenge of severe class imbalance — particularly for the rare *Kasar Berat* level — and the resampling and class-weighting strategies used to address it.

**Keywords:** hate-speech detection, abusive language, text classification, IndoBERTweet, TF-IDF, Indonesian NLP, class imbalance, Optuna.

---

## 1. Introduction

### 1.1 Background

Social media platforms have become a primary channel of public discourse in Indonesia, but the same openness that enables free expression also facilitates the rapid spread of abusive language and hate speech. Manual moderation does not scale to the volume of user-generated content, and binary "toxic / not toxic" classifiers provide insufficient signal for moderators who must triage content by severity. Indonesian text additionally poses NLP-specific difficulties: heavy use of slang (*bahasa gaul*), code-mixing with regional languages and English, inconsistent spelling, and deliberate obfuscation of profanity (e.g. spacing out letters as `a n j i n g`, or substituting digits for letters as in leet-speak).

### 1.2 Problem Statement

Existing publicly available Indonesian resources frame the task as multi-label binary detection (is it hate speech? is it abusive? is it strong/moderate/weak?). What is missing for practical moderation is a single, **ordinal severity score** that a downstream system can threshold and act upon, together with a deployable inference system that is robust to the obfuscation seen in the wild. The problem this project addresses is therefore: *given a short Indonesian text, automatically assign it one of four ordered profanity/abusiveness levels, robustly and with calibrated confidence, and expose this in a usable application.*

### 1.3 Objectives

1. Derive a consistent four-level ordinal labelling scheme from the multi-label Ibrohim & Budi (2019) corpus.
2. Build a reproducible preprocessing pipeline that normalises Indonesian slang and neutralises common profanity-obfuscation tactics.
3. Train, tune, and rigorously compare three classical ML models and a fine-tuned IndoBERTweet transformer on the same deterministic data split.
4. Address the dataset's severe class imbalance through resampling / class-weighting and evaluate its effect.
5. Deploy the models in an interactive web application with explainable output (preprocessing trace, per-model confidence, consensus, lexicon censoring).

### 1.4 Significance

A graded severity classifier enables severity-aware moderation: low-severity content can be flagged or shadow-handled while high-severity content is escalated immediately, in line with the original authors' motivation of helping authorities *prioritise* which hate speech to address first. By comparing lightweight classical models against a transformer, the project also quantifies the accuracy/cost trade-off relevant to organisations with limited compute, and the obfuscation-aware preprocessing contributes a practically useful component for Indonesian content-moderation systems.

---

## 2. Related Work

**Indonesian hate-speech and abusive-language detection.** The foundational resource for this work is Ibrohim & Budi (2019), who released a ~13,000-tweet Indonesian corpus with multi-label annotations covering whether a tweet is abusive, whether it is hate speech, and the target, category, and *level* (weak / moderate / strong) of hate speech. Their experiments with SVM, Naive Bayes, and Random Forest Decision Tree classifiers — combined with Binary Relevance, Label Power-set, and Classifier Chains transformations — established strong classical baselines and the accompanying *abusive* lexicon and *kamusalay* slang dictionary that we reuse here.

**Classical text classification.** TF-IDF feature extraction paired with linear classifiers (Multinomial Naive Bayes, Logistic Regression, linear SVM) remains a robust and computationally cheap baseline for short-text classification, and is the standard reference point against which neural models are measured.

**Transformer language models for Indonesian.** General-purpose BERT (Devlin et al., 2019) demonstrated the effectiveness of large pre-trained transformers fine-tuned on downstream tasks. For Indonesian, IndoBERT and the social-media-specialised **IndoBERTweet** (Koto et al., 2021) — pre-trained on Indonesian tweets — provide domain-matched contextual embeddings well suited to the noisy, slang-heavy text in this task; we fine-tune `indolem/indobertweet-base-uncased`.

**Class imbalance.** Abusive-content datasets are inherently skewed toward the non-abusive majority. Standard remedies include random over/under-sampling and synthetic minority oversampling (SMOTE; Chawla et al., 2002), as well as cost-sensitive learning via class weights. We empirically compare several of these strategies (Section 6.2).

**Hyperparameter optimisation.** We use Optuna (Akiba et al., 2019), a define-by-run framework with tree-structured Parzen estimator sampling and pruning, to tune both the classical models and the transformer.

---

## 3. Methodology

### 3.1 Dataset

We use the multi-label Indonesian Twitter hate-speech and abusive-language dataset of Ibrohim & Budi (2019), comprising **13,169 tweets**. The original multi-label annotations (`HS`, `Abusive`, `HS_Weak`, `HS_Moderate`, `HS_Strong`) are deterministically mapped onto a single ordinal four-level target:

| Level | Label (ID) | Label (EN) | Mapping rule (source columns) |
|:-----:|------------|------------|-------------------------------|
| 0 | Bersih / Nihil | Clean | `HS = 0` **and** `Abusive = 0` |
| 1 | Kasar Ringan / Rendah | Mildly Abusive | `HS_Weak = 1` **or** `Abusive = 1` |
| 2 | Kasar Sedang / Menengah | Moderately Abusive | `HS_Moderate = 1` |
| 3 | Kasar Berat / Tinggi | Strongly Abusive | `HS_Strong = 1` |

**Class distribution.** The resulting dataset is markedly imbalanced, dominated by the clean and mildly-abusive classes:

| Level | Approx. count | Share |
|:-----:|:-------------:|:-----:|
| 0 — Clean | 5,860 | 44.5 % |
| 1 — Mild | ~5,130 | ~39 % |
| 2 — Moderate | ~1,710 | ~13 % |
| 3 — Strong | ~470 | ~3.6 % |

This imbalance — the *Kasar Berat* class is roughly 12× rarer than *Bersih* — is the dominant modelling challenge (Section 7).

### 3.2 Data Splitting

The data is partitioned with a **deterministic, stratified 80 / 10 / 10 split** into train / validation / test (`RANDOM_STATE = 42`), implemented as a first 80/20 split followed by a 50/50 split of the held-out 20 %. Stratification preserves the four-level proportions in every partition. The resulting sizes are **train = 10,535**, **validation = 1,317**, **test = 1,317**. Any resampling is applied to the *training split only*, after the split, to avoid leaking synthetic or duplicated samples into validation/test.

### 3.3 Preprocessing

Text is normalised by a multi-step pipeline (`pipeline/preprocess_pipeline.py`) explicitly designed to defeat obfuscation:

1. **Lowercasing.**
2. **Collapsing spaced-out characters** — e.g. `a n j i n g` → `anjing`.
3. **Stripping intra-word punctuation** — e.g. `b*ngs*t` artefacts.
4. **Leet-speak substitution** — `4 → a`, `@ → a`, `1 → i`, `0 → o`, `3 → e`, etc.
5. **Removing remaining non-alphabetic characters.**

For the **classical models**, the cleaned text is additionally **tokenised, slang-normalised** via the *kamusalay* dictionary (mapping informal/slang tokens to their formal equivalents), and **stemmed** with the Sastrawi Indonesian stemmer. For **IndoBERTweet**, the cleaned text is passed directly to the transformer's own WordPiece tokeniser (no stemming or slang normalisation), preserving subword information the model was pre-trained on. Crucially, the serving path (`app/preprocessing.py`) reuses the exact same logic, keeping **train/serve parity** — a train/serve skew bug in this path was identified and fixed during development.

### 3.4 Model Architectures

| Model | Features | Key configuration |
|-------|----------|-------------------|
| **Multinomial Naive Bayes** | TF-IDF | Additive smoothing `alpha`, optional TF-IDF norm — both tuned |
| **Logistic Regression** | TF-IDF | `class_weight='balanced'`; regularisation `C` and `solver` tuned |
| **SVM (LinearSVC)** | TF-IDF (shared vectoriser with LR) | `class_weight='balanced'`; `C` tuned; calibrated for probability output |
| **IndoBERTweet** | Transformer WordPiece tokeniser, `max_len = 128` | Fine-tuned `indolem/indobertweet-base-uncased` with a 4-class classification head |

The classical models share TF-IDF representations; LinearSVC is wrapped in probability calibration so that the web app can display confidence scores consistently across all models.

### 3.5 Training Setup

- **Classical models** (`pipeline/train_classical.py`, shared logic in `_train_core.py`): each model's hyperparameters are tuned with **Optuna over 100 trials** (`N_TRIALS_CLASSICAL = 100`), selecting on **cross-validated macro-F1** on the training data; the best configuration is refit and evaluated once on the test set. Class imbalance is handled via `class_weight='balanced'` (final configuration) and was also studied via resampling variants (Section 6.2).
- **IndoBERTweet** (`pipeline/train_bert.py`): tuned with Optuna over **10 trials** (batch size, learning rate, warmup ratio, weight decay), then the best configuration is trained for **10 epochs** with the AdamW optimiser and a linear warmup-then-decay schedule (`MAX_LEN = 128`, `BATCH_SIZE = 16`). Training runs on GPU when available and falls back to CPU automatically.
- **Reproducibility:** fixed `RANDOM_STATE = 42` throughout; `FORCE_RETRAIN` controls whether cached artefacts are reused.

### 3.6 Evaluation Metrics

Because of the heavy class imbalance, **macro-averaged F1** is the primary metric, as it weights all four levels equally and is not dominated by the majority *Clean* class. We additionally report overall accuracy, and per-class precision / recall / F1 via the full classification report and confusion matrices (`pipeline/evaluate.py`).

---

## 4. Implementation & Results

### 4.1 System Details

The system is organised into two parts: a **training/evaluation pipeline** (`pipeline/`) and a **Streamlit web application** (`app/`). The pipeline handles data loading and label mapping (`data_loader.py`), preprocessing (`preprocess_pipeline.py`), the Indonesian swear lexicon with fuzzy matching (`lexicon.py`), model training (`train_classical.py`, `train_bert.py`), and reporting (`evaluate.py`). Trained artefacts are persisted to `saved_models/` (`*.pkl` for classical models and their TF-IDF vectorisers; a Hugging Face model directory for the fine-tuned transformer). The web app (`app/predictor.py`) loads these artefacts and runs live inference. The stack is Python with scikit-learn, PyTorch + Hugging Face Transformers, Sastrawi, Optuna, and Streamlit.

### 4.2 Hyperparameter Tuning

Optuna tuning selected the following best configurations (on cross-validated / validation macro-F1):

- **Logistic Regression:** `C ≈ 4.51`, `solver = lbfgs` (val CV macro-F1 ≈ 0.721).
- **Naive Bayes:** `alpha ≈ 0.055`, `norm = False` (val CV macro-F1 ≈ 0.654).
- **SVM (LinearSVC):** `C ≈ 0.99` (val CV macro-F1 ≈ 0.715).
- **IndoBERTweet (best trial):** `learning_rate ≈ 3.67 × 10⁻⁵`, `batch_size = 16`, `warmup_ratio ≈ 0.020`, `weight_decay ≈ 0.030` (val macro-F1 ≈ 0.731), then trained 10 epochs for the final model.

### 4.3 Final Results (held-out test set, n = 1,317)

| Model | Accuracy | Macro Precision | Macro Recall | **Macro F1** |
|-------|:--------:|:---------------:|:------------:|:------------:|
| Naive Bayes | 0.74 | 0.66 | 0.64 | 0.645 |
| SVM (LinearSVC) | 0.79 | 0.76 | 0.70 | 0.722 |
| Logistic Regression | 0.78 | 0.72 | 0.76 | 0.736 |
| **IndoBERTweet** | **0.83** | **0.78** | **0.77** | **0.777** |

**IndoBERTweet — per-class report (test):**

| Level | Precision | Recall | F1 | Support |
|:-----:|:---------:|:------:|:--:|:-------:|
| 0 — Clean | 0.89 | 0.90 | 0.90 | 586 |
| 1 — Mild | 0.82 | 0.82 | 0.82 | 513 |
| 2 — Moderate | 0.66 | 0.61 | 0.63 | 171 |
| 3 — Strong | 0.78 | 0.74 | 0.76 | 47 |
| **macro avg** | **0.78** | **0.77** | **0.78** | 1,317 |

**Logistic Regression (best classical) — per-class report (test):**

| Level | Precision | Recall | F1 | Support |
|:-----:|:---------:|:------:|:--:|:-------:|
| 0 — Clean | 0.88 | 0.85 | 0.87 | 586 |
| 1 — Mild | 0.79 | 0.76 | 0.78 | 513 |
| 2 — Moderate | 0.52 | 0.60 | 0.56 | 171 |
| 3 — Strong | 0.69 | 0.81 | 0.75 | 47 |
| **macro avg** | **0.72** | **0.76** | **0.74** | 1,317 |

### 4.4 Experiments & Visualizations

The evaluation script generates:

- **`confusion_matrices.png`** — per-model confusion matrices over the four levels.
- **`model_comparison.png`** — side-by-side macro precision/recall/F1 bar chart across all four models.
- **`bert_training_history.png`** — IndoBERTweet training/validation loss and F1 curves across epochs.
- **`outputs/result_metrics.png`** and **`outputs/hparam_tuning.csv`** — aggregated metrics and the full Optuna trial log.

### 4.5 Observations

- **The transformer wins, but the margin is modest.** IndoBERTweet leads LR by ~0.04 macro-F1 and ~0.05 accuracy — a meaningful but not overwhelming gap, given its far greater compute cost.
- **All models struggle most on Level 2 (Moderate).** This is the hardest class for every model (LR F1 0.56, IndoBERT F1 0.63), reflecting both its scarcity and genuine semantic overlap with Levels 1 and 3.
- **Class weighting helps the rarest class.** With `class_weight='balanced'`, LR actually achieves strong recall on Level 3 (Strong, 0.81) despite its scarcity, at the cost of some precision.
- **Classical train scores far exceed test scores** (e.g. LR train macro-F1 ≈ 0.91 vs test 0.74), indicating overfitting of the high-capacity TF-IDF + linear configurations — expected for sparse high-dimensional features.

---

## 5. The Web Application

The Streamlit app (`streamlit run app/app.py`) accepts free-text Indonesian input (or a chosen example) and displays:

- **Per-model predictions** with confidence scores for all four models.
- A **model consensus vote** aggregating the four predictions into a single decision.
- The **full preprocessing trace**, showing each normalisation step applied to the input (useful for transparency and debugging obfuscation handling).
- **Lexicon-based censoring** of detected abusive words, using the Indonesian swear lexicon with fuzzy matching to catch near-miss spellings.

---

## 6. Discussion & Limitations

### 6.1 Performance Analysis

The ranking IndoBERTweet > Logistic Regression > SVM > Naive Bayes is consistent with expectations: the domain-matched transformer captures contextual and subword cues that bag-of-words TF-IDF cannot, which matters most for the ambiguous middle classes. However, the relatively small gap (~4 macro-F1 points over LR) is notable — for deployments where latency, memory, or cost are constrained, the calibrated Logistic Regression model is a strong, much cheaper alternative that retains explainability through inspectable feature weights.

### 6.2 Class Imbalance and Resampling Trade-offs

Imbalance is the project's central difficulty. We implemented and compared several training-set strategies (`train_no_resample.py`, `train_random_oversample.py`, `train_smote.py`, `train_smoteenn_raw.py`, `train_undersample.py`):

- **Random oversampling** balances all four classes to 4,688 each (train = 18,752) and drives train macro-F1 to ~0.99, but this largely reflects memorisation of duplicated minority samples; test macro-F1 did not improve over the simpler approach and in some runs degraded, because duplicated rare examples do not add genuine signal.
- **Class weighting** (`class_weight='balanced'`) on the natural distribution proved the better-behaved choice for the final classical models, yielding strong minority-class recall without the inflated, misleading train scores of oversampling. This was adopted as the final configuration.
- **SMOTE / SMOTEENN / undersampling** were evaluated as comparison variants; synthetic interpolation in sparse TF-IDF space and aggressive undersampling of the majority offered no consistent test-set gain.

The trade-off is fundamentally between **minority recall and majority precision**: balancing techniques raise recall on Levels 2–3 but introduce false positives on Levels 0–1.

### 6.3 Challenges

- **Train/serve skew:** an early mismatch between training-time and serving-time preprocessing was discovered and corrected to ensure the app's predictions match offline evaluation.
- **Label derivation:** collapsing a rich multi-label annotation into a single ordinal level necessarily discards information and forces priority rules (e.g. *strong* dominates *moderate*), which may not match every annotator's intent.
- **Compute cost:** transformer fine-tuning and 100-trial classical tuning are time-consuming; the pipeline supports `MAX_SAMPLES` smoke tests and CPU fallback to mitigate this.

### 6.4 Limitations

1. **Single-corpus, single-platform:** trained only on one Twitter dataset; generalisation to other platforms, longer text, or evolving slang is unverified.
2. **Notebook vs. pipeline drift:** an earlier exploratory notebook reports lower classical scores (e.g. LR macro-F1 ≈ 0.65) from before the 100-trial tuning and preprocessing fixes; the authoritative final figures are those from the tuned pipeline (Section 4.3). The IndoBERTweet model was not re-run after the latest classical fixes, though it shares the same deterministic split.
3. **Ordinal structure not exploited:** all models treat the task as flat multi-class rather than ordinal regression, so a Level-0↔Level-3 confusion is penalised identically to a Level-2↔Level-3 confusion.
4. **No external/temporal test set:** results are in-distribution; robustness to concept drift is not measured.

---

## 7. Conclusion & Future Work

We built and evaluated a complete four-level Indonesian profanity/abusiveness detection system, comparing three TF-IDF classical models against a fine-tuned IndoBERTweet transformer on a deterministic stratified split of the Ibrohim & Budi (2019) corpus, and deployed it in an explainable Streamlit application. **IndoBERTweet achieved the best performance (macro-F1 0.777, accuracy 0.83)**, with Logistic Regression the strongest and far cheaper classical alternative (macro-F1 0.736). Obfuscation-aware preprocessing and class-weighting were key practical components, and the moderate-severity class (Level 2) remained the hardest for all models.

**Future work:**

1. **Ordinal-aware modelling** — treat the four levels as ordered (ordinal regression / CORAL-style heads) so that the loss reflects the cost of distant misclassifications.
2. **Data augmentation for rare classes** — back-translation or LLM-based paraphrasing to generate *genuinely novel* Level-2/3 examples rather than duplicates.
3. **Cross-platform and temporal evaluation** — test on data from other Indonesian platforms and later time periods to measure drift.
4. **Model ensembling** — formally combine the transformer with the calibrated classical models (the app's consensus vote is a first step toward this).
5. **Lexicon expansion and fuzzy-match tuning** — continuously update the swear lexicon and evaluate the censoring component quantitatively.
6. **Explainability** — integrate token-attribution (e.g. SHAP / attention visualisation) into the app for moderator trust.

---

## 8. References

[1] M. O. Ibrohim and I. Budi, "Multi-label Hate Speech and Abusive Language Detection in Indonesian Twitter," in *Proc. Third Workshop on Abusive Language Online (ALW3), ACL*, Florence, Italy, 2019, pp. 46–57, doi: 10.18653/v1/W19-3506.

[2] F. Koto, J. H. Lau, and T. Baldwin, "IndoBERTweet: A Pretrained Language Model for Indonesian Twitter with Effective Domain-Specific Vocabulary Initialization," in *Proc. 2021 Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2021, pp. 10660–10668.

[3] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in *Proc. NAACL-HLT*, 2019, pp. 4171–4186.

[4] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321–357, 2002.

[5] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A Next-generation Hyperparameter Optimization Framework," in *Proc. 25th ACM SIGKDD Int. Conf. Knowledge Discovery & Data Mining*, 2019, pp. 2623–2631.

[6] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[7] T. Wolf et al., "Transformers: State-of-the-Art Natural Language Processing," in *Proc. 2020 EMNLP: System Demonstrations*, 2020, pp. 38–45.

[8] Sastrawi Contributors, "Sastrawi: High-quality Indonesian Stemmer Library," GitHub repository. [Online]. Available: https://github.com/sastrawi/sastrawi

[9] Streamlit Inc., "Streamlit: A faster way to build and share data apps." [Online]. Available: https://streamlit.io

---

## Appendix

### Appendix A — Team Contribution Statement

| Member | Primary Contributions |
|--------|-----------------------|
| _[Team Member 1]_ | _[e.g. Data loading & label mapping, dataset analysis]_ |
| _[Team Member 2]_ | _[e.g. Preprocessing pipeline & lexicon/fuzzy-match module]_ |
| _[Team Member 3]_ | _[e.g. Classical model training, Optuna tuning, resampling experiments]_ |
| _[Team Member 4]_ | _[e.g. IndoBERTweet fine-tuning, evaluation, Streamlit app]_ |

> _All members contributed to report writing and review. Replace the placeholders above with each member's actual tasks and approximate percentage contribution as required by the course._

### Appendix B — Screenshots

The following generated artefacts and app screens should be attached here:

1. **`model_comparison.png`** — macro precision/recall/F1 comparison across the four models.
2. **`confusion_matrices.png`** — per-model confusion matrices.
3. **`bert_training_history.png`** — IndoBERTweet training/validation loss and F1 curves.
4. **Streamlit app** — main prediction screen showing per-model predictions, consensus, preprocessing trace, and lexicon censoring.

### Appendix C — Selected Code Snippets

**C.1 Multi-label → ordinal level mapping (`pipeline/data_loader.py`):**

```python
def map_to_level(row) -> int:
    """Map Kaggle hate-speech columns to a 0–3 profanity level."""
    if row['HS'] == 0 and row['Abusive'] == 0:
        return 0  # Nihil   (Clean)
    elif row['HS_Strong'] == 1:
        return 3  # Tinggi  (Strongly Abusive)
    elif row['HS_Moderate'] == 1:
        return 2  # Menengah (Moderately Abusive)
    else:
        return 1  # Rendah  (HS_Weak or Abusive)
```

**C.2 Deterministic stratified 80/10/10 split (`pipeline/data_loader.py`):**

```python
# First 80% train / 20% temp, then split temp 50/50 into val and test,
# stratifying on the label at every step (RANDOM_STATE = 42).
X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X_ml, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_tmp)
```

**C.3 Key configuration (`pipeline/config.py`):**

```python
INDOBERT_MODEL     = 'indolem/indobertweet-base-uncased'
MAX_LEN            = 128
BATCH_SIZE         = 16
EPOCHS             = 10     # final BERT training epochs
N_TRIALS           = 10     # Optuna trials (BERT, resampling comparisons)
N_TRIALS_CLASSICAL = 100    # Optuna trials (main classical trainer)
RANDOM_STATE       = 42
```

**C.4 Running the pipeline:**

```bash
# 1. Preprocess
python pipeline/preprocess_pipeline.py      # -> dataset_processed.csv
# 2. Train
python pipeline/train_classical.py          # -> saved_models/{lr,nb,svm}_*.pkl
python pipeline/train_bert.py               # -> saved_models/indobert_finetuned/
# 3. Evaluate
python pipeline/evaluate.py                 # -> outputs/*.png + reports
# 4. Serve
streamlit run app/app.py
```

---

<div align="center"><em>End of Report — Group 10 (LA01)</em></div>
