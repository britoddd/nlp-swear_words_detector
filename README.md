# 🔍 Indonesian Swear-Word & Hate-Speech Detector

Classifies Indonesian tweets into four levels of abusiveness (clean → strongly
abusive) with a fine-tuned IndoBERTweet and three TF-IDF baselines, trained on
13,169 labelled tweets from Ibrohim & Budi (2019).

**[Live demo](https://huggingface.co/spaces/britod/swear-words-detector) · [Dataset](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection) · [Demo video](https://drive.google.com/drive/folders/17NRBhcIx0w8HfKoxa9bIJOY93ZJ0E-SC?usp=sharing)**

---

## Results

| Model | Macro-F1 | Weighted-F1 | Accuracy |
|---|:---:|:---:|:---:|
| **Logistic Regression** | **0.74** | 0.79 | 0.78 |
| SVM (LinearSVC, calibrated) | 0.72 | 0.78 | 0.80 |
| IndoBERTweet (fine-tuned) | 0.71 | 0.76 | 0.76 |
| Naive Bayes | 0.65 | 0.73 | 0.74 |

Macro-F1 is the headline metric because the classes are badly imbalanced — level 3
is 3.6% of the data, so accuracy is dominated by the two majority classes and a
model that never predicts level 3 still scores in the mid-0.70s. On that metric the
char n-gram Logistic Regression **beats the fine-tuned transformer** (0.74 vs 0.71):
with ~13k noisy, heavily obfuscated tweets, character n-grams generalise better than
IndoBERTweet's wordpiece vocabulary, which has never seen `b4b1`. The app serves all
four models and shows a majority vote, so no single model's failure is silent.

What the table hides: every model collapses on **level 2 (moderate hate speech)** —
recall 0.33–0.60 — because the boundary between "weak" and "moderate" in the source
annotation is itself fuzzy. See [Limitations](#limitations).

Evaluated on a held-out stratified 10% test set, **1,317 examples**
(586 / 513 / 171 / 47 per level). Per-class scores:

| Level | LR (P / R / F1) | SVM (P / R / F1) | IndoBERTweet (P / R / F1) | NB (P / R / F1) | Support |
|---|:---:|:---:|:---:|:---:|:---:|
| 0 — Nihil (clean) | 0.88 / 0.85 / **0.87** | 0.85 / 0.89 / **0.87** | 0.83 / 0.88 / 0.85 | 0.82 / 0.82 / 0.82 | 586 |
| 1 — Rendah (mild) | 0.79 / 0.76 / 0.78 | 0.75 / 0.83 / **0.79** | 0.75 / 0.75 / 0.76 | 0.71 / 0.79 / 0.75 | 513 |
| 2 — Sedang (moderate) | 0.52 / 0.60 / **0.56** | 0.65 / 0.40 / 0.49 | 0.65 / 0.40 / 0.49 | 0.52 / 0.33 / 0.41 | 171 |
| 3 — Tinggi (strong) | 0.69 / 0.81 / **0.75** | 0.80 / 0.68 / 0.74 | 0.80 / 0.68 / 0.74 | 0.61 / 0.60 / 0.60 | 47 |

> Macro-F1, weighted-F1 and accuracy are aggregated from the per-class scores above,
> which are reported to two decimals — read them as ±0.01.

---

## What it does

Indonesian social-media profanity is rarely written plainly. Users space letters out
(`a n j i n g`), swap in digits (`b4b1`, `k0nt0l`), and lean on regional slang that no
formal dictionary covers — so a word-level classifier trained on clean text misses
most of what it is supposed to catch. This project takes raw Indonesian text and
returns a severity level rather than a binary flag, so a moderation queue can
prioritise level 3 over level 1 instead of treating every hit the same.

The Streamlit app returns four independent predictions plus a majority vote, the full
preprocessing trace (so you can see *why* a prediction changed), and a
lexicon-censored version of the input.

**Example** — shape of the app's output (labels and confidences are illustrative;
run the [demo](https://huggingface.co/spaces/britod/swear-words-detector) for real ones):

```
input:  dasar b4b1 l0 emang t0l0l bgt

preprocessing
  lowercase   dasar b4b1 l0 emang t0l0l bgt
  leet-fixed  dasar babi lo emang tolol bgt        <- input to IndoBERTweet
  stemmed     dasar babi kamu memang tolol banget  <- input to NB / LR / SVM

predictions (per model, with confidence)
  Naive Bayes           Kasar Ringan   (Mildly Abusive)
  Logistic Regression   Kasar Sedang   (Moderately Abusive)
  SVM (LinearSVC)       Kasar Sedang   (Moderately Abusive)
  IndoBERTweet          Kasar Sedang   (Moderately Abusive)
  --- consensus ----->  Kasar Sedang   (majority vote of 4)

censoring (exact lexicon match on the RAW input, not the cleaned text)
  lexicon hits: none — `b4b1` and `t0l0l` are not literal lexicon entries
```

That last line is not a typo. The classifiers see the de-obfuscated text and catch the
insult; the censor matches the raw string and misses it. Type `babi` plainly and it
returns `lexicon hits: babi` / `censored: dasar b**i …`. Making the censor leet-aware is
a one-line change (feed it `substitute_leet(text)`) that has not been made.

### Levels

| Level | Label (ID) | Label (EN) | Mapped from source labels |
|:-----:|---|---|---|
| 0 | Bersih / Nihil | Clean | `HS=0` and `Abusive=0` |
| 1 | Kasar Ringan / Rendah | Mildly Abusive | `HS_Weak=1` or `Abusive=1` |
| 2 | Kasar Sedang / Menengah | Moderately Abusive | `HS_Moderate=1` |
| 3 | Kasar Berat / Tinggi | Strongly Abusive | `HS_Strong=1` |

---

## Data

| | |
|---|---|
| Source | [Ibrohim & Budi (2019)](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection) — Indonesian Twitter, multi-label hate speech & abusive language |
| Size | 13,169 tweets, 4 derived classes |
| Split | 80 / 10 / 10 train-val-test, stratified, `random_state=42` |
| Class balance | 5,860 (44.5%) · 5,131 (39.0%) · 1,705 (12.9%) · 473 (3.6%) |
| Auxiliary | `abusive.csv` (125-word profanity lexicon) and `new_kamusalay.csv` (15,166-entry slang dictionary), both shipped with the source dataset |
| License | The source dataset carries the original authors' terms — see their repository |
| Known issues | 12× imbalance between level 0 and level 3; the weak/moderate hate-speech boundary is annotator-subjective; tweets are from 2018–2019 Indonesian Twitter, so slang has drifted |

The original dataset is **multi-label** (12 binary columns for target, category and
level). We collapse it into a single ordinal 0–3 severity label
(`pipeline/libs/data_loader.py::map_to_level`), checking the strongest signal first so
a tweet marked both `Abusive` and `HS_Strong` lands at level 3, not level 1. That
collapse is the main modelling decision on the data side, and it is lossy — target and
category information is discarded.

---

## How it works

1. **Normalisation** — lowercase → collapse spaced-out characters (`a n j i n g` →
   `anjing`) → strip intra-word punctuation → leet-speak substitution (`4`→`a`,
   `0`→`o`, `1`→`i`, `@`→`a`) → drop non-alphabetic characters.
   `pipeline/preprocess_pipeline.py`
2. **Branch** — the cleaned text goes straight to IndoBERTweet's own tokeniser; the
   classical branch continues with NLTK tokenisation, *kamusalay* slang
   normalisation, and Sastrawi stemming. `app/preprocessing.py` repeats both branches
   at serving time to keep train/serve parity.
3. **Features (classical)** — `char_wb` TF-IDF, n-grams 2–5, 50k features, `min_df=2`,
   sublinear TF. `pipeline/libs/_train_core.py::build_tfidf_features`
4. **Classical models** — Multinomial NB, Logistic Regression, and a probability-
   calibrated LinearSVC, all with `class_weight='balanced'`, tuned by Optuna over 100
   trials against cross-validated macro-F1.
5. **Transformer** — `indolem/indobertweet-base-uncased` fine-tuned for 4-way
   classification, 10 epochs, `max_len=128`; Optuna (10 trials × 3 epochs, with
   pruning) searches learning rate, weight decay, warmup ratio, and batch size.
6. **Serving** — `app/predictor.py` loads all four artefacts; `app/app.py` renders
   per-model confidence, the preprocessing trace, a majority-vote consensus, and
   lexicon censoring. Censoring (`app/preprocessing.py::censor_text`) is **exact
   matching** against the 125-word `abusive.csv`, applied to the raw input — it is
   independent of the classifiers and, unlike them, is not leet-aware. A fuzzy
   RapidFuzz matcher exists in `pipeline/libs/lexicon.py` but the app does not use it.

**The decision worth defending: character n-grams, not word n-grams.** The obvious
choice for TF-IDF text classification is word unigrams/bigrams. We use `char_wb` 2–5
grams instead because the adversarial surface here is *inside* the word — `k0nt0l`,
`kontooool`, and `k o n t o l` are three different word-level tokens but share nearly
all of their character n-grams. Combined with leet substitution up front, this is why
Logistic Regression stays competitive with a pretrained transformer at this dataset
size. The cost is a much larger, far less interpretable feature space, and no handling
of negation or context.

The resampling variants in `pipeline/alternatives/` (SMOTE, SMOTEENN, random over- and
under-sampling) were run as a comparison; the shipped models use no resampling at all
— `class_weight='balanced'` handled the imbalance at least as well, without
synthesising minority-class text.

---

## Limitations

- **Level 2 (moderate) is the weak point.** The best recall across all four models is
  0.60 (LR); SVM, IndoBERTweet and NB sit at 0.33–0.40. Moderate hate speech is
  routinely absorbed into level 1. If you need reliable moderate-vs-mild separation,
  this model does not give it to you.
- **Macro-F1 of 0.74 rests on 47 test examples for level 3.** That class's scores have
  wide confidence intervals; a handful of flipped predictions moves the number
  materially.
- **No context, no sarcasm, no target.** Every model scores a single utterance in
  isolation. Reclaimed slurs, quoted abuse, and jokes between friends all read as
  abusive.
- **Domain- and era-bound.** Trained on 2018–2019 Indonesian Twitter. Performance on
  formal Indonesian, on other platforms, or on newer slang is untested.
- **The censor is weaker than the classifiers.** It is exact string matching against a
  125-word list on the raw input, so it misses every obfuscated spelling the models
  themselves handle. Treat it as a display feature, not as redaction.
- **Next steps:** an ordinal-aware loss (confusing level 2 with 3 should cost less than
  confusing it with 0), focal loss or per-class threshold tuning for the minority
  classes, and a freshly annotated sample of current slang to measure the drift.

### Ethical use

This is a coursework/research prototype, not a moderation product. The output is a
**severity score for a piece of text**, not a judgement about a person — it says
nothing about intent, and it should not be used to auto-ban, auto-report, or profile
accounts. Given that level-2 recall is as low as 0.33, any deployment that acts on its
output without a human reviewer will both miss real abuse and penalise innocent text.
Do not use it for surveillance, for scoring individuals, or as evidence in any
disciplinary process.

---

## Project structure

```
.
├── app/                        # Streamlit web app (deployed to HF Spaces)
│   ├── app.py                  #   UI entry point
│   ├── predictor.py            #   loads the four models, runs inference
│   └── preprocessing.py        #   serving-time preprocessing (train/serve parity)
├── pipeline/
│   ├── preprocess_pipeline.py  #   normalisation → dataset_processed.csv
│   ├── train_classical.py      #   NB / LR / SVM, Optuna-tuned (shipped models)
│   ├── train_bert.py           #   IndoBERTweet fine-tuning
│   ├── evaluate.py             #   reports + confusion-matrix / comparison plots
│   ├── alternatives/           #   resampling-strategy comparison runs
│   └── libs/                   #   config, data loading/splits, lexicon, shared trainer
├── dataset/                    # Source data, abusive lexicon, kamusalay slang dict
├── logs/                       # Training logs from every run (Optuna trials included)
├── deploy/upload_models.py     # Pushes saved_models/ to the HF model Hub
├── dataset_processed.csv       # Preprocessed dataset (committed)
└── requirements.txt
```

`saved_models/` and `outputs/` are gitignored — see [Reproducibility](#reproducibility).

---

## Setup

```bash
git clone https://github.com/britoddd/indonesian-swear-words-detector.git
cd indonesian-swear-words-detector
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` covers the **app only** (it is the Hugging Face Space's manifest). To
run the training pipeline you also need the tuning, plotting and lexicon packages, which
are deliberately not in it:

```bash
pip install optuna matplotlib seaborn rapidfuzz openpyxl imbalanced-learn
```

Python 3.11 (the deployed Space is pinned to it). The pinned wheels matter — notably
`scikit-learn==1.7.1`, which must match the version that produced the `.pkl` files or
unpickling warns and breaks. A CUDA GPU is strongly recommended for `train_bert.py`:
the 10-trial Optuna search plus the 10-epoch final run takes hours on CPU. The
classical pipeline is CPU-only and finishes in tens of minutes, and the app runs fine
on CPU.

## Usage

```bash
# 1. Preprocess the dataset  -> dataset_processed.csv
python -m pipeline.preprocess_pipeline

# 2. Train
python -m pipeline.train_classical    # -> saved_models/{lr,nb,svm}_*.pkl
python -m pipeline.train_bert         # -> saved_models/indobert_finetuned/

# 3. Evaluate  -> outputs/confusion_matrices.png, outputs/model_comparison.png
python -m pipeline.evaluate

# 4. Run the app
streamlit run app/app.py
```

Gotchas that will bite you in the first five minutes:

- Run the pipeline scripts as **modules** (`python -m pipeline.…`) from the repo root
  — they import `pipeline.libs.config`, which fails if you `cd pipeline` first.
- `FORCE_RETRAIN = True` by default — training overwrites existing artefacts.
- `pipeline/libs/lexicon.py` still expects `paper/indonesian_swear_lexicon.xlsx`, which
  is gitignored and not in the repo. Nothing on the app or training path imports it, so
  this only bites if you call the fuzzy matcher directly.
- On Hugging Face Spaces, set `MODEL_REPO_ID` and the app downloads the 436 MB of
  weights from the Hub instead of the repo (`deploy/upload_models.py` uploads them).

---

## Reproducibility

- **Random seed:** `42` — fixes the stratified train/val/test split, classical model
  initialisation, and the Optuna TPE sampler. BERT fine-tuning still uses
  nondeterministic CUDA kernels, so expect small run-to-run movement in its scores.
- **Environment:** Python 3.11, versions pinned in `requirements.txt`.
- **Trained artefacts:** gitignored (436 MB). Regenerate them with the commands above,
  or pull the deployed copies from the Hugging Face model repo via `MODEL_REPO_ID`.
- **Logs:** every training run's Optuna trials and per-epoch metrics are committed
  under `logs/`, including the five resampling-strategy comparisons.

---

## References

- Ibrohim, M. O. & Budi, I. (2019). *Multi-label Hate Speech and Abusive Language
  Detection in Indonesian Twitter.* Proceedings of the Third Workshop on Abusive
  Language Online (ACL). [W19-3506](https://www.aclweb.org/anthology/W19-3506.pdf)
- Koto, F., Lau, J. H. & Baldwin, T. (2021). *IndoBERTweet: A Pretrained Language Model
  for Indonesian Twitter.* EMNLP.
  [`indolem/indobertweet-base-uncased`](https://huggingface.co/indolem/indobertweet-base-uncased)

## License

**MIT** — see [LICENSE](LICENSE). This covers the code in this repository only.

The data does not carry the same terms: the dataset, the abusive lexicon, and the
*kamusalay* slang dictionary in `dataset/` belong to Ibrohim & Budi and are governed by
the terms in [their repository](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection);
`indolem/indobertweet-base-uncased` carries its own model licence on the Hub. MIT-licensing
this code does not relicense any of them.
