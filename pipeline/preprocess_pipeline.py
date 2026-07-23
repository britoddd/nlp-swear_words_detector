"""
Preprocessing pipeline (7 steps) for the NLP swear-word detector.

Run as a script to apply preprocessing to the full dataset and save
dataset_processed.csv:

    python preprocess_pipeline.py
"""

import os
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from tqdm.auto import tqdm

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from pipeline.libs.config import OUTPUT_PATH, FORCE_RETRAIN
from pipeline.libs.data_loader import load_dataset, load_lexicons

# ── Module-level singletons (initialised once on import) ───────
_stemmer_factory = StemmerFactory()
_stemmer = _stemmer_factory.create_stemmer()

_stopword_factory = StopWordRemoverFactory()
stopword_list = set(_stopword_factory.get_stop_words())

LEET_MAP = {
    '0': 'o', '1': 'i', '2': 'z', '3': 'e', '4': 'a',
    '5': 's', '6': 'g', '7': 't', '8': 'b', '9': 'g',
    '@': 'a', '!': 'i', '$': 's',
}

# Populated by _ensure_slang_dict() on first use
_slang_dict: dict = {}


def _ensure_slang_dict():
    global _slang_dict
    if not _slang_dict:
        _, _slang_dict = load_lexicons()


# ── Step functions ─────────────────────────────────────────────

def detect_disguise(text: str) -> int:
    """Step 1: Flag text that contains leet-speak characters (returns 0/1)."""
    return 1 if re.search(r'[0-9@!$.|]', str(text)) else 0


def lowercase(text: str) -> str:
    return str(text).lower()


def remove_excessive_spaces(text: str) -> str:
    """Collapse spaced-out characters: 'a n j i n g' → 'anjing'."""
    def join_match(m):
        return m.group(0).replace(' ', '')
    return re.sub(r'(?<!\w)(\w\s){2,}\w(?!\w)', join_match, text)


def clean_punctuation(text: str) -> str:
    return re.sub(r'(?<=[a-z])[.\-_\';"“”](?=[a-z])', '', text)


def substitute_leet(text: str) -> str:
    return ''.join(LEET_MAP.get(c, c) for c in text)


def normalize_slang(tokens: list) -> list:
    _ensure_slang_dict()
    normalized = []
    for token in tokens:
        formal = _slang_dict.get(token, token)
        normalized.extend(formal.split())
    return normalized


def tokenize(text: str) -> list:
    return word_tokenize(text)


def stem_tokens(tokens: list) -> list:
    return [_stemmer.stem(t) for t in tokens]


def preprocess(text: str, for_bert: bool = False) -> str:
    """
    Full 7-step pipeline.
    for_bert=True  → skip tokenisation & stemming (BERT uses its own tokeniser).
    for_bert=False → returns stemmed, space-joined token string for TF-IDF.
    """
    text = lowercase(text)
    text = remove_excessive_spaces(text)
    text = clean_punctuation(text)
    text = substitute_leet(text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    if for_bert:
        return text

    tokens = tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = normalize_slang(tokens)
    tokens = stem_tokens(tokens)
    return ' '.join(tokens)


# ── Dataset-level application ──────────────────────────────────

def apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Add Kalimat Asli, Kalimat Dinormalisasi, and Disamarkan columns.

    'Kalimat Bert' is intentionally NOT stored — it is regenerated on demand by
    the trainers/evaluator (see their load_splits) directly from 'Kalimat Asli'.
    """
    df = df.copy()
    df['Kalimat Asli'] = df['Tweet'].astype(str)
    tqdm.pandas(desc='Preprocessing for ML')
    df['Kalimat Dinormalisasi'] = df['Kalimat Asli'].progress_apply(
        lambda x: preprocess(x, for_bert=False)
    )
    df['Disamarkan'] = df['Kalimat Asli'].apply(detect_disguise)
    return df


if __name__ == '__main__':
    df_raw = load_dataset()

    if not FORCE_RETRAIN and os.path.exists(OUTPUT_PATH):
        print(f'Preprocessed dataset already exists at {OUTPUT_PATH}')
        print('Set FORCE_RETRAIN=True in config.py to recompute.')
    else:
        print('Running preprocessing pipeline...')
        df_raw = apply_preprocessing(df_raw)
        out = df_raw[['Kalimat Asli', 'Kalimat Dinormalisasi', 'Level Kata Kasar', 'Disamarkan']]
        out.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
        print(f'Saved → {OUTPUT_PATH}  (shape: {out.shape})')
        disguised_pct = df_raw['Disamarkan'].mean() * 100
        print(f'Disguised sentences: {df_raw["Disamarkan"].sum():,} ({disguised_pct:.1f}%)')
