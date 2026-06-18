"""
App-side text preprocessing.

CRITICAL: the model inputs produced here MUST match what the models were trained
on, or predictions degrade (train/serve skew). Instead of re-implementing the
cleaning steps (which previously diverged — e.g. deleting digits instead of
leet-substituting them, and injecting 'url'/'user' tokens), we delegate to the
canonical pipeline.preprocess used by the training scripts:

    classical (NB/LR/SVM)  ← preprocess(text, for_bert=False)   # == 'Kalimat Dinormalisasi'
    IndoBERTweet           ← preprocess(text, for_bert=True)     # == 'Kalimat Bert'
"""

import os
import re
import sys

import pandas as pd

# Make the training pipeline importable so the app reuses the identical cleaning.
_PIPELINE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"
)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from preprocess_pipeline import (  # noqa: E402
    preprocess,
    lowercase,
    remove_excessive_spaces,
    clean_punctuation,
    substitute_leet,
    tokenize,
    normalize_slang,
)


class TextPreprocessor:
    def __init__(self, kamusalay_path: str, abusive_path: str):
        # Slang normalisation is handled inside pipeline.preprocess (it loads its
        # own kamusalay via config), so kamusalay_path is accepted for API
        # compatibility but not used directly here.
        abusive_df = pd.read_csv(abusive_path)
        self.abusive_words = set(abusive_df.iloc[:, 0].str.lower().dropna().tolist())

    def preprocess_classical(self, text: str) -> str:
        """TF-IDF / classical model input (stemmed, slang-normalised)."""
        return preprocess(text, for_bert=False)

    def preprocess_bert(self, text: str) -> str:
        """IndoBERTweet input (leet-substituted, alpha-only, no stemming)."""
        return preprocess(text, for_bert=True)

    def censor_text(self, text: str) -> tuple[str, list[str]]:
        words = text.split()
        censored_words = []
        found_abusive = []
        for word in words:
            clean = re.sub(r"[^\w]", "", word.lower())
            if clean in self.abusive_words:
                found_abusive.append(clean)
                if len(word) > 2:
                    masked = word[0] + "*" * (len(word) - 2) + word[-1]
                else:
                    masked = "*" * len(word)
                censored_words.append(masked)
            else:
                censored_words.append(word)
        return " ".join(censored_words), found_abusive

    def get_preprocessing_steps(self, text: str) -> dict:
        """Decompose the canonical pipeline for display. The two model-input keys
        ('bert_input', 'classical_input') are taken straight from preprocess() so
        they are guaranteed identical to training, regardless of the intermediate
        display strings."""
        s_lower = lowercase(text)
        s_desp  = remove_excessive_spaces(s_lower)
        s_clean = clean_punctuation(s_desp)
        s_leet  = substitute_leet(s_clean)
        s_alpha = re.sub(r"[^a-z\s]", " ", s_leet)
        s_alpha = re.sub(r"\s+", " ", s_alpha).strip()

        tokens      = [t for t in tokenize(s_alpha) if t.isalpha()]
        s_norm      = " ".join(normalize_slang(tokens))

        bert_input      = self.preprocess_bert(text)       # == s_alpha
        classical_input = self.preprocess_classical(text)  # stemmed final input

        return {
            "original":        text,
            "lowercase":       s_lower,
            "cleaned":         s_alpha,
            "normalized":      s_norm,
            "stemmed":         classical_input,
            "bert_input":      bert_input,
            "classical_input": classical_input,
        }
