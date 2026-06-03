import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


class TextPreprocessor:
    def __init__(self, kamusalay_path: str, abusive_path: str):
        kamus_df = pd.read_csv(kamusalay_path, header=None, names=["slang", "formal"], encoding="latin-1")
        self.slang_dict = dict(zip(kamus_df["slang"].str.lower(), kamus_df["formal"]))

        abusive_df = pd.read_csv(abusive_path)
        self.abusive_words = set(abusive_df["ABUSIVE"].str.lower().tolist())

        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

    def _lowercase(self, text: str) -> str:
        return text.lower()

    def _clean(self, text: str) -> str:
        text = re.sub(r"https?://\S+|www\.\S+", "url", text)
        text = re.sub(r"@\w+", "user", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\d+", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def _normalize_slang(self, text: str) -> str:
        words = text.split()
        return " ".join(self.slang_dict.get(w, w) for w in words)

    def preprocess(self, text: str, stem: bool = True) -> str:
        text = self._lowercase(text)
        text = self._clean(text)
        text = self._normalize_slang(text)
        if stem:
            text = self.stemmer.stem(text)
        return text

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
        step1 = self._lowercase(text)
        step2 = self._clean(step1)
        step3 = self._normalize_slang(step2)
        step4 = self.stemmer.stem(step3)
        return {
            "original": text,
            "lowercase": step1,
            "cleaned": step2,
            "normalized": step3,
            "stemmed": step4,
        }
