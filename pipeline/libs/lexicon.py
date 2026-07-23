"""
Profanity lexicon loader and fuzzy matcher.

The lexicon is read from paper/indonesian_swear_lexicon.xlsx (22 validated
entries across 3 severity levels). Falls back to a hardcoded copy if the
xlsx cannot be parsed.
"""

import zipfile
import re
from rapidfuzz import fuzz, process
from pipeline.libs.config import LEXICON_PATH

# Words with innocent dual meanings — excluded, handled contextually by IndoBERT
AMBIGUOUS_WORDS = {'anjing', 'babi', 'gila', 'edan', 'kampung'}


def _parse_xlsx_lexicon(path: str) -> dict:
    with zipfile.ZipFile(path) as z:
        with z.open('xl/worksheets/sheet1.xml') as f:
            content = f.read().decode('utf-8')
    rows = {}
    for m in re.finditer(r'<row r="(\d+)"[^>]*>(.*?)</row>', content, re.DOTALL):
        row_num = int(m.group(1))
        if row_num < 3:
            continue
        cells_text = re.findall(r'<is><t[^>]*>(.*?)</t></is>', m.group(2))
        if len(cells_text) >= 2:
            word = cells_text[0].strip().lower()
            severity_raw = cells_text[1].strip()
            level = int(severity_raw[0]) if severity_raw and severity_raw[0].isdigit() else None
            if word and level:
                rows[word] = level
    return rows


_HARDCODED_LEXICON = {
    'goblok': 1, 'tolol': 1, 'bego': 1, 'dungu': 1, 'idiot': 1, 'bodoh': 1,
    'kampungan': 1, 'kurang ajar': 1, 'sinting': 1,
    'bangsat': 2, 'bajingan': 2, 'brengsek': 2, 'keparat': 2,
    'sialan': 2, 'kampret': 2, 'celaka': 2, 'asu': 2,
    'jancok': 3, 'jancuk': 3, 'kontol': 3, 'memek': 3,
    'lonte': 3, 'pelacur': 3,
}


def load_lexicon() -> dict:
    """Return {word: severity_level} dict."""
    try:
        lexicon = _parse_xlsx_lexicon(LEXICON_PATH)
        print(f'Lexicon loaded from xlsx: {len(lexicon)} entries')
        return lexicon
    except Exception as e:
        print(f'xlsx load failed ({e}), using hardcoded lexicon')
        print(f'Lexicon loaded (hardcoded): {len(_HARDCODED_LEXICON)} entries')
        return _HARDCODED_LEXICON.copy()


# Module-level singleton
LEXICON: dict = load_lexicon()


def lexicon_match(text: str, threshold: int = 85) -> int:
    """
    Return the highest profanity level (0–3) found in preprocessed text.
    Uses exact matching plus RapidFuzz fuzzy matching for leet-speak variants.
    Multi-word entries are checked as consecutive bigrams.
    """
    if not text or not isinstance(text, str):
        return 0

    tokens = text.split()
    candidates = list(tokens)
    candidates += [tokens[i] + ' ' + tokens[i + 1] for i in range(len(tokens) - 1)]

    max_level = 0
    for candidate in candidates:
        if candidate in AMBIGUOUS_WORDS:
            continue
        if candidate in LEXICON:
            max_level = max(max_level, LEXICON[candidate])
            continue
        result = process.extractOne(candidate, LEXICON.keys(), scorer=fuzz.ratio)
        if result and result[1] >= threshold:
            max_level = max(max_level, LEXICON[result[0]])

    return max_level


if __name__ == '__main__':
    from preprocess_pipeline import preprocess

    test_cases = [
        ('tolol banget sih',        'level 1 exact'),
        ('dasar bangsat kamu',       'level 2 exact'),
        ('d4s4r j4nc0k',            'level 3 leet speak'),
        ('hari ini cerah',           'no profanity'),
        ('anjing saya sakit',        'ambiguous → 0'),
        ('kurang ajar kamu',         'multi-word level 1'),
    ]
    print('Lexicon match test:')
    for text, desc in test_cases:
        pre = preprocess(text, for_bert=False)
        level = lexicon_match(pre)
        print(f'  [{desc}] "{text}" → "{pre}" → Level {level}')
