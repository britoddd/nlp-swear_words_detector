"""Load the raw dataset, apply label mapping, and return a clean DataFrame."""

import pandas as pd
from config import DATA_PATH, ABUSIVE_PATH, SLANG_PATH


def map_to_level(row) -> int:
    """Map Kaggle hate-speech columns to a 0–3 profanity level."""
    if row['HS'] == 0 and row['Abusive'] == 0:
        return 0  # Nihil
    elif row['HS_Strong'] == 1:
        return 3  # Tinggi
    elif row['HS_Moderate'] == 1:
        return 2  # Menengah
    else:
        return 1  # Rendah (HS_Weak or Abusive)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')
    df['Level Kata Kasar'] = df.apply(map_to_level, axis=1)
    print(f'Loaded {len(df):,} rows from {DATA_PATH}')
    print('Level distribution:')
    for lvl, cnt in df['Level Kata Kasar'].value_counts().sort_index().items():
        label = {0: 'Nihil', 1: 'Rendah', 2: 'Menengah', 3: 'Tinggi'}[lvl]
        print(f'  Level {lvl} ({label}): {cnt:,} ({cnt/len(df)*100:.1f}%)')
    return df


def load_lexicons() -> tuple[list, dict]:
    """Return (abusive_word_list, slang_dict)."""
    abusive_words_df = pd.read_csv(ABUSIVE_PATH)
    abusive_word_list = abusive_words_df.iloc[:, 0].str.lower().dropna().tolist()

    slang_df = pd.read_csv(SLANG_PATH, header=None, names=['slang', 'formal'],
                           encoding='ISO-8859-1')
    slang_dict = dict(zip(slang_df['slang'].str.lower(), slang_df['formal'].str.lower()))

    print(f'Abusive lexicon: {len(abusive_word_list)} words')
    print(f'Slang dictionary: {len(slang_dict)} entries')
    return abusive_word_list, slang_dict


if __name__ == '__main__':
    df = load_dataset()
    load_lexicons()
