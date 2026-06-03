"""Load the raw dataset, apply label mapping, and return a clean DataFrame."""

import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATA_PATH, ABUSIVE_PATH, SLANG_PATH, RANDOM_STATE


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


def make_splits(X_ml: pd.Series, y: pd.Series, X_bert: pd.Series = None):
    """
    Deterministic 80/10/10 stratified split → (train, val, test).

    First splits 80% train / 20% temp, then splits temp 50/50 into val and test.
    Oversampling must be applied by the caller on the returned train split only.

    Without X_bert → returns (X_train_ml, X_val_ml, X_test_ml, y_train, y_val, y_test)
    With    X_bert → returns same + (X_train_bert, X_val_bert, X_test_bert) inserted after X_test_ml
    """
    if X_bert is not None:
        X_tr_ml, X_tmp_ml, X_tr_bert, X_tmp_bert, y_tr, y_tmp = train_test_split(
            X_ml, X_bert, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
        )
        X_val_ml, X_test_ml, X_val_bert, X_test_bert, y_val, y_test = train_test_split(
            X_tmp_ml, X_tmp_bert, y_tmp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_tmp,
        )
        return (
            X_tr_ml.reset_index(drop=True),   X_val_ml.reset_index(drop=True),   X_test_ml.reset_index(drop=True),
            X_tr_bert.reset_index(drop=True), X_val_bert.reset_index(drop=True), X_test_bert.reset_index(drop=True),
            y_tr.reset_index(drop=True),      y_val.reset_index(drop=True),      y_test.reset_index(drop=True),
        )
    else:
        X_tr_ml, X_tmp_ml, y_tr, y_tmp = train_test_split(
            X_ml, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
        )
        X_val_ml, X_test_ml, y_val, y_test = train_test_split(
            X_tmp_ml, y_tmp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_tmp,
        )
        return (
            X_tr_ml.reset_index(drop=True),  X_val_ml.reset_index(drop=True),  X_test_ml.reset_index(drop=True),
            y_tr.reset_index(drop=True),     y_val.reset_index(drop=True),     y_test.reset_index(drop=True),
        )


if __name__ == '__main__':
    df = load_dataset()
    load_lexicons()
