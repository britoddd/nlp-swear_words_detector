import os
import torch

# Project root is one level above this file (pipeline/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── File paths ─────────────────────────────────────────────────
DATA_PATH      = os.path.join(_ROOT, 'archive', 'data.csv')
ABUSIVE_PATH   = os.path.join(_ROOT, 'archive', 'abusive.csv')
SLANG_PATH     = os.path.join(_ROOT, 'archive', 'new_kamusalay.csv')
LEXICON_PATH   = os.path.join(_ROOT, 'paper', 'indonesian_swear_lexicon.xlsx')
OUTPUT_PATH    = os.path.join(_ROOT, 'dataset_processed.csv')
MODELS_DIR     = os.path.join(_ROOT, 'saved_models')
OUTPUTS_DIR    = os.path.join(_ROOT, 'outputs')

# ── Model identifiers ──────────────────────────────────────────
INDOBERT_MODEL    = 'indolem/indobertweet-base-uncased'
BERT_SAVE_PATH    = os.path.join(_ROOT, 'saved_models', 'indobert_finetuned')
LR_MODEL_PATH     = os.path.join(_ROOT, 'saved_models', 'lr_model.pkl')
LR_TFIDF_PATH     = os.path.join(_ROOT, 'saved_models', 'lr_tfidf.pkl')
NB_MODEL_PATH     = os.path.join(_ROOT, 'saved_models', 'nb_model.pkl')
NB_TFIDF_PATH     = os.path.join(_ROOT, 'saved_models', 'nb_tfidf.pkl')
SVM_MODEL_PATH    = os.path.join(_ROOT, 'saved_models', 'svm_model.pkl')

# ── Training hyperparameters ───────────────────────────────────
MAX_SAMPLES    = None   # set to int (e.g. 3000) for faster testing
MAX_LEN        = 128
BATCH_SIZE     = 16
EPOCHS         = 10    # epochs for final training run after tuning
LEARNING_RATE  = 2e-5  # fallback if tuning is skipped
RANDOM_STATE   = 42
TEST_SIZE      = 0.2

# ── Hyperparameter tuning (Optuna) ─────────────────────────────
N_TRIALS           = 10    # Optuna trials for BERT and the classical resampling-strategy comparisons
N_TRIALS_CLASSICAL = 100   # Optuna trials for the main classical trainer (train_classical.py)
TRIAL_EPOCHS       = 3     # epochs per trial (keep short for speed)

# ── Runtime flags ──────────────────────────────────────────────
# Set True to retrain even when saved artefacts already exist
FORCE_RETRAIN  = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
