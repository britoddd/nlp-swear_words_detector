import os
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, BertForSequenceClassification

LEVEL_LABELS = {
    0: ("Bersih", "Clean", "#28a745"),
    1: ("Kasar Ringan", "Mildly Abusive", "#ffc107"),
    2: ("Kasar Sedang", "Moderately Abusive", "#fd7e14"),
    3: ("Kasar Berat", "Strongly Abusive", "#dc3545"),
}


class ModelPredictor:
    def __init__(self, models_dir: str):
        # On Hugging Face Spaces the 436 MB of model artifacts live in a Hub
        # model repo (the Space git repo stays tiny). When MODEL_REPO_ID is set
        # we download a local snapshot once and load from it; otherwise we use
        # the local saved_models/ directory (local development).
        repo_id = os.environ.get("MODEL_REPO_ID")
        if repo_id:
            from huggingface_hub import snapshot_download
            models_dir = snapshot_download(repo_id=repo_id)
        self._load_classical(models_dir)
        self._load_bert(os.path.join(models_dir, "indobert_finetuned"))

    def _load_classical(self, models_dir: str):
        with open(os.path.join(models_dir, "lr_model.pkl"), "rb") as f:
            self.lr_model = pickle.load(f)
        with open(os.path.join(models_dir, "lr_tfidf.pkl"), "rb") as f:
            self.lr_tfidf = pickle.load(f)
        with open(os.path.join(models_dir, "nb_model.pkl"), "rb") as f:
            self.nb_model = pickle.load(f)
        with open(os.path.join(models_dir, "nb_tfidf.pkl"), "rb") as f:
            self.nb_tfidf = pickle.load(f)
        with open(os.path.join(models_dir, "svm_model.pkl"), "rb") as f:
            self.svm_model = pickle.load(f)

    def _load_bert(self, bert_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model = BertForSequenceClassification.from_pretrained(bert_path)
        self.bert_model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model.to(self.device)

    def predict_lr(self, text: str) -> tuple[int, list[float]]:
        features = self.lr_tfidf.transform([text])
        pred = int(self.lr_model.predict(features)[0])
        proba = self.lr_model.predict_proba(features)[0].tolist()
        return pred, proba

    def predict_nb(self, text: str) -> tuple[int, list[float]]:
        features = self.nb_tfidf.transform([text])
        pred = int(self.nb_model.predict(features)[0])
        proba = self.nb_model.predict_proba(features)[0].tolist()
        return pred, proba

    def predict_svm(self, text: str) -> tuple[int, list[float]]:
        features = self.lr_tfidf.transform([text])  # SVM shares LR's TF-IDF
        pred = int(self.svm_model.predict(features)[0])
        proba = self.svm_model.predict_proba(features)[0].tolist()
        return pred, proba

    def predict_bert(self, text: str) -> tuple[int, list[float]]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,  # match MAX_LEN used during fine-tuning
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.bert_model(**inputs).logits
        proba = torch.softmax(logits, dim=1).cpu().numpy()[0].tolist()
        pred = int(np.argmax(proba))
        return pred, proba
