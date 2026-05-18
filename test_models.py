import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictor import ModelPredictor, LEVEL_LABELS
from preprocessing import TextPreprocessor

BASE = os.path.dirname(os.path.abspath(__file__))
pp = TextPreprocessor(
    os.path.join(BASE, "archive", "new_kamusalay.csv"),
    os.path.join(BASE, "archive", "abusive.csv"),
)
pred = ModelPredictor(os.path.join(BASE, "saved_models"))

texts = ["Selamat pagi semua!", "Dasar bego lo!", "Anjing sialan kamu!"]
for t in texts:
    steps = pp.get_preprocessing_steps(t)
    nb_p, nb_prob = pred.predict_nb(steps["stemmed"])
    lr_p, lr_prob = pred.predict_lr(steps["stemmed"])
    bert_p, bert_prob = pred.predict_bert(steps["normalized"])
    print(f"Text: {t!r}")
    print(
        f"  NB={LEVEL_LABELS[nb_p][0]}({nb_prob[nb_p]:.2f})"
        f"  LR={LEVEL_LABELS[lr_p][0]}({lr_prob[lr_p]:.2f})"
        f"  BERT={LEVEL_LABELS[bert_p][0]}({bert_prob[bert_p]:.2f})"
    )
    censored, found = pp.censor_text(t)
    if found:
        print(f"  Censored: {censored}")
print("All OK")
