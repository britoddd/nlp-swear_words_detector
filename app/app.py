import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictor import LEVEL_LABELS, ModelPredictor
from preprocessing import TextPreprocessor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

st.set_page_config(
    page_title="Detektor Kata Kasar",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .label-box {
        text-align: center; padding: 18px; border-radius: 12px;
        margin-bottom: 12px; font-family: sans-serif;
    }
    .label-box h2 { margin: 0 0 4px 0; font-size: 1.5rem; }
    .label-box p  { margin: 0; font-size: 0.9rem; opacity: 0.85; }
    .model-header { text-align: center; font-size: 1.1rem; font-weight: 700; margin-bottom: 8px; }
    .step-row { display: flex; gap: 10px; align-items: flex-start; margin-bottom: 6px; }
    .step-badge {
        background: #4f6ef7; color: white; border-radius: 6px;
        padding: 2px 8px; font-size: 0.75rem; white-space: nowrap; margin-top: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Memuat model...")
def load_resources():
    preprocessor = TextPreprocessor(
        kamusalay_path=os.path.join(ARCHIVE_DIR, "new_kamusalay.csv"),
        abusive_path=os.path.join(ARCHIVE_DIR, "abusive.csv"),
    )
    predictor = ModelPredictor(MODELS_DIR)
    return preprocessor, predictor


def render_label_box(pred: int, proba: list[float], labels: dict):
    label_id, label_en, color = labels.get(pred, (str(pred), str(pred), "#6c757d"))
    st.markdown(
        f"""<div class="label-box" style="background:{color}18; border:2px solid {color};">
        <h2 style="color:{color};">{label_id}</h2>
        <p>{label_en}</p></div>""",
        unsafe_allow_html=True,
    )
    st.markdown("**Probabilitas / Confidence:**")
    for i, p in enumerate(proba):
        lbl, _, c = labels.get(i, (str(i), str(i), "#6c757d"))
        st.progress(float(p), text=f"{lbl}: {p:.1%}")


# ── Header ──────────────────────────────────────────────────────────────────
st.title("🔍 Detektor Kata Kasar & Ujaran Kebencian")
st.markdown(
    "**Indonesian Hate Speech & Abusive Language Detector** — "
    "menggunakan Naive Bayes, Logistic Regression, SVM, dan IndoBERTweet"
)
st.markdown("---")

# ── Load models ─────────────────────────────────────────────────────────────
try:
    preprocessor, predictor = load_resources()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.info("Jalankan `python train_classical.py` terlebih dahulu untuk melatih model klasikal.")
    st.stop()

# ── Input ────────────────────────────────────────────────────────────────────
st.subheader("📝 Masukkan Teks (Enter Indonesian Text)")

examples = [
    "Selamat pagi, semoga harimu menyenangkan!",
    "Dasar bego lo, ga bisa ngapa-ngapain!",
    "Anjing lu, gue benci banget sama elo!",
    "Kaum kafir memang harus diusir dari negeri ini.",
]

selected_example = st.selectbox("Atau pilih contoh teks:", ["— pilih contoh —"] + examples)
default_text = "" if selected_example == "— pilih contoh —" else selected_example

text_input = st.text_area(
    "Teks:",
    value=default_text,
    placeholder="Ketik atau tempel teks berbahasa Indonesia di sini...",
    height=140,
    label_visibility="collapsed",
)

col_btn, col_clear = st.columns([1, 5])
with col_btn:
    analyze = st.button("🔎 Analisis", type="primary", use_container_width=True)

# ── Analysis ─────────────────────────────────────────────────────────────────
if analyze:
    if not text_input.strip():
        st.warning("Masukkan teks terlebih dahulu.")
        st.stop()

    with st.spinner("Menganalisis..."):
        steps = preprocessor.get_preprocessing_steps(text_input)
        preprocessed_stemmed = steps["stemmed"]
        preprocessed_plain = steps["normalized"]
        censored_text, found_abusive = preprocessor.censor_text(text_input)

        nb_pred,   nb_proba   = predictor.predict_nb(preprocessed_stemmed)
        lr_pred,   lr_proba   = predictor.predict_lr(preprocessed_stemmed)
        svm_pred,  svm_proba  = predictor.predict_svm(preprocessed_stemmed)
        bert_pred, bert_proba = predictor.predict_bert(preprocessed_plain)

    st.markdown("---")

    # ── Preprocessing ────────────────────────────────────────────────────────
    st.subheader("⚙️ Preprocessing")
    with st.expander("Lihat tahap preprocessing teks", expanded=False):
        rows = [
            ("Teks Asli", steps["original"]),
            ("Lowercase", steps["lowercase"]),
            ("Bersih (URL/mention/tanda baca)", steps["cleaned"]),
            ("Normalisasi Slang (kamusalay)", steps["normalized"]),
            ("Stemming (Sastrawi)", steps["stemmed"]),
        ]
        for label, val in rows:
            st.markdown(
                f'<div class="step-row"><span class="step-badge">{label}</span><span>{val}</span></div>',
                unsafe_allow_html=True,
            )

    # ── Model Predictions ───────────────────────────────────────────────────
    st.subheader("🤖 Hasil Prediksi Model")
    col_nb, col_lr, col_svm, col_bert = st.columns(4)

    with col_nb:
        st.markdown('<p class="model-header">Naive Bayes</p>', unsafe_allow_html=True)
        render_label_box(nb_pred, nb_proba, LEVEL_LABELS)

    with col_lr:
        st.markdown('<p class="model-header">Logistic Regression</p>', unsafe_allow_html=True)
        render_label_box(lr_pred, lr_proba, LEVEL_LABELS)

    with col_svm:
        st.markdown('<p class="model-header">SVM (LinearSVC)</p>', unsafe_allow_html=True)
        render_label_box(svm_pred, svm_proba, LEVEL_LABELS)

    with col_bert:
        st.markdown('<p class="model-header">IndoBERTweet (Fine-tuned)</p>', unsafe_allow_html=True)
        render_label_box(bert_pred, bert_proba, LEVEL_LABELS)

    # ── Consensus ───────────────────────────────────────────────────────────
    st.markdown("---")
    votes = [nb_pred, lr_pred, svm_pred, bert_pred]
    consensus = max(set(votes), key=votes.count)
    c_label, c_en, c_color = LEVEL_LABELS.get(consensus, (str(consensus), str(consensus), "#6c757d"))

    st.subheader("📊 Konsensus Model")
    st.markdown(
        f"""<div class="label-box" style="background:{c_color}18; border:2px solid {c_color}; max-width:400px; margin:auto;">
        <h2 style="color:{c_color};">{c_label}</h2>
        <p>{c_en} — voting dari 4 model</p></div>""",
        unsafe_allow_html=True,
    )

    # ── Censoring ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔒 Sensor Kata Kasar")
    if found_abusive:
        st.warning(
            f"Kata kasar ditemukan dalam leksikon: **{', '.join(set(found_abusive))}**"
        )
        st.markdown(f"**Teks tersensor:** {censored_text}")
    else:
        st.success("Tidak ada kata kasar yang cocok dengan leksikon abusive.")

    # ── Label Legend ────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("ℹ️ Keterangan Label"):
        for lvl, (id_, en, col) in LEVEL_LABELS.items():
            st.markdown(
                f"- **Level {lvl} — {id_}** ({en})"
            )
        st.markdown(
            "_Model klasikal (NB, LR, SVM) dan IndoBERTweet dilatih pada dataset multi-label "
            "hate speech dan abusive language Twitter berbahasa Indonesia "
            "([Ibrohim & Budi, 2019](https://www.aclweb.org/anthology/W19-3506.pdf))._"
        )
