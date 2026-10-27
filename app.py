"""
Fake News Detector — Streamlit UI
Loads the models trained in the Colab notebook (BERT, LSTM, GRU) and classifies news text.

Expected folder layout (default: ./saved_models):
    saved_models/
        bert_model/            <- BERT weights + tokenizer (save_pretrained output)
        lstm_model.keras
        gru_model.keras
        keras_tokenizer.pkl
Any model that is missing is simply hidden from the sidebar.
"""
import os
import re
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MAX_LEN = 300          # must match the LSTM/GRU training notebook
BERT_MAX_LEN = 128     # must match the BERT training notebook
DEFAULT_MODEL_DIR = os.environ.get("MODEL_DIR", "saved_models")

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.markdown(
    """
    <style>
    .verdict-box {padding: 1.2rem 1.5rem; border-radius: 14px; margin-bottom: 0.5rem;}
    .verdict-fake {background: #fdecea; border-left: 8px solid #d93025;}
    .verdict-real {background: #e6f4ea; border-left: 8px solid #1e8e3e;}
    .verdict-unsure {background: #fef7e0; border-left: 8px solid #f9ab00;}
    .verdict-title {font-size: 1.8rem; font-weight: 700; margin: 0; color: #202124;}
    .verdict-sub {font-size: 1rem; margin: 0.2rem 0 0 0; color: #3c4043;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Text cleaning (identical to the notebook, used by LSTM/GRU only)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_stopwords():
    import nltk
    for pkg in ("stopwords", "punkt", "punkt_tab"):
        nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords
    return set(stopwords.words("english"))


def clean_text(text: str) -> str:
    from nltk.tokenize import word_tokenize
    stop_words = get_stopwords()
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in word_tokenize(text) if t not in stop_words and len(t) > 2]
    return " ".join(tokens)


# ----------------------------------------------------------------------------
# Model discovery & loading
# ----------------------------------------------------------------------------
def discover_models(model_dir: str) -> dict:
    d = Path(model_dir)
    found = {}
    if (d / "bert_model" / "config.json").exists():
        found["BERT"] = d / "bert_model"
    tok_ok = (d / "keras_tokenizer.pkl").exists()
    if tok_ok and (d / "lstm_model.keras").exists():
        found["LSTM"] = d / "lstm_model.keras"
    if tok_ok and (d / "gru_model.keras").exists():
        found["GRU"] = d / "gru_model.keras"
    return found


@st.cache_resource(show_spinner="Loading BERT model…")
def load_bert(path: str):
    import torch
    from transformers import BertTokenizerFast, BertForSequenceClassification
    tokenizer = BertTokenizerFast.from_pretrained(path)
    model = BertForSequenceClassification.from_pretrained(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return tokenizer, model, device


@st.cache_resource(show_spinner="Loading Keras model…")
def load_keras(model_path: str, tokenizer_path: str):
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import Tokenizer  # noqa: F401 (needed to unpickle)
    model = load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


# ----------------------------------------------------------------------------
# Prediction — every function returns P(fake) in [0, 1]
# ----------------------------------------------------------------------------
def predict_bert(text: str, path) -> float:
    import torch
    tokenizer, model, device = load_bert(str(path))
    enc = tokenizer(text, truncation=True, padding="max_length",
                    max_length=BERT_MAX_LEN, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    return float(torch.softmax(logits, dim=1)[0, 1].item())


def predict_keras(text: str, model_path, tokenizer_path) -> float:
    model, tokenizer = load_keras(str(model_path), str(tokenizer_path))
    seq = tokenizer.texts_to_sequences([clean_text(text)])[0][:MAX_LEN]
    padded = np.zeros((1, MAX_LEN), dtype="int32")      # post-padding, like training
    padded[0, : len(seq)] = seq
    return float(model.predict(padded, verbose=0).ravel()[0])


def run_model(name: str, text: str, models: dict, model_dir: str) -> float:
    if name == "BERT":
        return predict_bert(text, models["BERT"])
    return predict_keras(text, models[name], Path(model_dir) / "keras_tokenizer.pkl")


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
EXAMPLES = {
    "Sensational claim": (
        "SHOCKING: Secret documents LEAKED prove elites are hiding the truth from you! "
        "Mainstream media refuses to report this bombshell. Share before it gets deleted!"
    ),
    "Neutral report": (
        "The Senate voted 62 to 38 on Tuesday to approve the infrastructure funding bill, "
        "sending the measure to the House, where a vote is expected next week according to "
        "committee officials."
    ),
}


def set_example(key: str):
    st.session_state["article_text"] = EXAMPLES[key]


def verdict_box(prob_fake: float):
    label = "FAKE" if prob_fake >= 0.5 else "REAL"
    confidence = prob_fake if label == "FAKE" else 1 - prob_fake
    if confidence < 0.65:
        css, icon, note = "verdict-unsure", "⚠️", "Low confidence — treat this result with caution."
    elif label == "FAKE":
        css, icon, note = "verdict-fake", "🚩", "The writing patterns resemble articles labelled fake."
    else:
        css, icon, note = "verdict-real", "✅", "The writing patterns resemble articles labelled real."
    st.markdown(
        f"""<div class="verdict-box {css}">
              <p class="verdict-title">{icon} {label} &nbsp;·&nbsp; {confidence:.1%} confidence</p>
              <p class="verdict-sub">{note}</p></div>""",
        unsafe_allow_html=True,
    )
    return label, confidence


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
model_dir = st.sidebar.text_input("Saved models folder", value=DEFAULT_MODEL_DIR)
models = discover_models(model_dir)

if models:
    options = list(models.keys())
    if len(options) > 1:
        options.append("Compare all")
    choice = st.sidebar.radio("Model", options)
    st.sidebar.caption("Available: " + ", ".join(models.keys()))
else:
    choice = None

st.sidebar.divider()
st.sidebar.markdown(
    "**About**\n\nDeep-learning classifier (BERT, LSTM, GRU) trained to tell real "
    "news articles from fake ones using text patterns."
)

# ----------------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------------
st.title("📰 Fake News Detector")
st.caption("Paste a news headline or article and let the deep-learning models analyse it.")

if not models:
    st.error(
        f"No trained models found in `{model_dir}`. Copy the `saved_models` folder created by the "
        "Colab notebook next to `app.py` (or change the folder in the sidebar)."
    )
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

st.write("**Try an example:**")
ex_cols = st.columns(len(EXAMPLES) + 2)
for col, key in zip(ex_cols, EXAMPLES):
    col.button(key, on_click=set_example, args=(key,), use_container_width=True)

text = st.text_area(
    "Article text",
    key="article_text",
    height=220,
    placeholder="Paste the headline and body of a news article here…",
)

n_words = len(text.split())
st.caption(f"{n_words} words")

if st.button("🔍 Analyze", type="primary", disabled=n_words == 0):
    if n_words < 20:
        st.warning("Very short input — the models were trained on full articles, so accuracy may be lower.")

    with st.spinner("Analyzing…"):
        if choice == "Compare all":
            rows = []
            for name in models:
                p = run_model(name, text, models, model_dir)
                lbl = "FAKE" if p >= 0.5 else "REAL"
                rows.append({"Model": name, "Prediction": lbl,
                             "Confidence": p if lbl == "FAKE" else 1 - p, "P(fake)": p})
            df = pd.DataFrame(rows)

            votes_fake = (df["Prediction"] == "FAKE").sum()
            st.subheader("Model comparison")
            verdict_box(float(df["P(fake)"].mean()))
            st.caption(f"{votes_fake} of {len(df)} models say FAKE. Headline verdict uses the average probability.")
            st.dataframe(
                df[["Model", "Prediction", "Confidence"]],
                hide_index=True, use_container_width=True,
                column_config={"Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.1f%%")},
            )
            st.session_state.history.append({
                "Time": datetime.now().strftime("%H:%M:%S"), "Model": "Compare all",
                "Prediction": "FAKE" if df["P(fake)"].mean() >= 0.5 else "REAL",
                "P(fake)": round(float(df["P(fake)"].mean()), 3), "Text": text[:80] + "…"})
        else:
            p = run_model(choice, text, models, model_dir)
            label, conf = verdict_box(p)
            c1, c2 = st.columns(2)
            c1.metric("Probability FAKE", f"{p:.1%}")
            c2.metric("Probability REAL", f"{1 - p:.1%}")
            st.progress(p, text="← Real · Fake →")
            st.session_state.history.append({
                "Time": datetime.now().strftime("%H:%M:%S"), "Model": choice,
                "Prediction": label, "P(fake)": round(p, 3), "Text": text[:80] + "…"})

if st.session_state.history:
    with st.expander(f"🕘 History ({len(st.session_state.history)})"):
        st.dataframe(pd.DataFrame(st.session_state.history[::-1]), hide_index=True, use_container_width=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

st.divider()
st.caption(
    "⚠️ This tool detects *writing patterns* learned from a labelled dataset (mostly 2016-era US political news). "
    "It does not verify facts, and it can be wrong on other topics or time periods. "
    "Always cross-check important claims with trusted fact-checking sources."
)
