import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from baseline import normalize, tag_words, to_gujarati

st.set_page_config(page_title="Gujarati-English Code-Mixed NLP", page_icon="💬")

COLORS = {"GU": "#2e7d32", "HI": "#ef6c00", "EN": "#1565c0", "OTHER": "#757575"}
EMOJI = {"positive": "😊", "negative": "😞", "neutral": "😐", "mixed": "🤔"}


@st.cache_resource
def load_models():
    df = pd.read_csv("gujarati_english_codemixed_dataset.csv")
    df["clean"] = df["text"].apply(normalize)

    tfidf = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5)),
        LogisticRegression(max_iter=1000, class_weight="balanced"))
    tfidf.fit(df["clean"], df["sentiment"])

    from sentence_transformers import SentenceTransformer
    emb_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    emb_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    emb_clf.fit(emb_model.encode(df["clean"].tolist()), df["sentiment"])
    return tfidf, emb_model, emb_clf


st.title("💬 Gujarati–English Code-Mixed Text Understanding")
st.caption("Language ID → Normalization → Transliteration → Sentiment")

with st.spinner("Loading models (first run takes a minute)..."):
    tfidf, emb_model, emb_clf = load_models()

choice = st.radio("Sentiment model",
                  ["Multilingual Embeddings + LR", "TF-IDF + LR"],
                  horizontal=True)

text = st.text_area("Enter a code-mixed message",
                    "Aaje movie bahu saras hati but ending boring", height=100)

if st.button("Analyze") and text.strip():
    clean = normalize(text)

    st.subheader("1. Word-level language tags")
    html = " ".join(
        f"<span style='background:{COLORS[t]};color:white;padding:3px 8px;"
        f"border-radius:6px;margin:2px;display:inline-block'>{w} · {t}</span>"
        for w, t in tag_words(text))
    st.markdown(html, unsafe_allow_html=True)

    st.subheader("2. Normalized text")
    st.code(clean)

    st.subheader("3. Gujarati script (transliteration)")
    st.code(to_gujarati(text))

    st.subheader("4. Predicted sentiment")
    if choice.startswith("Multilingual"):
        vec = emb_model.encode([clean])
        label = emb_clf.predict(vec)[0]
        probs = dict(zip(emb_clf.classes_, emb_clf.predict_proba(vec)[0]))
    else:
        label = tfidf.predict([clean])[0]
        probs = dict(zip(tfidf.classes_, tfidf.predict_proba([clean])[0]))

    st.success(f"{EMOJI.get(label, '')} {label.upper()}")
    st.bar_chart(pd.Series(probs, name="probability"))

st.markdown("---")
st.caption("Legend: GU = Gujarati (Roman) · HI = Hindi · EN = English · OTHER = unknown")