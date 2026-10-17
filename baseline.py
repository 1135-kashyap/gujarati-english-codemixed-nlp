import re
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from indic_transliteration import sanscript

# =====================================================
# 1. NORMALIZATION
# =====================================================
SLANG = {"bahuu": "bahu", "gud": "good", "u": "you", "thx": "thanks",
         "sars": "saras", "sarass": "saras", "thnx": "thanks",
         "plz": "please", "pls": "please"}


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s/]", " ", text)          # remove punctuation
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)     # saraaas -> saraas
    return " ".join(SLANG.get(w, w) for w in text.split())


# =====================================================
# 2. WORD-LEVEL LANGUAGE IDENTIFICATION (GU / HI / EN / OTHER)
# =====================================================
GU_WORDS = {"che", "hati", "hatu", "hata", "saras", "bahu", "mast", "kharab",
            "chhu", "chho", "chhe", "nathi", "aaje", "kale", "tame", "maja",
            "thayu", "thai", "pan", "ane", "ekdum", "kem", "majama", "joiye",
            "karyu", "aavyo", "aavi", "gayo", "gayu", "thase", "hu", "tu",
            "mane", "ghare", "saru", "sundar", "vadhare", "ochi", "kai"}

HI_WORDS = {"nahi", "accha", "achha", "bohot", "bahut", "bakwas", "kya",
            "hai", "yaar", "bhai", "zabardast", "bekar", "faltu", "bura",
            "bilkul", "mujhe", "tum", "aap", "kaise", "thoda"}

EN_WORDS = {"movie", "good", "boring", "food", "match", "college", "exam",
            "phone", "app", "service", "thanks", "happy", "birthday",
            "result", "traffic", "hotel", "train", "bus", "office", "boss",
            "video", "post", "best", "but", "ending", "acting", "story",
            "music", "songs", "song", "camera", "battery", "delivery",
            "product", "quality", "love", "loved", "nice", "wow", "super",
            "please", "you", "plan", "trip", "team", "bowling", "batting"}


def tag_words(text):
    out = []
    for w in normalize(text).split():
        if w in GU_WORDS:
            tag = "GU"
        elif w in HI_WORDS:
            tag = "HI"
        elif w in EN_WORDS:
            tag = "EN"
        else:
            tag = "OTHER"
        out.append((w, tag))
    return out


# =====================================================
# 3. TRANSLITERATION (Roman -> Gujarati script)
# =====================================================
GU_DICT = {"hati": "હતી", "hatu": "હતું", "hata": "હતા", "che": "છે",
           "chhe": "છે", "chhu": "છું", "chho": "છો", "saras": "સરસ",
           "bahu": "બહુ", "aaje": "આજે", "kale": "કાલે", "mast": "મસ્ત",
           "kharab": "ખરાબ", "nathi": "નથી", "maja": "મજા", "tame": "તમે",
           "pan": "પણ", "ane": "અને", "ekdum": "એકદમ", "kem": "કેમ",
           "thayu": "થયું", "ghare": "ઘરે", "hu": "હું", "tu": "તું",
           "sundar": "સુંદર", "saru": "સારું", "mane": "મને"}


def to_gujarati(text):
    out = []
    for w, tag in tag_words(text):
        if tag == "EN":
            out.append(w)                      # keep English words as-is
        elif w in GU_DICT:
            out.append(GU_DICT[w])
        else:
            out.append(sanscript.transliterate(
                w, sanscript.ITRANS, sanscript.GUJARATI))
    return " ".join(out)


# =====================================================
# 4. TRAINING AND EVALUATION
# =====================================================
if __name__ == "__main__":
    df = pd.read_csv("gujarati_english_codemixed_dataset.csv")
    print(df["sentiment"].value_counts())
    df["clean"] = df["text"].apply(normalize)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["sentiment"], test_size=0.2,
        random_state=42, stratify=df["sentiment"])

    # ---------- Model 1: TF-IDF + Logistic Regression ----------
    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5)),
        LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("\n=== TF-IDF + Logistic Regression ===")
    print(classification_report(y_test, pred, zero_division=0))
    print(confusion_matrix(y_test, pred))

    # ---------- Demo of the pipeline ----------
    sample = "Aaje movie bahu saras hati but ending boring"
    print("\n=== PIPELINE DEMO ===")
    print("Input      :", sample)
    print("Normalized :", normalize(sample))
    print("Lang tags  :", tag_words(sample))
    print("Gujarati   :", to_gujarati(sample))
    print("Sentiment  :", model.predict([normalize(sample)])[0])

    # ---------- Model 2: Multilingual embeddings + LR ----------
    from sentence_transformers import SentenceTransformer

    print("\nLoading multilingual embedding model...")
    emb_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    X_emb_train = emb_model.encode(X_train.tolist())
    X_emb_test = emb_model.encode(X_test.tolist())

    emb_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    emb_clf.fit(X_emb_train, y_train)
    emb_pred = emb_clf.predict(X_emb_test)

    print("\n=== Multilingual Embeddings + Logistic Regression ===")
    print(classification_report(y_test, emb_pred, zero_division=0))
    print(confusion_matrix(y_test, emb_pred))

    # ---------- Comparison on the test split ----------
    print("\n=== MODEL COMPARISON (single 80/20 split) ===")
    print("TF-IDF + LR     | acc:", round(accuracy_score(y_test, pred), 2),
          "| macro F1:", round(f1_score(y_test, pred, average="macro"), 2))
    print("Embeddings + LR | acc:", round(accuracy_score(y_test, emb_pred), 2),
          "| macro F1:", round(f1_score(y_test, emb_pred, average="macro"), 2))

    # ---------- 5-fold cross-validation (more reliable) ----------
    print("\n=== 5-FOLD CROSS-VALIDATION (macro F1) ===")
    cv_tfidf = cross_val_score(model, df["clean"], df["sentiment"],
                               cv=5, scoring="f1_macro")
    emb_all = emb_model.encode(df["clean"].tolist())
    cv_emb = cross_val_score(
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        emb_all, df["sentiment"], cv=5, scoring="f1_macro")
    print("TF-IDF + LR     :", round(cv_tfidf.mean(), 3), "+/-", round(cv_tfidf.std(), 3))
    print("Embeddings + LR :", round(cv_emb.mean(), 3), "+/-", round(cv_emb.std(), 3))