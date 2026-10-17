# Gujarati–English Code-Mixed Text Understanding System

An NLP pipeline for romanized Gujarati–English (and some Hindi) code-mixed text, such as
_"Aaje movie bahu saras hati but ending boring"_.

**Course:** 221601906 – Natural Language Processing (NLP) Practicals (CEC)
**Problem statement:** 4 – Gujarati–English Code-Mixed Text Understanding System
**Author:** <YOUR NAME>, <ENROLLMENT NO.>

---

## What the system does

For any input message it produces:

1. **Word-level language identification:** each word is tagged GU, HI, EN, or OTHER
2. **Text normalization:** lowercasing, punctuation removal, repeated-letter reduction, and slang mapping
3. **Transliteration:** romanized Gujarati words are converted to Gujarati script, and English words are kept as they are
4. **Sentiment classification:** positive, negative, neutral, or mixed, using two models that are compared

## Pipeline

```
Raw message
   │
   ▼
Normalization ──► Word-level language ID (GU / HI / EN / OTHER)
   │                         │
   │                         ▼
   │                Transliteration (Roman → Gujarati script)
   ▼
Sentiment classifier
   ├── Model A: Character TF-IDF (2–5 grams) + Logistic Regression
   └── Model B: Multilingual sentence embeddings + Logistic Regression
```

## Dataset

File: `gujarati_english_codemixed_dataset.csv` (419 messages)

| Column      | Description                                                              |
| ----------- | ------------------------------------------------------------------------ |
| `id`        | Row number                                                               |
| `text`      | Romanized Gujarati–English (some Hindi) message                          |
| `sentiment` | positive / negative / neutral / mixed                                    |
| `topic`     | movie, food, cricket, weather, travel, college, service, social, general |

Class distribution: neutral 112, positive 103, mixed 103, negative 101.

**Source and limitations:** the messages were **written by the author (LLM-assisted) in the style of
real social-media text**. They are _synthetic_ and not scraped from real users. The vocabulary is
limited, so scores on real-world data are expected to be lower. Evaluating on real
messages is listed as future work.

## Results

Evaluation uses an 80/20 stratified split (84 test messages) and 5-fold cross-validation.

| Model                                         | Accuracy (split) | Macro F1 (split) | Macro F1 (5-fold CV) |
| --------------------------------------------- | ---------------- | ---------------- | -------------------- |
| TF-IDF (char n-grams) + Logistic Regression   | 0.88             | 0.88             | **0.888 ± 0.021**    |
| Multilingual embeddings + Logistic Regression | 0.77             | 0.77             | 0.670 ± 0.043        |

**Observations**

- Character n-gram TF-IDF clearly beats the general-purpose multilingual embeddings on romanized
  code-mixed text. A likely reason is that romanized Gujarati has inconsistent spelling that
  character n-grams handle well, while the embedding model was trained mostly on native-script text.
- Positive is the weakest class for TF-IDF (F1 0.79). It is confused with negative and neutral.
- The embedding model misreads messages such as _"Aaje movie bahu saras hati but ending boring"_
  as positive, ignoring the contrast after "but". TF-IDF predicts _mixed_, which is correct.

Screenshots are in the `screenshots/` folder.

## Project structure

```
.
├── baseline.py                                  # pipeline functions + training + evaluation
├── app.py                                       # Streamlit demo
├── gujarati_english_codemixed_dataset.csv       # dataset
├── requirements.txt
├── screenshots/                                 # app and result screenshots
└── README.md
```

## Setup and usage

```bash
# 1. Clone the repository
git clone <YOUR_GITHUB_REPO_LINK>
cd <REPO_FOLDER>

# 2. Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt

# 3. Train and evaluate both models
python baseline.py

# 4. Launch the demo app
python -m streamlit run app.py
```

The first run downloads the multilingual embedding model
(`paraphrase-multilingual-MiniLM-L12-v2`, about 470 MB) from Hugging Face.

## Tools and libraries

Python, pandas, scikit-learn, sentence-transformers, indic-transliteration, Streamlit.

## Limitations and future work

- The dataset is small and synthetic, so a larger set of real social-media messages is needed.
- The language identifier is lexicon-based. A trained word-level classifier would handle unseen words better.
- Transliteration uses a small dictionary plus rule-based conversion, so rare words may be inexact.
- Fine-tuning a transformer such as MuRIL or XLM-RoBERTa on more data could improve results.
- Add intent and topic classification (the dataset already contains a `topic` column).

## References

- Reimers and Gurevych, _Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks_, EMNLP 2019.
- Sentence-Transformers model: `paraphrase-multilingual-MiniLM-L12-v2` (Hugging Face).
- `indic-transliteration` library (ITRANS to Gujarati script).
- scikit-learn documentation: TF-IDF and Logistic Regression.
