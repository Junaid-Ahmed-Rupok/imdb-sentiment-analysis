# IMDb Sentiment Analysis
### TF-IDF + Machine Learning · 50,000 Reviews · 3 Models Compared

> Classifying movie reviews as **positive** or **negative** using classical NLP techniques —
> a clean, reproducible pipeline from raw text to serialized, production-ready models.

![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)

---

## Results

| # | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 1 | **Logistic Regression** | **89.76%** | 0.889 | 0.909 | 0.899 |
| 2 | Linear SVC | 88.88% | 0.885 | 0.894 | 0.889 |
| 3 | Naive Bayes | 86.40% | 0.853 | 0.880 | 0.866 |

Logistic Regression wins on all four metrics. The ~0.9% gap over LinearSVC doesn't justify the added complexity of a different solver unless inference latency is a hard constraint.

---

## Repository structure

```
imdb-sentiment-analysis/
├── IMDb_Sentiment_TFIDF_Comparison.ipynb   # All 12 cells, fully annotated
├── requirements.txt
├── README.md
└── images/
    └── confusion_matrix.png
```

---

## Pipeline

```
Raw CSV
  └─► 01 Clean        unescape HTML · strip tags · lowercase · remove non-alpha
        └─► 02 Vectorize  TF-IDF fit on train only · ngram (1,2) · 10k features · English stop words
              └─► 03 Split     stratified 80/20 · 40k train / 10k test · seed = 42
                    └─► 04 Train     Logistic Regression · Naive Bayes · LinearSVC
                          └─► 05 Evaluate  classification report · confusion matrix · custom predictions
```

**Why fit TF-IDF on train only?** Fitting on the full dataset leaks test-set vocabulary and term frequencies into the vectorizer, inflating reported performance. The vectorizer is fit on `X_train`, then `transform`-only is applied to `X_test`.

**Why bigrams?** `ngram_range=(1,2)` captures negation and intensifiers that unigrams miss entirely — "not good", "very bad", "would recommend" are meaningfully different from their constituent words scored independently.

---

## Confusion matrix — Logistic Regression

```
                    Predicted
                 Negative   Positive
Actual Negative    4,431  │    569     88.6% specificity
       Positive      455  │  4,545     90.9% recall
```

| Metric | Value | Formula |
|---|---|---|
| Accuracy | 89.76% | (4431 + 4545) / 10000 |
| Recall (positive) | 90.90% | 4545 / (4545 + 455) |
| Precision (positive) | 88.87% | 4545 / (4545 + 569) |
| Specificity | 88.62% | 4431 / (4431 + 569) |
| F1 score | 89.88% | 2 × (0.889 × 0.909) / (0.889 + 0.909) |

False negatives (455) are fewer than false positives (569), meaning the model slightly favors recall — it misses fewer true positives than it incorrectly flags negatives. This is generally acceptable for a sentiment classifier.

---

## Custom review predictions

| Review (truncated) | Predicted | Confidence |
|---|---|---|
| "This movie was absolutely fantastic! The acting, the plot, everything was perfect..." | ✅ Positive | 94.3% |
| "Terrible waste of time. Boring script, horrible acting, and the sound was awful..." | ✅ Negative | 100.0% |
| "The movie was okay. Some parts were interesting but it was too long and confusing..." | Negative | 86.1% |

The ambiguous review predicts negative because tokens like "too long" and "confusing" carry strong negative TF-IDF weight, outweighing "okay" and "interesting".

---

## Quickstart

**Google Colab (recommended)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Junaid-Ahmed-Rupok/imdb-sentiment-analysis/blob/main/IMDb_Sentiment_TFIDF_Comparison.ipynb)

Upload the [IMDb Dataset CSV](http://ai.stanford.edu/~amaas/data/sentiment/) to your Drive at:
`MyDrive/DATASETS_For the data analysis/IMDB Dataset.csv`

**Local**

```bash
git clone https://github.com/Junaid-Ahmed-Rupok/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
pip install -r requirements.txt
jupyter notebook IMDb_Sentiment_TFIDF_Comparison.ipynb
```

**`requirements.txt`**

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

---

## Saved artifacts

After a full run, the following are saved to `/content/imdb_sentiment/` (or your Drive):

| File | Description |
|---|---|
| `tfidf_vectorizer.pkl` | Fitted vectorizer. Pass new text through `clean_text()` first, then `vectorizer.transform([text])`. |
| `logistic_regression_model.pkl` | Best model. Call `.predict()` for labels or `.predict_proba()` for confidence scores. |
| `model_comparison.csv` | Accuracy / precision / recall / F1 for all three models in a single CSV. |
| `cleaned_imdb_reviews.csv` | Pre-cleaned dataset. Load directly to skip the ~2-minute cleaning step on reruns. |

---

## Key findings

1. **LR ≈ LinearSVC** on sparse TF-IDF features. Both are linear classifiers; the ~0.9% gap is within noise for most downstream uses.
2. **Naive Bayes underperforms by ~3.4%** because its feature-independence assumption breaks for phrase-level sentiment signals.
3. **Bigrams are load-bearing.** Removing `ngram_range=(1,2)` and using unigrams-only measurably reduces recall on negation phrases.
4. **No class imbalance.** The balanced 25k/25k split makes accuracy a reliable headline metric — precision and recall are symmetric across classes.
5. **Training is fast.** All three models train in under 4 seconds combined on a standard CPU, making iteration cheap.

---

## Potential improvements

| Area | Approach | Expected impact |
|---|---|---|
| Representations | Word2Vec / GloVe / fastText embeddings | Captures semantic similarity TF-IDF misses |
| Transformers | Fine-tune `distilbert-base-uncased` | +5–8% accuracy, needs GPU |
| Evaluation | 5-fold cross-validation | More reliable metric estimates than a single split |
| Preprocessing | spaCy lemmatization | Reduces vocabulary size, may improve generalization |
| Ensemble | Soft-voting LR + LinearSVC | Marginal gains on edge cases |
| Deployment | Streamlit / Gradio web app | Wraps `.pkl` files for live review scoring |

---

## Dataset

- **Name:** IMDb Large Movie Review Dataset
- **Source:** [Stanford AI Lab](http://ai.stanford.edu/~amaas/data/sentiment/) — Maas et al., ACL 2011
- **Size:** 50,000 reviews, perfectly balanced (25k positive / 25k negative)
- **Format:** CSV — `review` (raw text) and `sentiment` (`positive` / `negative`)
- **License:** Non-commercial research use only

---

## Author

**Junaid Ahmed Rupok** — [github.com/Junaid-Ahmed-Rupok](https://github.com/Junaid-Ahmed-Rupok)

---

*Star this repo if it helped you learn NLP or get started with sentiment analysis.*



What about this one?
