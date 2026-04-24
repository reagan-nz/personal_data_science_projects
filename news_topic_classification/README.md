# News Topic Classification: Preprocessing & Metadata Ablation Study

## Table of contents

- [Research question](#research-question)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Experiment design](#experiment-design)
- [Feature engineering](#feature-engineering)
- [Preprocessing strategies](#preprocessing-strategies)
- [Models](#models)
- [Evaluation metrics](#evaluation-metrics)
- [Code layout](#code-layout)
- [Setup](#setup)
- [Run the experiments](#run-the-experiments)
- [Error analysis (optional)](#error-analysis-optional)
- [Representative findings](#representative-findings)
- [Limitations and extensions](#limitations-and-extensions)
- [Citation](#citation)
- [License](#license)

## Research question

> **How much do preprocessing choices and metadata artifacts (e.g., news-wire prefixes) affect news topic classification performance?**

This project compares **classical NLP pipelines** (TF-IDF / optional SVD + linear and tree-based classifiers) under **controlled ablations** on preprocessing, so we can see how much of the performance comes from the text content versus formatting, noise removal, and incidental signals such as "Reuters" or "AP" at the start of a story.

## Motivation

In applied text classification, it is easy to **over-engineer** preprocessing: lemmatization, stop-word lists, and aggressive cleaning are often applied by default, but their marginal benefit is rarely measured against simpler baselines. At the same time, **metadata leakage** is common in news: wire-service bylines and datelines are correlated with topic and can inflate accuracy without the model “reading” the body text.

This repository implements a **reproducible grid** to quantify those effects on a public-style news topic dataset.

## Dataset

| Property | Value |
|--------|--------|
| **File** | `data/news_classification_dataset.json` (JSONL: one JSON object per line) |
| **Fields used** | `content` (article text), `annotation.label[0]` (topic) |
| **Total articles** | 7,600 |
| **Classes** | Business, SciTech, Sports, World (multiclass, single label per document) |
| **Format** | Each line: `content`, `annotation` with `label` as a one-element list, `metadata` (not used in training) |

**Note:** Class counts are imbalanced in the raw data (e.g., SciTech and World are more frequent than Business and Sports). All splits are **stratified** so each class is represented proportionally in train and test.

## Experiment design

The core experiment is a **three-way grid**:

1. **Preprocessing strategy** (5) — `raw`, `basic`, `remove_source_prefix_only`, `remove_noise_only`, `aggressive` (see [Preprocessing strategies](#preprocessing-strategies)).
2. **Feature configuration** (2) — **TF-IDF only** vs **TF-IDF + TruncatedSVD** (LSA-style dimensionality reduction).
3. **Classifier** (5) — `Dummy` (stratified), `LogisticRegression`, `SGDClassifier` (log loss), `LinearSVC`, `RandomForestClassifier` (see [Models](#models)).

**Total runs:** 5 × 2 × 5 = **50** distinct configurations, each evaluated on the **same** held-out test set for comparability.

**Reproducibility:** a fixed `random_state` (42) is used for `train_test_split`, SVD, and models that accept a seed. NLTK data is stored under a project-local `nltk_data/` directory when you first run code that needs tokenization / stopwords / WordNet.

## Feature engineering

| Component | Default behavior |
|----------|------------------|
| **TfidfVectorizer** | `ngram_range=(1, 2)` (unigrams + bigrams), `min_df=2`, `max_df=0.95`, `sublinear_tf=True` |
| **TruncatedSVD** (optional) | `n_components=100`, `n_iter=10` — projects sparse TF–IDF into 100 dense topics |

SVD is included as an ablation: it can **speed** training and **compress** the feature space, but it may also **discard** discriminative signal on this dataset size and vocabulary.

## Preprocessing strategies

| Key | What it does | Purpose in the study |
|-----|----------------|------------------------|
| `raw` | No text change | Upper bound of “use what you have” |
| `basic` | Lowercase, normalize whitespace, strip HTML/entity-style noise | Cheap normalization |
| `remove_source_prefix_only` | Remove leading wire-style prefixes (e.g., `AP -`, `Reuters -`, `NEW YORK (Reuters) -`) | Isolate **metadata** at the start of the article |
| `remove_noise_only` | Remove `&quot;`, `&lt;`, `&#...;` style junk without other changes | Noise vs. content |
| `aggressive` | Prefix removal + lowercase + noise + tokenization + stopwords + **POS-aware lemmatization** | “Full classical NLP” pipeline |

Implementations live in `src/preprocess.py` in a `PREPROCESSING_FUNCTIONS` map so new strategies can be added without changing the experiment loop.

## Models

| Model | Role |
|------|------|
| **DummyClassifier (stratified)** | Random baseline: expected accuracy ≈ class priors; defines a **floor** |
| **LogisticRegression** | Standard **linear** multiclass baseline; strong on sparse TF–IDF |
| **SGDClassifier (log_loss)** | Linear model trained with SGD; good default for **large sparse** features |
| **LinearSVC** | Linear SVM; often very competitive on high-dimensional text |
| **RandomForestClassifier** | **Nonlinear** ensemble; checks whether trees add value on top of linear baselines |

Models excluded (by design) from the main table: *k*NN, single decision trees, AdaBoost, Naive Bayes variants — they are either weak defaults for high-dimensional text or overlap with the baselines above.

## Evaluation metrics

| Metric | Role |
|--------|------|
| **Accuracy** | Overall fraction correct |
| **Macro precision / recall / F1** | Averages **across classes with equal weight** (important when class sizes differ) |
| **Primary sort key** | **Macro F1** in `results/metrics.csv` (descending) |

`evaluate.py` also returns a **confusion matrix**, a **text classification report**, and a **per-class metrics** DataFrame for deeper analysis.

## Code layout

```
news_topic_classification/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── news_classification_dataset.json
├── src/
│   ├── data_loader.py          # load JSONL → DataFrame; stratified split; optional holdout-by-class
│   ├── preprocess.py         # named preprocessing strategies
│   ├── features.py            # TF-IDF + optional TruncatedSVD
│   ├── models.py              # model registry
│   ├── evaluate.py            # metrics, confusion matrix, per-class table
│   ├── experiment_runner.py  # full 50-config grid; writes results/metrics.csv
│   └── error_analysis.py     # error rates by length, source-prefix flag, etc.
├── results/                   # generated (gitignored by default in .gitignore)
│   └── metrics.csv
├── nltk_data/                 # optional; NLTK resources downloaded on first use
└── archive/
    ├── original_notebook.ipynb
    └── nlp_utils_news.ipynb
```

## Setup

**Requirements:** Python 3.10+ and packages listed in `requirements.txt` (pandas, scikit-learn, nltk, beautifulsoup4, html5lib).

```bash
cd news_topic_classification
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run from the `news_topic_classification` directory so imports resolve. If you use a custom layout, set `PYTHONPATH` to the project root.

## Run the experiments

```bash
cd news_topic_classification
python -m src.experiment_runner
```

- Writes **`results/metrics.csv`**: one row per (preprocessing, `use_svd`, model) with accuracy, macro precision/recall/F1, and wall-clock **train** time in seconds.
- The runner also returns **in-memory** detail objects (predictions, confusion matrices, per-class reports) for downstream error analysis; see `error_analysis.py`.

## Error analysis (optional)

`src/error_analysis.py` builds tables for a single run, including:

- Misclassified examples with **original** text, true vs predicted label
- **Per-class** error rates
- **Length buckets** (terciles of word count) vs error rate
- **`has_source_prefix`** (regex on raw text) vs error rate — to see whether **metadata-heavy** starts correlate with **easier** or **harder** classification

Use this after you pick a row from `metrics.csv` (e.g., best macro F1) and re-run a single training path, or wire the stored `details` from `experiment_runner` in a small script or notebook.

## Representative findings

These numbers are **illustrative**; your exact file may differ slightly if code or data change.

- **Raw text + no SVD + linear model** (e.g. LinearSVC) performs **on par** with the **aggressive** pipeline — improvements from heavy cleaning are **small** (on the order of **fractions of a point** in macro F1, not a large gain).
- **SVD (100 components)** usually **lowers** macro F1 relative to full TF–IDF on this task — the compressed representation loses useful discriminative detail for this scale.
- **Removing only source-prefix lines** can **slightly** change scores vs raw — consistent with the idea that bylines are **weak but useful** correlates, not the only signal in the text.
- **Error analysis** (length / prefix): **shorter** articles often show **higher** error rates; articles **with** a detected wire-style prefix sometimes show **lower** error (dependent on the exact split; always recompute on your run).

**Interpretation (non-causal):** The results support treating **aggressive lemmatization** as *optional* for this type of pipeline if the goal is a strong bag-of-words + linear model. They also justify reporting **separate** ablations for **source-like metadata** so readers can judge how much the model leans on bylines.

## Limitations and extensions

- **Single train/test split** — not cross-validated; for publication-grade claims, add k-fold CV and confidence intervals.
- **Classical only** — no deep learning; a natural extension is a **transformer** baseline (e.g. small fine-tuned encoder) for comparison.
- **Hyperparameters** are **fixed defaults**; grid search is not part of the main grid (to keep the ablation about preprocessing, not tune-heavy models).

## Citation

If you use this project or the experimental setup in academic work, cite the **dataset source** and this repository. The JSON dataset in `data/` is a standard public-style news topic benchmark used in many educational notebooks; when publishing, name the **exact file version** and your **code commit hash** for reproducibility.

## License

This repository is for **research and education**. Verify licensing for the **dataset** and any code you redistribute.
