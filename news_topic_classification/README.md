# News Topic Classification: Preprocessing Ablation Study

**Research question:** How much do preprocessing choices and metadata artifacts affect news topic classification performance?

## Overview

This project investigates the sensitivity of classical NLP text classifiers to different levels of preprocessing on a 4-class news article dataset (Business, SciTech, Sports, World). It runs a controlled ablation across 5 preprocessing strategies, 2 feature configurations, and 5 baseline models, producing a 50-row results table for systematic comparison.

## Dataset

- **Source:** News Classification DataSet (JSONL, 7,600 articles)
- **Classes:** Business, SciTech, Sports, World
- **Location:** `data/news_classification_dataset.json`

## Project Structure

```
news_topic_classification/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── news_classification_dataset.json
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Dataset loading and train/test splitting
│   ├── preprocess.py           # 5 preprocessing strategies for ablation
│   ├── features.py             # TF-IDF vectorization with optional SVD
│   ├── models.py               # Baseline classifier registry
│   ├── evaluate.py             # Evaluation metrics and result formatting
│   ├── experiment_runner.py    # Full experiment grid orchestrator
│   └── error_analysis.py       # Research-style error analysis utilities
├── results/
│   └── metrics.csv             # Generated experiment results
└── archive/
    ├── original_notebook.ipynb # Original exploratory notebook
    └── nlp_utils_news.ipynb    # Original NLP utility notebook
```

## Preprocessing Strategies

| Strategy | Description |
|---|---|
| `raw` | No modification; original text as-is |
| `basic` | Lowercase, normalize whitespace, remove encoding noise |
| `remove_source_prefix_only` | Strip leading wire-service tags (AP, Reuters, etc.) |
| `remove_noise_only` | Remove HTML-entity remnants (`&quot;`, `&#36;`, etc.) |
| `aggressive` | Full pipeline: source removal + lowercase + noise removal + punctuation/number removal + stopword removal + lemmatization |

## Models

| Model | Rationale |
|---|---|
| DummyClassifier | Stratified-random baseline (performance floor) |
| LogisticRegression | Standard linear text classification baseline |
| SGDClassifier (log_loss) | Scalable online linear model |
| LinearSVC | Max-margin linear classifier |
| RandomForestClassifier | Non-linear ensemble for comparison |

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full experiment grid:

```bash
cd news_topic_classification
python -m src.experiment_runner
```

Results are saved to `results/metrics.csv`.

## Key Findings

- Heavy preprocessing provides only marginal gains over raw text (+0.25 F1 points)
- SVD dimensionality reduction consistently degrades performance on this dataset
- Source-prefix metadata (Reuters, AP) acts as weak but useful signal for classification
- Short articles are harder to classify than long articles across all configurations

## Requirements

- Python 3.10+
- pandas, scikit-learn, nltk, beautifulsoup4
