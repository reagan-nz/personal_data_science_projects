"""
error_analysis.py
-----------------
Research-style error analysis for a single experiment condition.

Given the *details* list produced by experiment_runner, this module:
  1. Builds a DataFrame of predictions with metadata columns.
  2. Identifies which label pairs are most confused.
  3. Surfaces misclassified examples for manual inspection.
  4. Computes error rates stratified by article length and by a
     "has_source_prefix" flag, testing whether metadata-heavy articles
     are easier or harder to classify.

Designed to be called from a notebook or script:

    from src.error_analysis import build_analysis_df, confusion_summary, ...
"""

import re

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Same pattern used in preprocess.py for consistency
_SOURCE_PREFIX_RE = re.compile(
    r"^"
    r"(?:"
    r"[A-Z][A-Za-z.]+(?:\s*-\s)"
    r"|"
    r"[A-Z ]{2,}?\s*\([A-Za-z.]+\)\s*-?\s*"
    r"|"
    r"By\s+[A-Z ]+\s+"
    r")"
)


# ------------------------------------------------------------------
# 1. Build analysis DataFrame
# ------------------------------------------------------------------

def build_analysis_df(detail):
    """Create a DataFrame from one experiment *detail* dict.

    Columns added:
        content         -- original article text
        true_label      -- ground-truth label (string)
        pred_label      -- predicted label (string)
        correct         -- bool
        word_count      -- number of whitespace-delimited tokens in content
        has_source_prefix -- bool, whether the raw text starts with a source tag
        length_bucket   -- categorical: short / medium / long
    """
    label_names = detail["label_names"]

    df = pd.DataFrame({
        "content": detail["test_contents"],
        "preprocessed_text": detail["test_texts"],
        "true_label": [label_names[i] for i in detail["y_test"]],
        "pred_label": [label_names[i] for i in detail["y_pred"]],
    })

    df["correct"] = df["true_label"] == df["pred_label"]
    df["word_count"] = df["content"].str.split().str.len()
    df["has_source_prefix"] = df["content"].apply(
        lambda t: bool(_SOURCE_PREFIX_RE.match(t))
    )

    # Bucket articles by length (terciles)
    df["length_bucket"] = pd.qcut(
        df["word_count"], q=3, labels=["short", "medium", "long"]
    )

    return df


# ------------------------------------------------------------------
# 2. Confusion summary
# ------------------------------------------------------------------

def confusion_summary(detail):
    """Return a labeled confusion matrix as a DataFrame."""
    labels = detail["label_names"]
    cm = confusion_matrix(detail["y_test"], detail["y_pred"])
    return pd.DataFrame(cm, index=labels, columns=labels)


def most_confused_pairs(detail, top_n=5):
    """Return the most-confused (true, predicted) pairs with counts.

    Only off-diagonal entries are considered.
    """
    labels = detail["label_names"]
    cm = confusion_matrix(detail["y_test"], detail["y_pred"])

    pairs = []
    n = len(labels)
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append({
                    "true": labels[i],
                    "predicted": labels[j],
                    "count": cm[i, j],
                })

    return (
        pd.DataFrame(pairs)
        .sort_values("count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ------------------------------------------------------------------
# 3. Misclassified examples
# ------------------------------------------------------------------

def misclassified_examples(analysis_df, n_per_class=5):
    """Return a sample of misclassified articles for manual inspection.

    Groups by true_label and samples up to *n_per_class* errors per class.
    """
    errors = analysis_df[~analysis_df["correct"]].copy()
    samples = []
    for label, group in errors.groupby("true_label"):
        sample = group.head(n_per_class)
        samples.append(sample)

    if not samples:
        return pd.DataFrame()
    return pd.concat(samples).reset_index(drop=True)


def per_class_error_counts(analysis_df):
    """Number and rate of errors per true class."""
    grouped = analysis_df.groupby("true_label").agg(
        total=("correct", "size"),
        n_correct=("correct", "sum"),
    )
    grouped["n_errors"] = grouped["total"] - grouped["n_correct"]
    grouped["error_rate"] = grouped["n_errors"] / grouped["total"]
    return grouped.sort_values("error_rate", ascending=False)


# ------------------------------------------------------------------
# 4. Error rates by article length and metadata flag
# ------------------------------------------------------------------

def error_rate_by_length(analysis_df):
    """Error rate grouped by length_bucket."""
    return (
        analysis_df.groupby("length_bucket", observed=True)
        .agg(
            total=("correct", "size"),
            n_errors=("correct", lambda s: (~s).sum()),
        )
        .assign(error_rate=lambda d: d["n_errors"] / d["total"])
    )


def error_rate_by_source_prefix(analysis_df):
    """Error rate for articles with vs. without a source prefix."""
    return (
        analysis_df.groupby("has_source_prefix")
        .agg(
            total=("correct", "size"),
            n_errors=("correct", lambda s: (~s).sum()),
        )
        .assign(error_rate=lambda d: d["n_errors"] / d["total"])
    )


def error_rate_by_length_and_prefix(analysis_df):
    """Cross-tabulation of error rate by length bucket and source prefix."""
    return (
        analysis_df.groupby(["length_bucket", "has_source_prefix"], observed=True)
        .agg(
            total=("correct", "size"),
            n_errors=("correct", lambda s: (~s).sum()),
        )
        .assign(error_rate=lambda d: d["n_errors"] / d["total"])
    )


# ------------------------------------------------------------------
# 5. Convenience: run full analysis for one experiment
# ------------------------------------------------------------------

def full_error_report(detail, n_examples=5, verbose=True):
    """Run all analyses for a single experiment detail dict.

    Returns a dict of DataFrames suitable for notebook display or CSV export.
    """
    adf = build_analysis_df(detail)

    report = {
        "analysis_df": adf,
        "confusion": confusion_summary(detail),
        "most_confused": most_confused_pairs(detail),
        "misclassified_samples": misclassified_examples(adf, n_per_class=n_examples),
        "per_class_errors": per_class_error_counts(adf),
        "error_by_length": error_rate_by_length(adf),
        "error_by_prefix": error_rate_by_source_prefix(adf),
        "error_by_length_and_prefix": error_rate_by_length_and_prefix(adf),
    }

    if verbose:
        config = f"{detail['preprocessing']} / {'SVD' if detail['use_svd'] else 'no SVD'} / {detail['model']}"
        print(f"Error analysis for: {config}")
        print(f"  Accuracy: {detail['accuracy']:.4f}  |  Macro F1: {detail['macro_f1']:.4f}")
        print()
        print("Confusion matrix:")
        print(report["confusion"])
        print()
        print("Most confused pairs:")
        print(report["most_confused"])
        print()
        print("Per-class error rates:")
        print(report["per_class_errors"])
        print()
        print("Error rate by article length:")
        print(report["error_by_length"])
        print()
        print("Error rate by source-prefix presence:")
        print(report["error_by_prefix"])

    return report
