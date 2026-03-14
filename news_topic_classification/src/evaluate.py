"""
evaluate.py
-----------
Consistent evaluation harness for all experiment conditions.

Returns structured results so the experiment runner can aggregate them
into a single results table.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(model, X_test, y_test, label_names=None):
    """Score a fitted model on the test split.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_test : array-like
    y_test : array-like of int (encoded labels)
    label_names : list[str] or None
        Human-readable class names for the confusion matrix and report.

    Returns
    -------
    dict with keys:
        accuracy      : float
        macro_precision : float
        macro_recall  : float
        macro_f1      : float
        confusion_matrix : np.ndarray
        classification_report : str
        per_class_df  : pd.DataFrame  (precision / recall / f1 per class)
        y_pred        : np.ndarray
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    report_str = classification_report(
        y_test, y_pred,
        target_names=label_names,
        zero_division=0,
    )

    # Per-class metrics as a DataFrame for downstream analysis
    report_dict = classification_report(
        y_test, y_pred,
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )
    per_class = {
        k: v for k, v in report_dict.items()
        if k not in ("accuracy", "macro avg", "weighted avg")
    }
    per_class_df = pd.DataFrame(per_class).T

    return {
        "accuracy": acc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
        "confusion_matrix": cm,
        "classification_report": report_str,
        "per_class_df": per_class_df,
        "y_pred": y_pred,
    }


def results_to_dataframe(records):
    """Convert a list of flat result dicts into a sorted DataFrame.

    Each record should contain at least:
        preprocessing, use_svd, model, accuracy, macro_precision,
        macro_recall, macro_f1
    """
    df = pd.DataFrame(records)
    df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    return df


def save_results(df, path):
    """Persist the results table to CSV."""
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")
