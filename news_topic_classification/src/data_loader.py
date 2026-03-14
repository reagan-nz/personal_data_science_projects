"""
data_loader.py
--------------
Load the news classification JSONL dataset into a pandas DataFrame.
Provides helpers for train/test splitting with stratification.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "news_classification_dataset.json"


def load_dataset(path=None):
    """Load the JSONL news dataset and return a clean DataFrame.

    Each line of the file is a JSON object with keys:
        - content: article text
        - annotation.label: list with a single topic string
        - metadata: annotator metadata (ignored here)

    Returns
    -------
    pd.DataFrame with columns [content, label]
    """
    path = Path(path) if path else DEFAULT_DATA_PATH

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            records.append(
                {
                    "content": obj["content"],
                    "label": obj["annotation"]["label"][0],
                }
            )

    df = pd.DataFrame(records)
    return df


def split_data(df, test_size=0.3, random_state=42):
    """Stratified train/test split on the label column.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns [content, label].
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    train_df, test_df : pd.DataFrame
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def holdout_by_class(df, n_per_class=5, random_state=42):
    """Sample a small holdout set with *n_per_class* examples per label.

    Useful for qualitative inspection or demo predictions.

    Returns
    -------
    holdout_df, remaining_df : pd.DataFrame
    """
    holdout_parts = []
    remaining_parts = []

    for label, group in df.groupby("label"):
        sampled = group.sample(n=n_per_class, random_state=random_state)
        holdout_parts.append(sampled)
        remaining_parts.append(group.drop(sampled.index))

    holdout_df = pd.concat(holdout_parts).reset_index(drop=True)
    remaining_df = pd.concat(remaining_parts).reset_index(drop=True)
    return holdout_df, remaining_df
