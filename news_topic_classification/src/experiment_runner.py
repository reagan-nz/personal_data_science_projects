"""
experiment_runner.py
--------------------
Run the full ablation grid:

    preprocessing variants  x  SVD settings  x  baseline models

Results are collected into a single DataFrame and saved to
results/metrics.csv.  Detailed per-experiment outputs (confusion matrices,
classification reports, predictions) are kept in memory for downstream
error analysis.

Usage
-----
    python -m src.experiment_runner          # from project root
    python src/experiment_runner.py          # also works
"""

import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.data_loader import load_dataset, split_data
from src.evaluate import evaluate_model, results_to_dataframe, save_results
from src.features import build_features
from src.models import get_models
from src.preprocess import PREPROCESSING_FUNCTIONS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RANDOM_STATE = 42
TEST_SIZE = 0.3


def run_experiments(data_path=None, verbose=True):
    """Execute the full experiment grid.

    Returns
    -------
    results_df : pd.DataFrame
        One row per (preprocessing, svd_setting, model) combination with
        accuracy, macro_precision, macro_recall, macro_f1.
    details : list[dict]
        Full evaluation dicts (confusion matrices, predictions, etc.)
    """
    # ------------------------------------------------------------------
    # 1. Load raw data once
    # ------------------------------------------------------------------
    df = load_dataset(path=data_path)
    train_df, test_df = split_data(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"])
    y_test = le.transform(test_df["label"])
    label_names = list(le.classes_)

    if verbose:
        print(f"Dataset loaded: {len(train_df)} train / {len(test_df)} test")
        print(f"Labels: {label_names}")
        print(f"Preprocessing variants: {list(PREPROCESSING_FUNCTIONS.keys())}")
        print(f"SVD settings: [False, True]")
        print(f"Models: {list(get_models().keys())}")
        print("-" * 70)

    records = []   # flat rows for the results CSV
    details = []   # rich dicts for error analysis

    # ------------------------------------------------------------------
    # 2. Iterate over the ablation grid
    # ------------------------------------------------------------------
    for prep_name, prep_fn in PREPROCESSING_FUNCTIONS.items():
        if verbose:
            print(f"\n=== Preprocessing: {prep_name} ===")

        # Apply preprocessing (train and test independently to avoid leakage)
        train_texts = train_df["content"].apply(prep_fn).values
        test_texts = test_df["content"].apply(prep_fn).values

        for use_svd in [False, True]:
            svd_label = "with_SVD" if use_svd else "no_SVD"
            if verbose:
                print(f"  Feature config: {svd_label}")

            feat = build_features(
                train_texts, test_texts,
                use_svd=use_svd,
                random_state=RANDOM_STATE,
            )
            X_train = feat["X_train"]
            X_test = feat["X_test"]

            # For LinearSVC with sparse input after SVD (dense), sklearn
            # handles both, so no conversion needed.

            for model_name, model in get_models().items():
                t0 = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - t0

                result = evaluate_model(
                    model, X_test, y_test, label_names=label_names,
                )

                row = {
                    "preprocessing": prep_name,
                    "use_svd": use_svd,
                    "model": model_name,
                    "accuracy": round(result["accuracy"], 4),
                    "macro_precision": round(result["macro_precision"], 4),
                    "macro_recall": round(result["macro_recall"], 4),
                    "macro_f1": round(result["macro_f1"], 4),
                    "train_time_s": round(train_time, 2),
                }
                records.append(row)

                detail = {
                    **row,
                    "confusion_matrix": result["confusion_matrix"],
                    "classification_report": result["classification_report"],
                    "per_class_df": result["per_class_df"],
                    "y_pred": result["y_pred"],
                    "y_test": y_test,
                    "label_names": label_names,
                    "test_texts": test_texts,
                    "test_labels": test_df["label"].values,
                    "test_contents": test_df["content"].values,
                }
                details.append(detail)

                if verbose:
                    print(
                        f"    {model_name:30s}  "
                        f"F1={row['macro_f1']:.4f}  "
                        f"Acc={row['accuracy']:.4f}  "
                        f"({train_time:.1f}s)"
                    )

    # ------------------------------------------------------------------
    # 3. Aggregate and save
    # ------------------------------------------------------------------
    results_df = results_to_dataframe(records)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "metrics.csv"
    save_results(results_df, out_path)

    if verbose:
        print("\n" + "=" * 70)
        print("Top 10 configurations by macro F1:")
        print(results_df.head(10).to_string(index=False))

    return results_df, details


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    results_df, details = run_experiments()
