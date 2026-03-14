"""
models.py
---------
Registry of baseline classifiers for news topic classification.

Design choices
--------------
* DummyClassifier     -- stratified-random baseline; establishes the floor.
* LogisticRegression  -- the standard linear baseline for text; fast, strong,
                         and interpretable via feature coefficients.
* SGDClassifier       -- online linear model with log-loss; scales well to
                         large vocabularies and is a common NLP workhorse.
* LinearSVC          -- max-margin linear classifier; often the strongest
                         linear method on TF-IDF features.
* RandomForest        -- non-linear ensemble; included to check whether
                         non-linearity helps on top of TF-IDF/SVD features.

Excluded models
---------------
* KNeighborsClassifier -- poor in high-dimensional TF-IDF space.
* DecisionTree         -- strictly dominated by RandomForest.
* AdaBoost             -- not a standard NLP text baseline; mediocre results.
* GaussianNB           -- assumes continuous features; MultinomialNB is better
                          for raw TF-IDF but incompatible with SVD (negative
                          values).  Omitted to keep the comparison clean.
"""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

RANDOM_STATE = 42


def get_models():
    """Return an ordered dict of {name: estimator} pairs.

    All models use fixed random_state for reproducibility and reasonable
    defaults that are not computationally expensive.
    """
    return {
        "Dummy (stratified)": DummyClassifier(
            strategy="stratified",
            random_state=RANDOM_STATE,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "SGD (log_loss)": SGDClassifier(
            loss="log_loss",
            penalty="l2",
            max_iter=1000,
            tol=1e-3,
            random_state=RANDOM_STATE,
        ),
        "LinearSVC": LinearSVC(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
