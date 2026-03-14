"""
features.py
------------
Build TF-IDF feature matrices with optional TruncatedSVD dimensionality
reduction.  All objects are returned so they can be reused for holdout /
production data.
"""

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(
    train_texts,
    test_texts,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
):
    """Fit a TF-IDF vectorizer on *train_texts* and transform both splits.

    Parameters
    ----------
    train_texts, test_texts : array-like of str
    ngram_range : tuple
    min_df, max_df : int or float

    Returns
    -------
    X_train : sparse matrix
    X_test  : sparse matrix
    vectorizer : fitted TfidfVectorizer
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer


def apply_svd(X_train, X_test, n_components=100, random_state=42):
    """Reduce dimensionality with TruncatedSVD (Latent Semantic Analysis).

    Parameters
    ----------
    X_train, X_test : sparse or dense matrix
    n_components : int
    random_state : int

    Returns
    -------
    X_train_reduced : ndarray
    X_test_reduced  : ndarray
    svd : fitted TruncatedSVD
    """
    svd = TruncatedSVD(
        n_components=n_components,
        n_iter=10,
        random_state=random_state,
    )
    X_train_reduced = svd.fit_transform(X_train)
    X_test_reduced = svd.transform(X_test)
    return X_train_reduced, X_test_reduced, svd


def build_features(
    train_texts,
    test_texts,
    use_svd=False,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    svd_components=100,
    random_state=42,
):
    """End-to-end feature builder: TF-IDF with optional SVD.

    Returns
    -------
    dict with keys:
        X_train, X_test   -- feature matrices (sparse or dense)
        vectorizer         -- fitted TfidfVectorizer
        svd                -- fitted TruncatedSVD or None
    """
    X_train, X_test, vectorizer = build_tfidf(
        train_texts, test_texts,
        ngram_range=ngram_range, min_df=min_df, max_df=max_df,
    )

    svd = None
    if use_svd:
        X_train, X_test, svd = apply_svd(
            X_train, X_test,
            n_components=svd_components, random_state=random_state,
        )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "vectorizer": vectorizer,
        "svd": svd,
    }
