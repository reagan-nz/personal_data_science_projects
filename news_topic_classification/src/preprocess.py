"""
preprocess.py
-------------
Multiple preprocessing strategies for ablation experiments.

Each strategy is a callable that maps a raw article string to a cleaned string.
All strategies are registered in PREPROCESSING_FUNCTIONS so the experiment
runner can iterate over them by name.

Strategies
----------
raw                        -- identity; return original text
basic                      -- lowercase, normalize whitespace, strip formatting
remove_source_prefix_only  -- strip leading source tags (AP -, Reuters -, ...)
remove_noise_only          -- remove html-entity remnants (quot, lt, gt, amp)
aggressive                 -- full pipeline: source removal + lowercase +
                              noise removal + punctuation/number removal +
                              stop-word removal + lemmatization
"""

import re
import string

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Ensure required NLTK data is available (downloads to project-local dir)
# ---------------------------------------------------------------------------
from pathlib import Path as _Path

_NLTK_DATA_DIR = str(_Path(__file__).resolve().parent.parent / "nltk_data")
nltk.data.path.insert(0, _NLTK_DATA_DIR)

for _resource in ("punkt", "punkt_tab", "averaged_perceptron_tagger",
                  "averaged_perceptron_tagger_eng", "wordnet", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{_resource}" if "punkt" in _resource
                       else f"corpora/{_resource}" if _resource in ("wordnet", "stopwords")
                       else f"taggers/{_resource}")
    except LookupError:
        nltk.download(_resource, download_dir=_NLTK_DATA_DIR, quiet=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STOP_WORDS = set(stopwords.words("english"))

# Regex that catches common news-wire prefixes:
#   "AP -", "Reuters -", "AFP -", "SPACE.com -",
#   "NEW YORK (Reuters) -", "LONDON (AP) -", etc.
_SOURCE_PREFIX_RE = re.compile(
    r"^"
    r"(?:"
    r"[A-Z][A-Za-z.]+(?:\s*-\s)"            # e.g. "AP -", "SPACE.com -"
    r"|"
    r"[A-Z ]{2,}?\s*\([A-Za-z.]+\)\s*-?\s*" # e.g. "NEW YORK (Reuters) -"
    r"|"
    r"By\s+[A-Z ]+\s+"                       # e.g. "By MICHAEL LIEDTKE ..."
    r")"
)

# HTML-entity remnants that survive naive stripping
_NOISE_TOKENS = {"quot", "amp", "lt", "gt", "nbsp"}
_NOISE_RE = re.compile(
    r"&(?:quot|amp|lt|gt|nbsp);?"  # &quot; or bare quot
    r"|#\d+;"                       # &#36; etc.
    r"|\\+"                         # stray backslashes
)

_LEMMATIZER = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_whitespace(text):
    """Collapse runs of whitespace (including newlines) into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def _remove_source_prefix(text):
    return _SOURCE_PREFIX_RE.sub("", text)


def _remove_noise(text):
    """Remove html-entity leftovers and encoding artifacts."""
    return _NOISE_RE.sub(" ", text)


def _wordnet_pos(treebank_tag):
    """Map a Penn Treebank POS tag to a WordNet POS constant."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN  # default


def _lemmatize_tokens(tokens):
    tagged = nltk.pos_tag(tokens)
    return [
        _LEMMATIZER.lemmatize(word, pos=_wordnet_pos(tag))
        for word, tag in tagged
    ]


# ---------------------------------------------------------------------------
# Public preprocessing strategies
# ---------------------------------------------------------------------------

def preprocess_raw(text):
    """Identity -- return original text unchanged."""
    return text


def preprocess_basic(text):
    """Lowercase, strip formatting noise, normalize whitespace."""
    text = text.lower()
    text = _remove_noise(text)
    text = _normalize_whitespace(text)
    return text


def preprocess_remove_source_prefix_only(text):
    """Remove leading source-style prefixes; leave everything else intact."""
    text = _remove_source_prefix(text)
    text = _normalize_whitespace(text)
    return text


def preprocess_remove_noise_only(text):
    """Remove HTML-entity remnants and encoding junk; leave casing and
    structure intact."""
    text = _remove_noise(text)
    text = _normalize_whitespace(text)
    return text


def preprocess_aggressive(text):
    """Full pipeline: source removal -> lowercase -> noise removal ->
    tokenize -> remove punctuation & numbers -> remove stopwords ->
    POS-aware lemmatization.  Returns a space-joined string of tokens."""
    text = _remove_source_prefix(text)
    text = text.lower()
    text = _remove_noise(text)
    text = _normalize_whitespace(text)

    tokens = word_tokenize(text)

    # Remove pure-punctuation and pure-number tokens
    tokens = [
        t for t in tokens
        if t not in string.punctuation
        and not t.isnumeric()
        and t not in _NOISE_TOKENS
    ]

    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = _lemmatize_tokens(tokens)

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Strategy registry -- experiment_runner iterates over this dict
# ---------------------------------------------------------------------------

PREPROCESSING_FUNCTIONS = {
    "raw": preprocess_raw,
    "basic": preprocess_basic,
    "remove_source_prefix_only": preprocess_remove_source_prefix_only,
    "remove_noise_only": preprocess_remove_noise_only,
    "aggressive": preprocess_aggressive,
}
