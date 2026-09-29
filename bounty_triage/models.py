"""The three triage models and the keyword baseline they are measured against."""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, normalize

from .cvss import SEVERITIES

# RustSec's own category vocabulary (advisory-db CONTRIBUTING.md).
CATEGORIES = [
    "code-execution",
    "crypto-failure",
    "denial-of-service",
    "file-disclosure",
    "format-injection",
    "memory-corruption",
    "memory-exposure",
    "privilege-escalation",
    "thread-safety",
]

# The baseline a triager would write by hand on day one.
KEYWORDS = {
    "code-execution": ["arbitrary code", "code execution", "execute arbitrary", "rce", "command injection"],
    "crypto-failure": ["timing", "side-channel", "side channel", "constant-time", "signature", "nonce",
                       "cryptograph", "private key", "ciphertext", "certificate"],
    "denial-of-service": ["denial of service", "dos", "panic", "infinite loop", "stack overflow",
                          "exhaust", "hang", "crash", "unbounded", "out of memory", "oom"],
    "file-disclosure": ["path traversal", "directory traversal", "symlink", "arbitrary file"],
    "format-injection": ["injection", "escape", "sanitiz", "crlf", "header", "xss", "html"],
    "memory-corruption": ["use-after-free", "use after free", "double free", "out-of-bounds write",
                          "buffer overflow", "memory corruption", "heap overflow", "dangling"],
    "memory-exposure": ["uninitialized memory", "uninitialized", "out-of-bounds read", "leak", "exposure"],
    "privilege-escalation": ["privilege", "permission", "bypass", "authoriz", "authentication"],
    "thread-safety": ["send", "sync", "data race", "thread"],
}


def load_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def text_of(adv: dict) -> str:
    return f"{adv['title']}\n\n{adv['description']}"


def mask_package(text: str, package: str) -> str:
    """Hide a crate's name (in `-` and `_` spellings) so matching must rely on the bug itself."""
    if not package:
        return text
    variants = {package, package.replace("-", "_"), package.replace("_", "-")}
    pattern = r"\b(" + "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True)) + r")\b"
    return re.sub(pattern, "<crate>", text, flags=re.I)


def keyword_hits(text: str) -> list[str]:
    low = text.lower()
    return [c for c, words in KEYWORDS.items() if any(re.search(r"\b" + re.escape(w), low) for w in words)]


class KeywordFeatures(BaseEstimator, TransformerMixin):
    """One 0/1 column per category: does the text use that category's keywords? Gives the
    linear model an expert prior for classes with too few examples to learn from."""

    def fit(self, texts, y=None):
        return self

    def transform(self, texts):
        rows = [[1.0 if c in keyword_hits(t) else 0.0 for c in CATEGORIES] for t in texts]
        return csr_matrix(np.array(rows))


def _features(keywords: bool = False) -> FeatureUnion:
    parts = [
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                 token_pattern=r"(?u)\b[\w-]{2,}\b")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
    ]
    if keywords:
        parts.append(("keywords", KeywordFeatures()))
    return FeatureUnion(parts)


class KeywordClassifier:
    """Label a report with every category whose keywords it mentions."""

    def predict(self, texts: list[str]) -> list[list[str]]:
        return [keyword_hits(t) for t in texts]


class CategoryClassifier:
    """Multi-label vulnerability class: TF-IDF word + character n-grams (optionally plus keyword
    indicators), one logistic regression per class, and always at least one label.

    With `tune_thresholds`, each class gets its own decision threshold, chosen to maximise F1 on
    5-fold cross-validated predictions over the *training* data only."""

    GRID = np.round(np.arange(0.05, 0.95, 0.05), 2)

    def __init__(self, threshold: float = 0.5, tune_thresholds: bool = False, keywords: bool = False):
        self.thresholds = np.full(len(CATEGORIES), threshold)
        self.tune_thresholds = tune_thresholds
        self.binarizer = MultiLabelBinarizer(classes=CATEGORIES)
        self.model = make_pipeline(
            _features(keywords),
            OneVsRestClassifier(LogisticRegression(max_iter=3000, C=8.0, class_weight="balanced")),
        )

    def fit(self, texts: list[str], labels: list[list[str]]) -> CategoryClassifier:
        y = self.binarizer.fit_transform(labels)
        if self.tune_thresholds:
            folds = KFold(n_splits=5, shuffle=True, random_state=0)
            oof = cross_val_predict(self.model, texts, y, cv=folds, method="predict_proba")
            for k in range(y.shape[1]):
                if y[:, k].sum() == 0:
                    continue
                scores = [f1_binary(y[:, k].astype(bool), oof[:, k] >= t) for t in self.GRID]
                self.thresholds[k] = float(self.GRID[int(np.argmax(scores))])
        self.model.fit(texts, y)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.model.predict_proba(texts)

    def predict(self, texts: list[str]) -> list[list[str]]:
        probs = self.predict_proba(texts)
        out = []
        for row in probs:
            picked = [c for c, p, t in zip(CATEGORIES, row, self.thresholds) if p >= t]
            out.append(picked or [CATEGORIES[int(np.argmax(row / self.thresholds))]])
        return out


def f1_binary(truth: np.ndarray, pred: np.ndarray) -> float:
    tp = float(np.sum(truth & pred))
    denom = float(np.sum(truth) + np.sum(pred))
    return 2 * tp / denom if denom else 0.0


class SeverityModel:
    """CVSS severity band from text alone. A suggestion for the triager, never a verdict."""

    def __init__(self):
        self.model = make_pipeline(
            _features(),
            LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced"),
        )

    def fit(self, texts: list[str], bands: list[str]) -> SeverityModel:
        self.model.fit(texts, bands)
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return list(self.model.predict(texts))

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        classes = list(self.model.classes_)
        return [{c: float(p) for c, p in zip(classes, row)} for row in self.model.predict_proba(texts)]


@dataclass
class Match:
    id: str
    score: float


class Deduplicator:
    """Nearest known advisories for a new report: cosine similarity over TF-IDF word and
    character n-grams, averaged. Character n-grams carry identifiers like
    `SmallVec::insert_many` that word tokens split apart."""

    def __init__(self):
        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1,
                                    token_pattern=r"(?u)\b[\w:-]{2,}\b", stop_words="english")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.ids: list[str] = []

    def fit(self, ids: list[str], texts: list[str]) -> Deduplicator:
        self.ids = list(ids)
        self._w = self.word.fit_transform(texts)
        self._c = self.char.fit_transform(texts)
        return self

    def similarity(self, texts: list[str]) -> np.ndarray:
        w = (self.word.transform(texts) @ self._w.T).toarray()
        c = (self.char.transform(texts) @ self._c.T).toarray()
        return (w + c) / 2

    def query(self, text: str, k: int = 5) -> list[Match]:
        sims = self.similarity([text])[0]
        top = np.argsort(-sims)[:k]
        return [Match(self.ids[i], float(sims[i])) for i in top]


class EmbeddingDeduplicator:
    """Same interface over a sentence-embedding model (optional dependency)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional

        self.encoder = SentenceTransformer(model_name)
        self.ids: list[str] = []

    def fit(self, ids: list[str], texts: list[str]) -> EmbeddingDeduplicator:
        self.ids = list(ids)
        self._e = self._encode(texts)
        return self

    def _encode(self, texts: list[str]) -> np.ndarray:
        return normalize(np.asarray(self.encoder.encode(texts, batch_size=64, show_progress_bar=False)))

    def similarity(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts) @ self._e.T


def _names_package(text: str, package: str) -> bool:
    if not package:
        return False
    variants = {package, package.replace("-", "_"), package.replace("_", "-")}
    return any(re.search(r"(?<![\w-])" + re.escape(v) + r"(?![\w-])", text, re.I) for v in variants)


class DuplicateJudge:
    """Is the best match the same bug, or just the nearest of many unrelated advisories?

    The top similarity alone separates these poorly at high precision. The judge adds how far
    the best match stands out from the runner-up, and whether the report names the matched
    crate, and fits a logistic regression on those."""

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    @staticmethod
    def features(sims: np.ndarray, text: str, packages: list[str]) -> list[float]:
        order = np.argsort(-sims)[:2]
        top, second = float(sims[order[0]]), float(sims[order[1]])
        named = 1.0 if _names_package(text, packages[order[0]]) else 0.0
        return [top, top - second, named, top * named]

    @classmethod
    def training_pairs(cls, sims: np.ndarray, texts: list[str], truth: list[int], packages: list[str]):
        """Each query gives a positive (its advisory is indexed) and a negative (the same query
        with its advisory removed, as if the bug were new)."""
        X, y, groups = [], [], []
        for q, (row, text, t) in enumerate(zip(sims, texts, truth)):
            X.append(cls.features(row, text, packages))
            y.append(1)
            removed = row.copy()
            removed[t] = -1.0
            X.append(cls.features(removed, text, packages))
            y.append(0)
            groups += [q, q]
        return np.array(X), np.array(y), np.array(groups)

    def fit(self, X: np.ndarray, y: np.ndarray) -> DuplicateJudge:
        self.model.fit(X, y)
        return self

    def probability(self, sims: np.ndarray, text: str, packages: list[str]) -> float:
        return float(self.model.predict_proba([self.features(sims, text, packages)])[0, 1])


def combine(*sims: np.ndarray) -> np.ndarray:
    return np.mean(np.stack(sims), axis=0)


__all__ = [
    "CATEGORIES", "SEVERITIES", "DuplicateJudge", "KeywordClassifier", "CategoryClassifier", "SeverityModel",
    "Deduplicator", "EmbeddingDeduplicator", "Match", "combine", "load_jsonl_gz",
    "mask_package", "text_of",
]
