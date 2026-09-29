"""The three triage models and the keyword baseline they are measured against."""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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


def _features() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                 token_pattern=r"(?u)\b[\w-]{2,}\b")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
    ])


class KeywordClassifier:
    """Label a report with every category whose keywords it mentions."""

    def predict(self, texts: list[str]) -> list[list[str]]:
        out = []
        for t in texts:
            low = t.lower()
            out.append([c for c, words in KEYWORDS.items()
                        if any(re.search(r"\b" + re.escape(w), low) for w in words)])
        return out


class CategoryClassifier:
    """Multi-label vulnerability class: TF-IDF word + character n-grams, one logistic
    regression per class, and always at least one label (every report has a class)."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.binarizer = MultiLabelBinarizer(classes=CATEGORIES)
        self.model = make_pipeline(
            _features(),
            OneVsRestClassifier(LogisticRegression(max_iter=3000, C=8.0, class_weight="balanced")),
        )

    def fit(self, texts: list[str], labels: list[list[str]]) -> CategoryClassifier:
        y = self.binarizer.fit_transform(labels)
        self.model.fit(texts, y)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.model.predict_proba(texts)

    def predict(self, texts: list[str]) -> list[list[str]]:
        probs = self.predict_proba(texts)
        out = []
        for row in probs:
            picked = [c for c, p in zip(CATEGORIES, row) if p >= self.threshold]
            out.append(picked or [CATEGORIES[int(np.argmax(row))]])
        return out


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


def combine(*sims: np.ndarray) -> np.ndarray:
    return np.mean(np.stack(sims), axis=0)


__all__ = [
    "CATEGORIES", "SEVERITIES", "KeywordClassifier", "CategoryClassifier", "SeverityModel",
    "Deduplicator", "EmbeddingDeduplicator", "Match", "combine", "load_jsonl_gz",
    "mask_package", "text_of",
]
