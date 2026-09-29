"""Measure every model against its baseline on a time-based split and write the results.

    python -m bounty_triage.evaluate --data data --out reports

Training uses advisories published before SPLIT_DATE and testing uses everything after it, so
the numbers estimate how the models do on reports that arrive *later*, which is how a triage
queue sees them. A random split would leak near-identical advisories for the same crate
across train and test and overstate accuracy.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import MultiLabelBinarizer

from .cvss import SEVERITIES
from .models import (
    CATEGORIES,
    CategoryClassifier,
    Deduplicator,
    DuplicateJudge,
    KeywordClassifier,
    SeverityModel,
    combine,
    load_jsonl_gz,
    mask_package,
    text_of,
)

SPLIT_DATE = "2024-01-01"


def _split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    return [r for r in rows if r["date"] < SPLIT_DATE], [r for r in rows if r["date"] >= SPLIT_DATE]


def _multilabel_scores(true: list[list[str]], pred: list[list[str]]) -> dict:
    mlb = MultiLabelBinarizer(classes=CATEGORIES)
    y, p = mlb.fit_transform(true), mlb.transform(pred)
    prec, rec, f1, support = precision_recall_fscore_support(y, p, zero_division=0)
    return {
        "micro_f1": round(float(f1_score(y, p, average="micro", zero_division=0)), 3),
        "macro_f1": round(float(f1_score(y, p, average="macro", zero_division=0)), 3),
        "exact_match": round(float(np.mean([set(a) == set(b) for a, b in zip(true, pred)])), 3),
        "any_correct": round(float(np.mean([bool(set(a) & set(b)) for a, b in zip(true, pred)])), 3),
        "per_label": {
            c: {"precision": round(float(pr), 3), "recall": round(float(r), 3),
                "f1": round(float(f), 3), "support": int(s)}
            for c, pr, r, f, s in zip(CATEGORIES, prec, rec, f1, support)
        },
    }


def eval_categories(advisories: list[dict]) -> dict:
    rows = [a for a in advisories if a["categories"]]
    train, test = _split(rows)
    texts = [text_of(a) for a in test]
    truth = [a["categories"] for a in test]

    train_texts, train_labels = [text_of(a) for a in train], [a["categories"] for a in train]
    variants = {
        "logreg_threshold_0.5": CategoryClassifier(),
        "logreg_tuned_thresholds": CategoryClassifier(tune_thresholds=True),
        "logreg_keywords_tuned": CategoryClassifier(tune_thresholds=True, keywords=True),
    }
    out = {
        "train": len(train),
        "test": len(test),
        "label_counts_test": dict(Counter(c for t in truth for c in t)),
        "keyword_baseline": _multilabel_scores(truth, KeywordClassifier().predict(texts)),
    }
    for name, model in variants.items():
        model.fit(train_texts, train_labels)
        out[name] = _multilabel_scores(truth, model.predict(texts))
        out[name]["thresholds"] = dict(zip(CATEGORIES, map(float, model.thresholds)))
    return out


def eval_severity(advisories: list[dict]) -> dict:
    rows = [a for a in advisories if a.get("severity")]
    train, test = _split(rows)
    truth = [a["severity"] for a in test]
    model = SeverityModel().fit([text_of(a) for a in train], [a["severity"] for a in train])
    pred = model.predict([text_of(a) for a in test])
    majority = Counter(a["severity"] for a in train).most_common(1)[0][0]

    def scores(p: list[str]) -> dict:
        rank = {s: i for i, s in enumerate(SEVERITIES)}
        return {
            "accuracy": round(float(np.mean([a == b for a, b in zip(truth, p)])), 3),
            "macro_f1": round(float(f1_score(truth, p, average="macro", zero_division=0)), 3),
            "within_one_band": round(float(np.mean([abs(rank[a] - rank[b]) <= 1
                                                    for a, b in zip(truth, p)])), 3),
        }

    return {
        "train": len(train),
        "test": len(test),
        "test_distribution": dict(Counter(truth)),
        "model": scores(pred),
        "majority_baseline": {"predicts": majority, **scores([majority] * len(truth))},
        "confusion": {t: dict(Counter(p for tt, p in zip(truth, pred) if tt == t)) for t in SEVERITIES},
    }


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]{3,}", text.lower()))


def _ranking_metrics(sims: np.ndarray, truth_idx: list[int]) -> dict:
    ranks = []
    for row, t in zip(sims, truth_idx):
        ranks.append(int(np.sum(row > row[t])) + 1)
    ranks = np.array(ranks)
    return {
        "recall@1": round(float(np.mean(ranks <= 1)), 3),
        "recall@5": round(float(np.mean(ranks <= 5)), 3),
        "recall@10": round(float(np.mean(ranks <= 10)), 3),
        "mrr": round(float(np.mean(1.0 / ranks)), 3),
        "queries": int(len(ranks)),
    }


def _novelty(sims: np.ndarray, truth_idx: list[int]) -> dict:
    """Could the top score alone tell a duplicate from a new bug? Positive: the true advisory
    is in the index. Negative: the same query with its advisory removed."""
    pos = sims.max(axis=1)
    masked = sims.copy()
    masked[np.arange(len(truth_idx)), truth_idx] = -1
    neg = masked.max(axis=1)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    order = np.argsort(-s)
    tp = np.cumsum(y[order])
    precision = tp / np.arange(1, len(s) + 1)
    ok = np.where(precision >= 0.95)[0]
    best = ok.max() if len(ok) else None
    return {
        "roc_auc": round(float(roc_auc_score(y, s)), 3),
        "threshold_at_95pct_precision": None if best is None else round(float(s[order][best]), 3),
        "duplicate_recall_at_that_threshold": None if best is None else round(float(tp[best] / len(pos)), 3),
    }


def _precision_threshold(y: np.ndarray, s: np.ndarray, target: float = 0.95) -> tuple[float | None, float | None]:
    order = np.argsort(-s)
    tp = np.cumsum(y[order])
    precision = tp / np.arange(1, len(s) + 1)
    ok = np.where(precision >= target)[0]
    if not len(ok):
        return None, None
    best = ok.max()
    return float(s[order][best]), float(tp[best] / y.sum())


def _operating_points(y: np.ndarray, s: np.ndarray) -> dict:
    """ROC-AUC, duplicates caught at 95/90/80% precision, and the best-F1 threshold."""
    out: dict = {"roc_auc": round(float(roc_auc_score(y, s)), 3)}
    for target in (0.95, 0.9, 0.8):
        threshold, recall = _precision_threshold(y, s, target)
        out[f"at_{int(target * 100)}pct_precision"] = {
            "threshold": None if threshold is None else round(threshold, 3),
            "duplicate_recall": None if recall is None else round(recall, 3),
        }
    order = np.argsort(-s)
    tp = np.cumsum(y[order])
    k = np.arange(1, len(s) + 1)
    f1 = 2 * tp / (k + y.sum())
    best = int(np.argmax(f1))
    out["best_f1"] = {"f1": round(float(f1[best]), 3), "threshold": round(float(s[order][best]), 3),
                      "precision": round(float(tp[best] / k[best]), 3),
                      "duplicate_recall": round(float(tp[best] / y.sum()), 3)}
    return out


def eval_judge(sims: np.ndarray, reports: list[dict], truth: list[int], advisories: list[dict]) -> dict:
    """Out-of-fold duplicate/new calls, folds grouped by query so a query's positive and
    negative example never sit on both sides of a split."""
    packages = [a["package"] for a in advisories]
    X, y, groups = DuplicateJudge.training_pairs(sims, [r["text"] for r in reports], truth, packages)
    oof = cross_val_predict(LogisticRegression(max_iter=1000), X, y, groups=groups,
                            cv=GroupKFold(n_splits=5), method="predict_proba")[:, 1]
    # How often the best match after removing the true advisory is another advisory for the
    # same crate: the hard negatives that cap precision.
    same_crate = float(np.mean([X[i, 2] for i in range(1, len(X), 2)]))
    return {
        "features": ["top similarity", "margin over runner-up", "report names the crate", "top x named"],
        "new_bug_top_match_is_same_crate": round(same_crate, 3),
        "judge": _operating_points(y, oof),
        "top_score_only": _operating_points(y, X[:, 0]),
    }


def eval_dedup(advisories: list[dict], reports: list[dict], embeddings: bool) -> dict:
    by_id = {a["id"]: i for i, a in enumerate(advisories)}
    reports = [r for r in reports if r["rustsec_id"] in by_id]
    truth = [by_id[r["rustsec_id"]] for r in reports]
    overlap = [
        len(_tokens(r["text"]) & _tokens(text_of(advisories[t]))) /
        max(1, len(_tokens(r["text"]) | _tokens(text_of(advisories[t]))))
        for r, t in zip(reports, truth)
    ]
    independent = [i for i, o in enumerate(overlap) if o < 0.3]

    out: dict = {
        "index_size": len(advisories),
        "queries": len(reports),
        "independent_queries": len(independent),
        "median_token_jaccard_query_vs_advisory": round(float(np.median(overlap)), 3),
    }
    for variant in ("raw", "crate_name_masked"):
        if variant == "raw":
            corpus = [text_of(a) for a in advisories]
            queries = [r["text"] for r in reports]
        else:
            corpus = [mask_package(text_of(a), a["package"]) for a in advisories]
            queries = [mask_package(r["text"], advisories[t]["package"]) for r, t in zip(reports, truth)]

        dedup = Deduplicator().fit([a["id"] for a in advisories], corpus)
        sims = {"tfidf_word_char": dedup.similarity(queries)}
        if embeddings:
            from .models import EmbeddingDeduplicator

            emb = EmbeddingDeduplicator().fit([a["id"] for a in advisories], corpus)
            sims["minilm_embedding"] = emb.similarity(queries)
            sims["hybrid"] = combine(sims["tfidf_word_char"], sims["minilm_embedding"])

        if variant == "raw":
            out["duplicate_judge"] = eval_judge(sims["tfidf_word_char"], reports, truth, advisories)
        out[variant] = {}
        for name, s in sims.items():
            out[variant][name] = {
                "all": _ranking_metrics(s, truth),
                "independent_only": _ranking_metrics(s[independent], [truth[i] for i in independent]),
                "novelty": _novelty(s, truth),
            }
    return out


def to_markdown(m: dict) -> str:
    c, s, d = m["categories"], m["severity"], m["dedup"]
    lines = [
        "# Results",
        "",
        f"Generated by `python -m bounty_triage.evaluate` on RustSec advisory-db commit "
        f"`{m['source'].get('advisory_db_commit', '?')[:12]}`. Train: advisories before {SPLIT_DATE}. "
        "Test: everything after.",
        "",
        "## Vulnerability class (multi-label, 9 RustSec categories)",
        "",
        f"{c['train']} training advisories, {c['test']} test advisories.",
        "",
        "| | micro-F1 | macro-F1 | exact match | at least one label right |",
        "|---|---|---|---|---|",
    ]
    names = (
        ("keyword baseline", "keyword_baseline"),
        ("TF-IDF + logistic regression, threshold 0.5", "logreg_threshold_0.5"),
        ("… with per-class thresholds tuned by CV on train", "logreg_tuned_thresholds"),
        ("… plus keyword features (default)", "logreg_keywords_tuned"),
    )
    for name, key in names:
        r = c[key]
        lines.append(f"| {name} | {r['micro_f1']} | {r['macro_f1']} | {r['exact_match']} | {r['any_correct']} |")
    best = c["logreg_keywords_tuned"]
    lines += ["", "Per class, default model vs keyword baseline:", "",
              "| class | test support | tuned threshold | precision | recall | F1 | baseline F1 |",
              "|---|---|---|---|---|---|---|"]
    for cat in CATEGORIES:
        r, b = best["per_label"][cat], c["keyword_baseline"]["per_label"][cat]
        lines.append(f"| {cat} | {r['support']} | {best['thresholds'][cat]} | {r['precision']} | "
                     f"{r['recall']} | {r['f1']} | {b['f1']} |")

    lines += [
        "",
        "## Severity band from text (CVSS v3 base score bands)",
        "",
        f"{s['train']} training, {s['test']} test. Test distribution: {s['test_distribution']}.",
        "",
        "| | accuracy | macro-F1 | within one band |",
        "|---|---|---|---|",
        f"| model | {s['model']['accuracy']} | {s['model']['macro_f1']} | {s['model']['within_one_band']} |",
        f"| always `{s['majority_baseline']['predicts']}` | {s['majority_baseline']['accuracy']} | "
        f"{s['majority_baseline']['macro_f1']} | {s['majority_baseline']['within_one_band']} |",
        "",
        "## Duplicate detection (NVD CVE text → RustSec advisory)",
        "",
        f"{d['queries']} CVE descriptions searched against {d['index_size']} advisories. "
        f"Median token overlap between a CVE and its advisory: {d['median_token_jaccard_query_vs_advisory']}. "
        f"{d['independent_queries']} CVEs share under 30% of their tokens with the advisory "
        "(\"independent\").",
        "",
        "| variant | method | R@1 | R@5 | MRR | R@1 independent | novelty ROC-AUC | dup recall @95% precision |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for variant in ("raw", "crate_name_masked"):
        for name, r in d[variant].items():
            a, i, n = r["all"], r["independent_only"], r["novelty"]
            lines.append(
                f"| {variant} | {name} | {a['recall@1']} | {a['recall@5']} | {a['mrr']} | "
                f"{i['recall@1']} | {n['roc_auc']} | {n['duplicate_recall_at_that_threshold']} |"
            )
    j = d["duplicate_judge"]

    def row(name: str, r: dict) -> str:
        return (f"| {name} | {r['roc_auc']} | {r['at_95pct_precision']['duplicate_recall']} | "
                f"{r['at_90pct_precision']['duplicate_recall']} | {r['at_80pct_precision']['duplicate_recall']} | "
                f"{r['best_f1']['f1']} (P {r['best_f1']['precision']}, R {r['best_f1']['duplicate_recall']}) |")

    lines += [
        "",
        "### Duplicate or new?",
        "",
        "Every query is scored twice: once with its advisory in the index (a duplicate) and once with it "
        "removed (a new bug). Five-fold cross-validation, grouped by query. Cells are the share of duplicates "
        "caught while holding precision at the given level.",
        "",
        "| decision rule | ROC-AUC | at 95% precision | at 90% | at 80% | best F1 |",
        "|---|---|---|---|---|---|",
        row("top similarity alone", j["top_score_only"]),
        row("logistic regression: top score, margin, crate named", j["judge"]),
        "",
        f"When the bug is new, the best remaining match is another advisory for the same crate "
        f"{j['new_bug_top_match_is_same_crate']:.0%} of the time. Those look like duplicates on every feature "
        "here, which is what limits precision at the top.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--embeddings", action="store_true", help="also evaluate a MiniLM embedding model")
    args = ap.parse_args()

    advisories = load_jsonl_gz(args.data / "advisories.jsonl.gz")
    reports = load_jsonl_gz(args.data / "cve_reports.jsonl.gz")
    metrics = {
        "source": json.loads((args.data / "SOURCE.json").read_text()),
        "split_date": SPLIT_DATE,
        "categories": eval_categories(advisories),
        "severity": eval_severity(advisories),
        "dedup": eval_dedup(advisories, reports, args.embeddings),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.out / "RESULTS.md").write_text(to_markdown(metrics))
    print(to_markdown(metrics))


if __name__ == "__main__":
    main()
