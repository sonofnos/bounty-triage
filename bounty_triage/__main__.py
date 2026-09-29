"""Triage an incoming security report.

    python -m bounty_triage triage report.md
    cat report.txt | python -m bounty_triage triage -

Prints a Markdown triage note: likely vulnerability class, a suggested severity band, and the
closest known advisories with a duplicate call. The models are fitted on the bundled dataset
at start-up (a few seconds); every output is a suggestion for the human triager.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

from .models import (
    CATEGORIES,
    CategoryClassifier,
    Deduplicator,
    DuplicateJudge,
    SeverityModel,
    load_jsonl_gz,
    text_of,
)

# Fallbacks when reports/metrics.json is missing: (likely duplicate, possible duplicate).
DEFAULT_JUDGE_THRESHOLDS = (0.9, 0.5)
DEFAULT_SCORE_THRESHOLDS = (0.6, 0.35)


def _judge_thresholds(reports_dir: Path) -> tuple[float, float]:
    """"Likely duplicate" at the 90%-precision operating point measured in evaluation,
    "possible duplicate" at the best-F1 point."""
    try:
        judge = json.loads((reports_dir / "metrics.json").read_text())["dedup"]["duplicate_judge"]["judge"]
        likely = judge["at_90pct_precision"]["threshold"]
        possible = judge["best_f1"]["threshold"]
        return (float(likely) if likely is not None else DEFAULT_JUDGE_THRESHOLDS[0], float(possible))
    except (OSError, KeyError, ValueError, TypeError):
        return DEFAULT_JUDGE_THRESHOLDS


@dataclass
class Models:
    categories: CategoryClassifier
    severity: SeverityModel
    dedup: Deduplicator
    judge: DuplicateJudge | None
    advisories: list[dict]


def fit_models(advisories: list[dict], cve_reports: list[dict] | None = None) -> Models:
    classified = [a for a in advisories if a["categories"]]
    categories = CategoryClassifier(tune_thresholds=True, keywords=True).fit(
        [text_of(a) for a in classified], [a["categories"] for a in classified]
    )
    rated = [a for a in advisories if a.get("severity")]
    severity = SeverityModel().fit([text_of(a) for a in rated], [a["severity"] for a in rated])
    dedup = Deduplicator().fit([a["id"] for a in advisories], [text_of(a) for a in advisories])

    judge = None
    index = {a["id"]: i for i, a in enumerate(advisories)}
    pairs = [r for r in (cve_reports or []) if r["rustsec_id"] in index]
    if pairs:
        sims = dedup.similarity([r["text"] for r in pairs])
        X, y, _ = DuplicateJudge.training_pairs(
            sims, [r["text"] for r in pairs], [index[r["rustsec_id"]] for r in pairs],
            [a["package"] for a in advisories],
        )
        judge = DuplicateJudge().fit(X, y)
    return Models(categories, severity, dedup, judge, advisories)


def triage(report: str, advisories: list[dict], thresholds: tuple[float, float] | None = None, k: int = 5,
           models: Models | None = None) -> str:
    """Triage note for one report. `thresholds` are (likely, possible) cut-offs on the duplicate
    probability when a judge is fitted, or on raw similarity when not. The tool never closes a
    report on its own: the best it says is "likely duplicate, confirm"."""
    models = models or fit_models(advisories)
    categories, severity, dedup = models.categories, models.severity, models.dedup
    by_id = {a["id"]: a for a in advisories}

    probs = categories.predict_proba([report])[0]
    ranked = sorted(zip(CATEGORIES, probs), key=lambda x: -x[1])
    labels = categories.predict([report])[0]
    sev = severity.predict_proba([report])[0]
    top_sev = max(sev, key=sev.get)
    matches = dedup.query(report, k)
    sims = dedup.similarity([report])[0]
    if models.judge is not None:
        likely, possible = thresholds or DEFAULT_JUDGE_THRESHOLDS
        confidence = models.judge.probability(sims, report, [a["package"] for a in advisories])
        basis = f"duplicate probability {confidence:.2f}"
    else:
        likely, possible = thresholds or DEFAULT_SCORE_THRESHOLDS
        confidence = matches[0].score
        basis = f"similarity {confidence:.2f}"

    title = next((line.strip("# ").strip() for line in report.splitlines() if line.strip()), "report")
    lines = [f"# Triage: {title[:100]}", ""]
    best = matches[0]
    adv = by_id[best.id]
    if confidence >= likely:
        lines += [f"**Likely duplicate of {best.id}** ({adv['package']}; {basis}). "
                  "Confirm and close as known, or explain what is new.", ""]
    elif confidence >= possible:
        lines += [f"**Possible duplicate of {best.id}** ({adv['package']}; {basis}). Compare the two "
                  "before treating this as new: it may be a different bug in the same component.", ""]
    else:
        lines += [f"**No close match** among {len(advisories)} known advisories ({basis}). Treat as new.", ""]

    lines += ["## Suggested class", ""]
    lines += [f"- {c}: {p:.2f}{'  <-' if c in labels else ''}" for c, p in ranked[:4]]
    lines += ["", "## Suggested severity", "",
              f"{top_sev} ({sev[top_sev]:.2f}). From text alone; set the real value from a CVSS vector "
              "or the programme's impact table after reproducing.", "",
              "## Closest known advisories", "", "| advisory | crate | similarity | title |", "|---|---|---|---|"]
    for m in matches:
        a = by_id[m.id]
        lines.append(f"| {m.id} | {a['package']} | {m.score:.2f} | {a['title'][:70]} |")
    lines += ["", "## Next steps", "",
              "- [ ] Reproduce on the affected version; record exact commit and steps",
              "- [ ] Confirm impact and severity; agree it with the reporter",
              "- [ ] Assign an owner and a fix deadline from the severity SLA",
              "- [ ] Coordinate disclosure date; credit the reporter"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(prog="bounty_triage")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("triage", help="triage one report (file path or - for stdin)")
    t.add_argument("report")
    t.add_argument("--data", type=Path, default=Path("data"))
    t.add_argument("--reports", type=Path, default=Path("reports"))
    t.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    text = sys.stdin.read() if args.report == "-" else Path(args.report).read_text(encoding="utf-8")
    models = _load_or_fit(args.data)
    print(triage(text, models.advisories, _judge_thresholds(args.reports), args.k, models), end="")


def _load_or_fit(data: Path) -> Models:
    """Fitting takes a minute (threshold tuning cross-validates), so cache it next to the data.
    The cache is only reused while it is newer than every data file."""
    cache = data / ".models.pkl"
    sources = [data / "advisories.jsonl.gz", data / "cve_reports.jsonl.gz"]
    newest = max(p.stat().st_mtime for p in sources if p.exists())
    if cache.exists() and cache.stat().st_mtime > newest:
        with cache.open("rb") as f:
            return pickle.load(f)
    reports_path = data / "cve_reports.jsonl.gz"
    models = fit_models(load_jsonl_gz(data / "advisories.jsonl.gz"),
                        load_jsonl_gz(reports_path) if reports_path.exists() else None)
    with cache.open("wb") as f:
        pickle.dump(models, f)
    return models


if __name__ == "__main__":
    main()
