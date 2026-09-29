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
import sys
from pathlib import Path

from .models import (
    CATEGORIES,
    CategoryClassifier,
    Deduplicator,
    SeverityModel,
    load_jsonl_gz,
    text_of,
)

# Used when reports/metrics.json is missing: the similarity above which past evaluation kept
# duplicate calls at >= 95% precision.
DEFAULT_DUPLICATE_THRESHOLD = 0.35


def _threshold(reports_dir: Path) -> float:
    try:
        metrics = json.loads((reports_dir / "metrics.json").read_text())
        value = metrics["dedup"]["raw"]["tfidf_word_char"]["novelty"]["threshold_at_95pct_precision"]
        return float(value) if value is not None else DEFAULT_DUPLICATE_THRESHOLD
    except (OSError, KeyError, ValueError, TypeError):
        return DEFAULT_DUPLICATE_THRESHOLD


def triage(report: str, advisories: list[dict], threshold: float, k: int = 5) -> str:
    classified = [a for a in advisories if a["categories"]]
    categories = CategoryClassifier().fit(
        [text_of(a) for a in classified], [a["categories"] for a in classified]
    )
    rated = [a for a in advisories if a.get("severity")]
    severity = SeverityModel().fit([text_of(a) for a in rated], [a["severity"] for a in rated])
    dedup = Deduplicator().fit([a["id"] for a in advisories], [text_of(a) for a in advisories])
    by_id = {a["id"]: a for a in advisories}

    probs = categories.predict_proba([report])[0]
    ranked = sorted(zip(CATEGORIES, probs), key=lambda x: -x[1])
    labels = categories.predict([report])[0]
    sev = severity.predict_proba([report])[0]
    top_sev = max(sev, key=sev.get)
    matches = dedup.query(report, k)

    title = next((line.strip("# ").strip() for line in report.splitlines() if line.strip()), "report")
    lines = [f"# Triage: {title[:100]}", ""]
    best = matches[0]
    if best.score >= threshold:
        adv = by_id[best.id]
        lines += [f"**Likely duplicate of {best.id}** ({adv['package']}, similarity {best.score:.2f} "
                  f">= {threshold:.2f}). Confirm and close as known, or explain what is new.", ""]
    else:
        lines += [f"**No close match** among {len(advisories)} known advisories "
                  f"(best {best.score:.2f} < {threshold:.2f}). Treat as new.", ""]

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
    advisories = load_jsonl_gz(args.data / "advisories.jsonl.gz")
    print(triage(text, advisories, _threshold(args.reports), args.k), end="")


if __name__ == "__main__":
    main()
