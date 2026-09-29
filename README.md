# bounty-triage

Machine learning for the first hour of a security report's life. Given a report, it drafts a triage note: the likely vulnerability class, a severity suggestion, the closest known advisories, and whether the report is probably a duplicate. A person makes every decision; the [playbook](docs/TRIAGE_PLAYBOOK.md) says where the model's output fits.

Everything is trained and measured on real data: the 1,232 advisories in the [RustSec advisory database](https://github.com/rustsec/advisory-db), and the 384 CVE descriptions that NVD wrote independently for the same bugs (fetched from [OSV](https://osv.dev)).

```
$ python -m bounty_triage triage report.md
# Triage: Rojo's "rojo serve" HTTP API (default port 34872) has no Host/Origin header validation, ...

**Likely duplicate of RUSTSEC-2026-0279** (rojo; duplicate probability 0.88). Confirm and close as known, or explain what is new.
...
```

That report is the NVD text for a CVE; the model had never seen it. Full note: [reports/EXAMPLE.md](reports/EXAMPLE.md).

## Results

All numbers come from `python -m bounty_triage.evaluate` in CI, on the committed dataset snapshot. Full tables: [reports/RESULTS.md](reports/RESULTS.md).

**Setup.** Models train on advisories published before 2024 and are tested on everything after, which is how a triage queue meets reports. A random split would put near-identical advisories for the same crate on both sides and flatter the numbers.

**Vulnerability class** (9 RustSec categories, multi-label, 326 test advisories)

| | micro-F1 | macro-F1 | at least one label right |
|---|---|---|---|
| keyword rules | 0.603 | 0.534 | 0.61 |
| TF-IDF + logistic regression | 0.678 | 0.515 | 0.663 |
| + per-class thresholds tuned by cross-validation on train, + keyword features | **0.733** | **0.610** | **0.758** |

The plain model lost to keywords on rare classes (four test examples of thread-safety). Feeding the keyword hits in as features and tuning each class's threshold on training folds fixed that.

**Duplicate detection** (NVD's text searched against all 1,232 advisories)

| | right advisory first | in top 5 |
|---|---|---|
| TF-IDF, word + character n-grams | **0.883** | 0.969 |
| MiniLM sentence embeddings | 0.737 | 0.893 |
| same, crate names masked out of both texts | 0.852 / 0.612 | 0.943 / 0.786 |

Character n-grams win because security reports lean on identifiers like `SmallVec::insert_many` that word tokenisers and general-purpose embeddings split apart.

**Duplicate or new?** Ranking is not the same as deciding. Each query is scored with its advisory present and with it removed; the model must tell the two apart.

| rule | ROC-AUC | duplicates caught at 90% precision | at 80% |
|---|---|---|---|
| top similarity above a threshold | 0.878 | 0.26 | 0.81 |
| logistic regression on top similarity, margin over the runner-up, report names the crate | **0.912** | **0.72** | **0.89** |

Precision above 90% is hard: when a bug is new, the best remaining match is another advisory for the same crate 30% of the time. So the tool never auto-closes. It says **likely duplicate** at the 90%-precision point, **possible duplicate, compare first** at the best-F1 point, and **new** below that.

**Severity from text** is the honest negative result: 0.56 accuracy against 0.55 for always answering "high". It is shown as a placeholder only. Severity comes from reproducing the bug.

## Layout

| | |
|---|---|
| `bounty_triage/advisories.py` | RustSec advisory parser (TOML front matter + Markdown) |
| `bounty_triage/cvss.py` | CVSS v3.x base score from the FIRST specification, tested against reference vectors |
| `bounty_triage/models.py` | classifier, severity model, deduplicator, duplicate judge, keyword baseline |
| `bounty_triage/evaluate.py` | the evaluation above |
| `bounty_triage/fetch.py` | rebuilds `data/` (run by the `dataset` workflow, which commits a snapshot) |
| `docs/TRIAGE_PLAYBOOK.md` | intake to disclosure, with blockchain-specific impact levels |

```
pip install -e ".[dev]"            # add [embeddings] for the MiniLM comparison
pytest
python -m bounty_triage.evaluate --data data --out reports
python -m bounty_triage triage path/to/report.md
```

## Data and licences

`data/` is a snapshot of rustsec/advisory-db (CC0-1.0) at the commit in `data/SOURCE.json`, plus CVE descriptions from OSV.dev (CC-BY-4.0), which mirrors NVD. Code is Apache-2.0.
