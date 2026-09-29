"""Build the dataset: RustSec advisories plus the independently written CVE text for each.

    python -m bounty_triage.fetch --advisory-db path/to/advisory-db --out data/

CVE descriptions come from OSV (`api.osv.dev/v1/vulns/CVE-...`), which mirrors NVD. They
are written by the CVE numbering authority, not by RustSec, so they serve as a second,
independent report of the same bug: the ground truth for duplicate detection.
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .advisories import load_advisory_db

OSV = "https://api.osv.dev/v1/vulns/{}"


def _osv(cve: str) -> dict | None:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(OSV.format(cve), timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2**attempt)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2**attempt)
    return None


def write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--advisory-db", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    commit = subprocess.run(
        ["git", "-C", str(args.advisory_db), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    advisories = load_advisory_db(args.advisory_db)
    write_jsonl_gz(args.out / "advisories.jsonl.gz", [a.to_dict() for a in advisories])

    wanted = {
        alias: adv.id
        for adv in advisories
        for alias in adv.aliases
        if alias.startswith("CVE-")
    }
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = dict(zip(wanted, pool.map(_osv, wanted)))

    reports = []
    for cve, record in sorted(records.items()):
        text = (record or {}).get("details", "").strip()
        if text:
            reports.append({"id": cve, "rustsec_id": wanted[cve], "text": text,
                            "summary": (record or {}).get("summary", "")})
    write_jsonl_gz(args.out / "cve_reports.jsonl.gz", reports)

    meta = {
        "advisory_db_commit": commit,
        "advisories": len(advisories),
        "cve_aliases": len(wanted),
        "cve_reports_with_text": len(reports),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.out / "SOURCE.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
