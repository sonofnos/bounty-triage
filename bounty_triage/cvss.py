"""CVSS v3.0/v3.1 base score, implemented from the FIRST specification (section 7)."""

from __future__ import annotations

import math

_AV = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}
_AC = {"L": 0.77, "H": 0.44}
_PR_UNCHANGED = {"N": 0.85, "L": 0.62, "H": 0.27}
_PR_CHANGED = {"N": 0.85, "L": 0.68, "H": 0.5}
_UI = {"N": 0.85, "R": 0.62}
_CIA = {"H": 0.56, "L": 0.22, "N": 0.0}

SEVERITIES = ["low", "medium", "high", "critical"]


def _roundup(value: float) -> float:
    """The spec's Roundup: smallest one-decimal number >= value, robust to float error."""
    scaled = round(value * 100_000)
    if scaled % 10_000 == 0:
        return scaled / 100_000
    return (math.floor(scaled / 10_000) + 1) / 10.0


def base_score(vector: str) -> float:
    parts = vector.split("/")
    if not parts[0].startswith("CVSS:3"):
        raise ValueError(f"not a CVSS v3 vector: {vector}")
    m = dict(p.split(":", 1) for p in parts[1:])
    changed = m["S"] == "C"

    iss = 1 - (1 - _CIA[m["C"]]) * (1 - _CIA[m["I"]]) * (1 - _CIA[m["A"]])
    if changed:
        impact = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15
    else:
        impact = 6.42 * iss
    pr = (_PR_CHANGED if changed else _PR_UNCHANGED)[m["PR"]]
    exploitability = 8.22 * _AV[m["AV"]] * _AC[m["AC"]] * pr * _UI[m["UI"]]

    if impact <= 0:
        return 0.0
    if changed:
        return _roundup(min(1.08 * (impact + exploitability), 10))
    return _roundup(min(impact + exploitability, 10))


def severity(score: float) -> str | None:
    if score == 0:
        return None
    if score < 4.0:
        return "low"
    if score < 7.0:
        return "medium"
    if score < 9.0:
        return "high"
    return "critical"


def severity_from_vector(vector: str) -> str | None:
    try:
        return severity(base_score(vector))
    except (ValueError, KeyError):
        return None
