"""Parse RustSec advisories into flat records.

Each advisory in rustsec/advisory-db is a Markdown file that opens with a fenced TOML block
(`[advisory]`, `[versions]`, ...) followed by `# Title` and a free-text description.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .cvss import severity_from_vector

_FRONT_MATTER = re.compile(r"\A```toml\s*\n(.*?)\n```\s*\n(.*)\Z", re.S)


@dataclass
class Advisory:
    id: str
    package: str
    date: str
    title: str
    description: str
    categories: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    informational: str | None = None
    cvss: str | None = None
    severity: str | None = None
    withdrawn: bool = False

    @property
    def text(self) -> str:
        return f"{self.title}\n\n{self.description}"

    def to_dict(self) -> dict:
        return asdict(self)


def parse_advisory(markdown: str) -> Advisory:
    match = _FRONT_MATTER.match(markdown.lstrip("﻿"))
    if not match:
        raise ValueError("advisory does not start with a ```toml front matter block")
    meta = tomllib.loads(match.group(1))["advisory"]
    body = match.group(2).strip()

    title, _, description = body.partition("\n")
    title = title.lstrip("#").strip()
    cvss = meta.get("cvss")
    return Advisory(
        id=meta["id"],
        package=meta.get("package", ""),
        date=str(meta.get("date", "")),
        title=title,
        description=_clean(description),
        categories=list(meta.get("categories", [])),
        keywords=list(meta.get("keywords", [])),
        aliases=list(meta.get("aliases", [])),
        informational=meta.get("informational"),
        cvss=cvss,
        severity=severity_from_vector(cvss) if cvss else None,
        withdrawn="withdrawn" in meta,
    )


def _clean(text: str) -> str:
    """Drop Markdown link targets and collapse whitespace; keep the words."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    return re.sub(r"\s+", " ", text).strip()


def load_advisory_db(root: Path) -> list[Advisory]:
    """Load every crate advisory under `<root>/crates`, skipping withdrawn ones."""
    out = []
    for path in sorted((root / "crates").glob("*/RUSTSEC-*.md")):
        adv = parse_advisory(path.read_text(encoding="utf-8"))
        if not adv.withdrawn:
            out.append(adv)
    return out
