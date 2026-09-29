import pytest

pytest.importorskip("sklearn")

from bounty_triage.__main__ import triage  # noqa: E402
from bounty_triage.models import (  # noqa: E402
    CategoryClassifier,
    Deduplicator,
    KeywordClassifier,
    mask_package,
)


def adv(i, package, title, description, categories, severity="high", date="2022-01-01"):
    return {
        "id": f"RUSTSEC-2022-{i:04d}", "package": package, "title": title, "description": description,
        "categories": categories, "severity": severity, "date": date,
    }


CORPUS = [
    adv(1, "smallvec", "Buffer overflow in SmallVec::insert_many",
        "insert_many wrote past the end of the heap buffer when size_hint lied", ["memory-corruption"]),
    adv(2, "hyper", "Unbounded header parsing causes denial of service",
        "a malicious peer can send endless headers and exhaust memory, crashing the server",
        ["denial-of-service"], "medium"),
    adv(3, "ring", "Timing side channel in signature verification",
        "comparison is not constant-time so an attacker can recover the key through timing",
        ["crypto-failure"], "medium"),
    adv(4, "tar", "Path traversal when unpacking archives",
        "entries with ../ components are written outside the destination directory",
        ["file-disclosure"]),
    adv(5, "arrayvec", "Use after free in ArrayVec::drain",
        "drain keeps a dangling pointer to freed memory and reads it after free", ["memory-corruption"]),
    adv(6, "tokio", "Data race in shared task queue",
        "the queue implements Send and Sync without synchronisation, causing a data race between threads",
        ["thread-safety"], "medium"),
] * 3


def test_mask_package_hides_both_spellings():
    text = "The serde-json crate (serde_json) mis-parses; Serde-Json users should upgrade."
    assert mask_package(text, "serde-json") == (
        "The <crate> crate (<crate>) mis-parses; <crate> users should upgrade."
    )


def test_keyword_baseline():
    got = KeywordClassifier().predict(["a use-after-free leads to a crash"])[0]
    assert "memory-corruption" in got and "denial-of-service" in got


def test_classifier_learns_categories_and_always_answers():
    texts = [f"{a['title']} {a['description']}" for a in CORPUS]
    model = CategoryClassifier().fit(texts, [a["categories"] for a in CORPUS])
    assert model.predict(["heap buffer overflow writes past the end"])[0] == ["memory-corruption"]
    assert model.predict(["zzz qqq"])[0], "must always return at least one label"


def test_dedup_finds_the_same_bug_described_differently():
    unique = CORPUS[:6]
    dedup = Deduplicator().fit([a["id"] for a in unique], [a["title"] + " " + a["description"] for a in unique])
    best = dedup.query("An issue was discovered in the tar crate: archive extraction allows "
                       "directory traversal via ../ paths", k=3)
    assert best[0].id == "RUSTSEC-2022-0004"


def test_triage_note_flags_duplicates_and_new_reports():
    note = triage("Buffer overflow in SmallVec::insert_many: insert_many writes past the end of the heap buffer",
                  CORPUS, threshold=0.3)
    assert "Likely duplicate of RUSTSEC-2022-0001" in note
    assert "memory-corruption" in note
    note = triage("Governance proposal enactment ignores the configured delay", CORPUS, threshold=0.3)
    assert "No close match" in note
