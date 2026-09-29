from pathlib import Path

import pytest

from bounty_triage.advisories import load_advisory_db, parse_advisory
from bounty_triage.cvss import base_score, severity, severity_from_vector

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "vector,score",
    [
        ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", 9.8),
        ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H", 10.0),
        ("CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H", 7.8),
        ("CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:N/I:N/A:H", 5.9),
        ("CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N", 6.1),
        ("CVSS:3.0/AV:N/AC:L/PR:L/UI:N/S:C/C:H/I:H/A:H", 9.9),
        ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N", 0.0),
    ],
)
def test_cvss_base_scores_match_the_first_calculator(vector, score):
    assert base_score(vector) == score


def test_severity_bands():
    assert [severity(s) for s in (0.0, 3.9, 4.0, 6.9, 7.0, 8.9, 9.0)] == [
        None, "low", "medium", "medium", "high", "high", "critical",
    ]
    assert severity_from_vector("CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N") is None
    assert severity_from_vector("garbage") is None


def test_parses_a_real_advisory():
    adv = parse_advisory((FIXTURES / "RUSTSEC-2021-0003.md").read_text())
    assert adv.id == "RUSTSEC-2021-0003"
    assert adv.package == "smallvec"
    assert adv.title == "Buffer overflow in SmallVec::insert_many"
    assert adv.categories == ["memory-corruption"]
    assert "CVE-2021-25900" in adv.aliases
    assert adv.severity == "critical"
    # Markdown link targets are dropped, link text kept.
    assert "Yechan Bae (@Qwaz)" in adv.description
    assert "https://github.com/Qwaz" not in adv.description


def test_rejects_files_without_front_matter():
    with pytest.raises(ValueError):
        parse_advisory("# Just a title\n\nNo metadata.")


def test_loader_skips_withdrawn(tmp_path):
    crate = tmp_path / "crates" / "demo"
    crate.mkdir(parents=True)
    body = '```toml\n[advisory]\nid = "RUSTSEC-2099-{n}"\npackage = "demo"\ndate = "2099-01-01"\n{extra}```\n\n# T\n\nD\n'
    (crate / "RUSTSEC-2099-0001.md").write_text(body.format(n="0001", extra=""))
    (crate / "RUSTSEC-2099-0002.md").write_text(body.format(n="0002", extra='withdrawn = "2099-02-01"\n'))
    ids = [a.id for a in load_advisory_db(tmp_path)]
    assert ids == ["RUSTSEC-2099-0001"]
