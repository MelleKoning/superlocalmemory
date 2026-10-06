# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A derived fact must still say what its source memory said.

Synthetic, de-identified fixtures. Each damaged case is something a
paraphrasing model was seen to do, or would plausibly do; each faithful case
is a paraphrase that must NOT be flagged, so genuine extraction keeps working.
"""

from __future__ import annotations

import pytest

from superlocalmemory.encoding.source_fidelity import (
    check_fact_against_source,
    date_is_supported,
)

KESTREL = "The Kestrel checkpoint was recalled at rank 1 in 2004.6 ms."
POLICY = "Never publish without approval from the release owner."
ORDER = "Ship the iOS build before the Android build."
STATUS = "Previously the team used Jenkins, currently it uses GitHub Actions."

DAMAGED = [
    (KESTREL, "The Kestrel checkpoint was recalled at rank 1 on 2004-06-16.", "number_became_date"),
    (KESTREL, "The recall happened on June 1st, 2004", "number_became_date"),
    (KESTREL, "The recall occurred in 2004", "number_became_date"),
    ("Release v4.1.21 shipped.", "The release shipped on 2021-04-01.", "number_became_date"),
    (KESTREL, "The Kestrel checkpoint was recalled at rank 1 in 2014.6 ms.", "unsupported_number"),
    ("Costs rose 45% to $12.50.", "Costs rose 54% to $12.50.", "unsupported_number"),
    ("The meeting moved to the library.", "On 2026-10-06 the meeting moved to the library.",
     "unsupported_date"),
    ("The launch is on 6 Oct 2026.", "The launch is on 7 October 2026.", "unsupported_date"),
    (POLICY, "Publish with approval from the release owner.", "negation_lost"),
    ("We do not use Jenkins; we use GitHub Actions.", "The team uses Jenkins.", "negation_lost"),
    (ORDER, "Ship the Android build before the iOS build.", "order_reversed"),
    ("Ship iOS before Android.", "Ship Android before iOS.", "order_reversed"),
    (STATUS, "The team uses Jenkins for builds.", "status_lost"),
]

FAITHFUL = [
    (KESTREL, "The Kestrel checkpoint was recalled at rank 1 in 2004.6 milliseconds."),
    (KESTREL, "The Kestrel checkpoint recall took 2,004.6 ms."),
    ("Release v4.1.21 shipped on 2021-04-01.", "The release shipped on 2021-04-01."),
    ("I went to the gym yesterday.", "Caroline went to the gym on 2026-10-05."),
    ("The launch is on 6 Oct 2026.", "The launch is on 2026-10-06."),
    ("The launch is on 06.10.2026.", "The launch is on 2026-10-06."),
    ("The launch is on March 15.", "The launch is on 2026-03-15."),
    ("We met in June 2025.", "The team met in June 2025."),
    ("Last year the team moved.", "In 2025 the team moved."),
    (POLICY, "The team must never publish without the release owner approving."),
    (ORDER, "Ship the Android build after the iOS build."),
    (ORDER, "The iOS build ships before the Android build."),
    (STATUS, "The team uses GitHub Actions."),
    (STATUS, "The team previously used Jenkins."),
    ("We do not use Jenkins; we use GitHub Actions.", "The team uses GitHub Actions."),
    ("No, the build uses GitHub Actions.", "The build uses GitHub Actions."),
    ("Costs rose 45% to $12.50 on port 8765 (#149).",
     "Costs rose 45% to $12.5 on port 8765 (#149)."),
    ("We had 1,234 users.", "The product had 1234 users."),
    ("We have 1,999 users.", "The product has 1999 users."),
    ("Recall took 3.0 s at p95.", "Recall took 3 s at p95."),
    ("Commit 8f3a2b1 fixed it in 3s.", "The fix in commit 8f3a2b1 took 3s."),
    ("Alpha uses SQLite. Beta uses Postgres.", "Beta relies on PostgreSQL"),
]


@pytest.mark.parametrize(("source", "fact", "reason"), DAMAGED)
def test_a_change_of_meaning_is_flagged(source: str, fact: str, reason: str) -> None:
    report = check_fact_against_source(fact, source)
    assert not report.ok
    assert reason in report.reasons, report.reasons


@pytest.mark.parametrize(("source", "fact"), FAITHFUL)
def test_a_faithful_paraphrase_is_not_flagged(source: str, fact: str) -> None:
    report = check_fact_against_source(fact, source)
    assert report.ok, report.reasons


def test_an_exact_span_is_always_faithful() -> None:
    assert check_fact_against_source("recalled at rank 1 in 2004.6 ms", KESTREL).ok
    assert check_fact_against_source(KESTREL, KESTREL).ok


def test_empty_inputs_are_not_flagged() -> None:
    assert check_fact_against_source("", KESTREL).ok
    assert check_fact_against_source("anything", "").ok


@pytest.mark.parametrize(("iso", "source", "supported"), [
    ("2004-06-01", KESTREL, False),
    ("2026-10-06", "We met today.", True),
    ("2026-10-06", "We met.", False),
    ("2026-03-15", "Launch March 15.", True),
    (None, "anything", True),
])
def test_date_field_support(iso: str | None, source: str, supported: bool) -> None:
    assert date_is_supported(iso, source) is supported
