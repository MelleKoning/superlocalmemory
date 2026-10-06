# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Measurements are never handed to a date parser; genuine dates still parse.

4.1.21 fuzzy-parsed any string: "4.1.21" became 1 April 2021, "port 8765"
the year 8765, "$12.50" the 12th of this month, "1,234" the year 234.
"""

from __future__ import annotations

import pytest

from superlocalmemory.encoding.date_resolution import keep_supported_dates, try_parse_date
from superlocalmemory.encoding.fact_extractor import _try_parse_date
from superlocalmemory.encoding.temporal_parser import TemporalParser
from superlocalmemory.encoding.typed_values import (
    is_typed_measurement,
    looks_like_calendar_date,
    typed_number_cores,
)
from superlocalmemory.storage.models import AtomicFact

REF = "2026-10-06"  # a Tuesday

MEASUREMENTS = [
    "2004.6 ms", "2004.6", "1,234", "45%", "3.0 s", "$12.50", "port 8765",
    "v4.1.21", "4.1.21", "v2.3", "#149", "8765", "12.5", "2023",
]

GENUINE_DATES = [
    ("2026-10-06", "2026-10-06"),
    ("6 Oct 2026", "2026-10-06"),
    ("March 15, 2026", "2026-03-15"),
    ("3/15/2026", "2026-03-15"),
    ("next Tuesday", "2026-10-13"),
    ("last Monday", "2026-10-05"),
    ("yesterday", "2026-10-05"),
    ("in 2 weeks", "2026-10-20"),
    ("3 days ago", "2026-10-03"),
]


@pytest.mark.parametrize("token", MEASUREMENTS)
def test_a_measurement_is_typed_and_is_not_a_date(token: str) -> None:
    assert is_typed_measurement(token)
    assert not looks_like_calendar_date(token)
    assert try_parse_date(token, REF) is None
    assert _try_parse_date(token, REF) is None, "the extractor uses the same guard"


@pytest.mark.parametrize(("raw", "expected"), GENUINE_DATES)
def test_genuine_dates_still_parse(raw: str, expected: str) -> None:
    assert looks_like_calendar_date(raw)
    assert try_parse_date(raw, REF) == expected


def test_a_dotted_date_is_a_date_but_a_version_is_not() -> None:
    assert looks_like_calendar_date("06.10.2026")
    assert not is_typed_measurement("06.10.2026")
    assert is_typed_measurement("4.1.21") and not looks_like_calendar_date("4.1.21")


def test_number_words_without_a_unit_are_not_dates() -> None:
    assert try_parse_date("next 3", REF) is None
    assert try_parse_date("Q3", REF) is None


def test_typed_number_cores_compare_values_not_spelling() -> None:
    text = "Recall 2004.6 ms, cost $12.50, 45% of 1,234 on port 8765, v4.1.21 (#149)."
    assert typed_number_cores(text) == {
        "2004.6", "12.5", "45", "1234", "8765", "4.1.21", "149",
    }
    # a hex id or "rank 1" is not a measurement
    assert typed_number_cores("commit 8f3a2b1 at rank 1") == frozenset()


class TestTemporalParserNeverDatesAMeasurement:
    def test_measurement_sentence_has_no_dates(self) -> None:
        found = TemporalParser(REF).extract_dates_from_text(
            "The Kestrel checkpoint was recalled at rank 1 in 2004.6 ms.")
        assert found == {"referenced_date": None, "interval_start": None,
                         "interval_end": None}

    def test_a_range_of_measurements_is_not_an_interval(self) -> None:
        found = TemporalParser(REF).extract_dates_from_text("Latency went from 1,234 to 2,000 ms.")
        assert found["interval_start"] is None and found["interval_end"] is None

    def test_versions_ports_money_and_ids_have_no_dates(self) -> None:
        found = TemporalParser(REF).extract_dates_from_text(
            "Release v4.1.21 on port 8765 cost $12.50 (#149).")
        assert found["referenced_date"] is None

    def test_a_genuine_interval_still_parses(self) -> None:
        found = TemporalParser(REF).extract_dates_from_text(
            "The freeze runs from March 1 to March 5, 2026.")
        assert found["interval_start"].startswith("2026-03-01")
        assert found["interval_end"].startswith("2026-03-05")


class TestModelDateFields:
    """A model's date field survives only when the source supports it."""

    @staticmethod
    def _fact(referenced: str | None) -> AtomicFact:
        return AtomicFact(fact_id="f1", content="The Kestrel checkpoint was recalled at rank 1",
                          referenced_date=referenced)

    def test_a_date_read_out_of_a_measurement_is_cleared(self) -> None:
        source = "The Kestrel checkpoint was recalled at rank 1 in 2004.6 ms."
        [fact] = keep_supported_dates([self._fact("2004-06-01")], source)
        assert fact.referenced_date is None
        assert fact.content == "The Kestrel checkpoint was recalled at rank 1"

    def test_the_conversation_date_is_not_an_event_date(self) -> None:
        [fact] = keep_supported_dates([self._fact("2026-10-06")], "The checkpoint moved.")
        assert fact.referenced_date is None

    def test_a_resolved_relative_date_is_kept(self) -> None:
        [fact] = keep_supported_dates([self._fact("2026-10-05")], "It moved yesterday.")
        assert fact.referenced_date == "2026-10-05"

    def test_a_stated_date_is_kept(self) -> None:
        [fact] = keep_supported_dates([self._fact("2026-10-06")], "It moved on 6 Oct 2026.")
        assert fact.referenced_date == "2026-10-06"
