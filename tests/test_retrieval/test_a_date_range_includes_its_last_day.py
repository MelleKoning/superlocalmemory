# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A date-only range covers its whole last day (R6).

"2026-07-01..2026-07-31" read its end as 2026-07-31 00:00, so everything that
happened on 31 July after midnight was outside "July", and a one-day range
("2026-10-04..2026-10-04") held only the instant of midnight.
"""

from __future__ import annotations

import pytest

from superlocalmemory.retrieval.time_window import in_window, parse_window


@pytest.mark.parametrize("spec", [
    "2026-07-01..2026-07-31",
    "2026-07-01,2026-07-31",
    ("2026-07-01", "2026-07-31"),
    "2026-07-31..2026-07-01",          # written backwards
])
def test_the_last_day_is_inside(spec) -> None:
    bounds = parse_window(spec)
    assert in_window("2026-07-31 18:30:00", bounds)
    assert in_window("2026-07-31T23:59:59.999999Z", bounds)
    assert in_window("2026-07-01 00:00:00", bounds)
    assert not in_window("2026-08-01 00:00:00", bounds)
    assert not in_window("2026-06-30 23:59:59", bounds)


def test_a_one_day_range_is_that_whole_day() -> None:
    bounds = parse_window("2026-10-04..2026-10-04")
    assert in_window("2026-10-04 12:00:00", bounds)
    assert not in_window("2026-10-05 00:00:00", bounds)


def test_an_end_with_a_time_is_taken_exactly() -> None:
    bounds = parse_window("2026-07-01..2026-07-31T00:00:00Z")
    assert in_window("2026-07-31 00:00:00", bounds)
    assert not in_window("2026-07-31 18:30:00", bounds)
