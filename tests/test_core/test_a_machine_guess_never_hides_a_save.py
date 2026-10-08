# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""SLM's own unreviewed guess that a save updates an older memory hides nothing.

When consolidation (or the temporal check) guesses that a new save updates an
older memory, the save path files a correction CASE for a person to review.
Recall used to withhold the case's successor -- the memory the caller had just
saved -- until that review: on a fresh store 7 of 27 new notes became
unfindable after enrichment. Varun's decision (2026-10-08): until reviewed, both
memories stay findable; approving retires the older one; rejecting keeps both.
A proposed content correction (its successor exists only because of the
proposal) is still withheld until reviewed.
"""

from __future__ import annotations

from tests.test_core.test_no_machine_correction_inside_one_save import _cases, _save, _script


def _withheld(engine, fact_ids: list[str]) -> set[str]:
    db = engine._db
    return (db.get_correction_inadmissible_fact_ids(list(fact_ids), "default")
            | db.get_nonapplied_correction_successor_ids(list(fact_ids), "default"))


def test_a_save_the_machine_thinks_updates_an_older_one_stays_findable(
        engine_with_mock_deps, monkeypatch) -> None:
    engine = engine_with_mock_deps
    older = _save(engine, "Hollowmere keeps the synthetic brass lantern on the porch.")
    _script(engine, monkeypatch, lambda fact, pending: older[0] if not pending else None)
    newer = _save(engine, "Hollowmere now keeps the synthetic brass lantern in the shed.")
    assert _cases(engine) == 1, "not vacuous: a machine case was filed"
    assert _withheld(engine, newer + older) == set()


def test_a_rejected_machine_guess_keeps_both(engine_with_mock_deps, monkeypatch) -> None:
    engine = engine_with_mock_deps
    older = _save(engine, "Kestrel Harbour stores the synthetic copper ledger upstairs.")
    _script(engine, monkeypatch, lambda fact, pending: older[0] if not pending else None)
    newer = _save(engine, "Kestrel Harbour now stores the synthetic copper ledger downstairs.")
    engine._db.execute("UPDATE correction_cases SET status='rejected'")
    assert _withheld(engine, newer + older) == set()
