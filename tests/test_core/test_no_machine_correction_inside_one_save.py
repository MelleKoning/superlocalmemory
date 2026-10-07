# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A save never files a machine correction between two of its own facts.

A pending machine correction withholds its newer fact from recall until a
person reviews it. On a 22k-fact store 37% of the machine cases paired two
facts of the SAME save (one memory's sentences consolidated against each
other), so a memory hid part of itself. A correction of an OLDER memory is
still proposed, including one the save's duplicate merged into.
"""

from __future__ import annotations


def _actor() -> str:
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id

    return local_trusted_actor_id("python-api")


def _save(engine, text: str) -> list[str]:
    from superlocalmemory.core.engine_ingestion import canonical_store

    receipt = canonical_store(engine, text, source_type="python-api", trusted_actor_id=_actor(),
                              require_complete=True, return_receipt=True)
    return list(receipt.final_fact_ids)


def _script(engine, monkeypatch, choose) -> list[str]:
    """Consolidation that proposes an update of ``choose(...)`` when it names one."""
    cons = engine._consolidator
    real = cons.consolidate
    proposed: list[str] = []

    def consolidate(new_fact, profile_id, *, exclude_fact_ids=None, pending_fact_ids=()):
        target = choose(new_fact, tuple(pending_fact_ids))
        if target is None:
            return real(new_fact, profile_id, exclude_fact_ids=exclude_fact_ids,
                        pending_fact_ids=pending_fact_ids)
        proposed.append(target)
        return cons._execute_update(new_fact, engine._db.get_fact(target), profile_id,
                                    reason="scripted")

    monkeypatch.setattr(cons, "consolidate", consolidate)
    return proposed


def _cases(engine) -> int:
    return int(dict(engine._db.execute("SELECT COUNT(*) AS n FROM correction_cases")[0])["n"])


_TWO = ("Brindlemoor moved the synthetic pewter kettle to the third shelf. "
        "Brindlemoor later moved the synthetic pewter kettle to the fourth shelf.")


def test_no_case_between_two_facts_of_one_save(engine_with_mock_deps, monkeypatch) -> None:
    engine = engine_with_mock_deps
    proposed = _script(engine, monkeypatch, lambda fact, pending: pending[0] if pending else None)
    _save(engine, _TWO)
    assert proposed, "not vacuous: a same-save update was proposed"
    assert _cases(engine) == 0


def test_a_correction_of_an_older_memory_is_still_filed(engine_with_mock_deps,
                                                         monkeypatch) -> None:
    engine = engine_with_mock_deps
    older = _save(engine, "Hollowmere keeps the synthetic brass lantern on the porch.")
    proposed = _script(engine, monkeypatch,
                       lambda fact, pending: older[0] if not pending else None)
    _save(engine, "Hollowmere now keeps the synthetic brass lantern in the shed.")
    assert proposed == [older[0]]
    assert _cases(engine) == 1
