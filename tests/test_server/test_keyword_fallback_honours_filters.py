# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The over-budget keyword fallback (``/recall``'s last-resort LIKE search)
must honour the same hard filters full recall does: project/agent/about/kind
facets, quarantine exclusion, and caller-replacement (temporal validity)
exclusion.

4.1.19 L2-09 / L3-10: before this fix the fallback read straight off
``atomic_facts`` with no clause at all and no ``facets`` parameter, so a
caller who asked for ``project=zephyr`` or ``kind=status`` could be handed a
quarantined row from another project with no facet applied and no word about
it in the envelope.
"""

from __future__ import annotations

from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord


def _save(engine, content, *, project=None, kind=None, source=None):
    meta = {"project": project} if project else {}
    mid = engine._db.store_memory(
        MemoryRecord(profile_id=engine.profile_id, content=content, metadata=meta),
    )
    return engine._db.store_fact(
        AtomicFact(profile_id=engine.profile_id, memory_id=mid, content=content,
                  fact_type=FactType.SEMANTIC, memory_kind=kind, memory_kind_source=source),
    )


def test_fallback_excludes_quarantined_rows(engine_with_mock_deps) -> None:
    from superlocalmemory.server.unified_daemon import _recall_keyword_fallback

    keep = _save(engine_with_mock_deps, "deploy pipeline is green")
    quarantined = _save(engine_with_mock_deps, "deploy secret leaked in logs")
    engine_with_mock_deps._db.execute(
        "UPDATE atomic_facts SET quarantined=1 WHERE fact_id=?", (quarantined,),
    )

    out = _recall_keyword_fallback(engine_with_mock_deps, "deploy", 10)

    ids = {r["fact_id"] for r in out["results"]}
    assert ids == {keep}, ids


def test_fallback_excludes_caller_replaced_rows(engine_with_mock_deps) -> None:
    from superlocalmemory.server.unified_daemon import _recall_keyword_fallback

    old = _save(engine_with_mock_deps, "deploy status: green")
    new = _save(engine_with_mock_deps, "deploy status: red")
    engine_with_mock_deps._db.execute(
        "UPDATE fact_temporal_validity SET system_expired_at=?, "
        "invalidation_reason='replaced_by_caller' WHERE fact_id=?",
        ("2026-10-03T00:00:00+00:00", old),
    )

    out = _recall_keyword_fallback(engine_with_mock_deps, "deploy", 10)

    ids = {r["fact_id"] for r in out["results"]}
    assert ids == {new}, ids


def test_fallback_applies_project_and_kind_facets(engine_with_mock_deps) -> None:
    from superlocalmemory.retrieval.facets import Facets
    from superlocalmemory.server.unified_daemon import _recall_keyword_fallback

    zephyr_status = _save(
        engine_with_mock_deps, "Zephyr deploy is blocked on TLS",
        project="zephyr", kind="status", source="user",
    )
    atlas_decision = _save(
        engine_with_mock_deps, "Atlas deploy key rotation happens on Mondays",
        project="atlas", kind="decision", source="user",
    )

    out = _recall_keyword_fallback(
        engine_with_mock_deps, "deploy", 10,
        facets=Facets.of(project="zephyr", kind="status"),
    )

    ids = {r["fact_id"] for r in out["results"]}
    assert ids == {zephyr_status}, ids
    assert atlas_decision not in ids


def test_fallback_without_facets_is_unfiltered_as_before(engine_with_mock_deps) -> None:
    from superlocalmemory.server.unified_daemon import _recall_keyword_fallback

    a = _save(engine_with_mock_deps, "deploy one")
    b = _save(engine_with_mock_deps, "deploy two")

    out = _recall_keyword_fallback(engine_with_mock_deps, "deploy", 10)

    assert {r["fact_id"] for r in out["results"]} == {a, b}


def test_fallback_facet_failure_returns_nothing_and_says_why(
    engine_with_mock_deps, monkeypatch,
) -> None:
    """A facet that cannot be verified must cost results, not silently
    fall back to the unfiltered pool — and the envelope must say why."""
    from superlocalmemory.retrieval import facets as facets_mod
    from superlocalmemory.retrieval.facets import Facets
    from superlocalmemory.server.unified_daemon import _recall_keyword_fallback

    _save(engine_with_mock_deps, "deploy one", project="zephyr", source="user")

    def _boom(*a, **k):
        raise RuntimeError("entity resolver unavailable")

    monkeypatch.setattr(facets_mod, "matching_fact_ids", _boom)

    out = _recall_keyword_fallback(
        engine_with_mock_deps, "deploy", 10,
        facets=Facets.of(project="zephyr"),
    )

    assert out["results"] == []
    assert out.get("facet_filter_error"), out
