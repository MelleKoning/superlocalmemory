# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Kind-aware ordering as recall runs it: real settings, the query's subject
read from the store without writing to it, and an off switch that leaves recall
exactly as it would be without the pass (L2-05, L2-16)."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest

from superlocalmemory.core import recall_pipeline as rp
from superlocalmemory.core.config import RetrievalConfig, SLMConfig
from superlocalmemory.encoding.entity_resolver import EntityResolver
from superlocalmemory.retrieval import kind_aware
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import (
    AtomicFact,
    CanonicalEntity,
    FactType,
    RecallResponse,
    RetrievalResult,
)

_QUERY = "what is the current status of the Atlas migration"


# --- L2-16: real, validated config fields -----------------------------------------

def test_the_settings_are_retrieval_config_fields_with_safe_defaults() -> None:
    names = RetrievalConfig.__dataclass_fields__
    assert "kind_aware" in names and "kind_aware_boost" in names
    rc = RetrievalConfig()
    assert rc.kind_aware is True
    assert rc.kind_aware_boost == kind_aware.DEFAULT_BOOST


@pytest.mark.parametrize("raw,expected", [
    (50, 0.5), (0.3, 0.3), (-1.0, 0.0), (float("nan"), 0.15), ("lots", 0.15), (None, 0.15),
])
def test_the_boost_is_validated_and_clamped(raw, expected) -> None:
    assert RetrievalConfig(kind_aware_boost=raw).kind_aware_boost == expected


@pytest.mark.parametrize("raw,expected", [(False, False), (True, True), ("false", True),
                                          (0, True), (None, True)])
def test_the_switch_takes_only_a_real_boolean(raw, expected) -> None:
    assert RetrievalConfig(kind_aware=raw).kind_aware is expected


def test_the_config_file_sets_both_and_a_boost_of_50_is_clamped(tmp_path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "mode": "a", "active_profile": "default",
        "retrieval": {"kind_aware": False, "kind_aware_boost": 50},
    }))
    loaded = SLMConfig.load(cfg_path)
    assert loaded.retrieval.kind_aware is False
    assert loaded.retrieval.kind_aware_boost == 0.5
    loaded.save(cfg_path)
    saved = json.loads(cfg_path.read_text())["retrieval"]
    assert saved["kind_aware"] is False and saved["kind_aware_boost"] == 0.5


# --- recall itself ------------------------------------------------------------------

def _facts(subject_ids=("ent_atlas",)):
    old = AtomicFact(fact_id="atlas-old", content="Atlas migration blocked on schema review",
                     fact_type=FactType.SEMANTIC, created_at="2026-09-01",
                     canonical_entities=list(subject_ids), memory_kind="status",
                     memory_kind_source="user")
    noise = [AtomicFact(fact_id=f"noise{i}", content=f"unrelated note {i}",
                        created_at="2026-09-02") for i in range(3)]
    new = AtomicFact(fact_id="atlas-new", content="Atlas migration done, schema approved",
                     fact_type=FactType.SEMANTIC, created_at="2026-10-02",
                     canonical_entities=list(subject_ids), memory_kind="status",
                     memory_kind_source="caller")
    unnamed = AtomicFact(fact_id="kind-match", content="deploy windows are open",
                         created_at="2026-09-03", memory_kind="status",
                         memory_kind_source="caller")
    return [old, *noise, unnamed, new]


def _engine(make_facts, *, db=None, resolver=None):
    def _recall(query, profile_id, mode=None, limit=10, **kwargs):
        facts = make_facts()
        scores = [0.90, 0.80, 0.78, 0.76, 0.75, 0.40]
        results = [RetrievalResult(fact=f, score=s, ranking_score=s, confidence=1.0)
                   for f, s in zip(facts, scores)]
        return RecallResponse(query=query, results=results, query_type="lookup")
    ns = SimpleNamespace(recall=_recall, _sufficiency_judge=None)
    if db is not None:
        ns._db = db
    if resolver is not None:
        ns._entity = SimpleNamespace(_resolver=resolver)
    return ns


def _run(config, engine, db=None, query=_QUERY):
    return rp.run_recall(query, "default", fast=True, config=config, retrieval_engine=engine,
                         trust_scorer=None, embedder=None, db=db, llm=None, hooks=None)


def _snapshot(response) -> str:
    data = dataclasses.asdict(response)
    data.pop("retrieval_time_ms", None)
    data.pop("query_id", None)
    # 4.1.20 (652ceabf) attaches answer_check_trace with wall-clock timings.
    # Like retrieval_time_ms they differ run to run; its provenance fields
    # (detail, backend, threshold, reordered) stay in the comparison.
    trace = data.get("answer_check_trace")
    if isinstance(trace, dict):
        data["answer_check_trace"] = {
            k: v for k, v in trace.items()
            if k not in ("retrieval_ms", "judge_ms", "total_ms")
        }
    return json.dumps(data, sort_keys=True, default=str)


def test_turning_it_off_is_byte_identical_to_recall_without_the_pass(
        mode_a_config, monkeypatch) -> None:
    monkeypatch.setattr(rp, "apply_ranking", lambda resp, *a, **k: resp)
    engine = _engine(_facts)
    subject = frozenset({"ent_atlas"})
    monkeypatch.setattr(kind_aware, "query_subject", lambda *a, **k: subject)

    on = _snapshot(_run(mode_a_config, engine))
    mode_a_config.retrieval.kind_aware = False
    off = _snapshot(_run(mode_a_config, engine))
    with monkeypatch.context() as m:
        m.setattr(kind_aware, "apply_for_recall", lambda results, *a, **k: results)
        absent = _snapshot(_run(mode_a_config, engine))
    assert off == absent
    assert on != off, "control: with the pass on, this recall's order must differ"


def _store(tmp_path):
    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    atlas = db.store_entity(CanonicalEntity(canonical_name="Atlas", profile_id="default",
                                            entity_type="project"))
    varun = db.store_entity(CanonicalEntity(canonical_name="Varun", profile_id="default",
                                            entity_type="person"))
    return db, atlas, varun


def _counts(db) -> tuple[int, int]:
    ents = dict(db.execute("SELECT COUNT(*) AS n FROM canonical_entities")[0])["n"]
    aliases = dict(db.execute("SELECT COUNT(*) AS n FROM entity_aliases")[0])["n"]
    return ents, aliases


@pytest.mark.parametrize("with_resolver", [True, False])
def test_recall_reads_the_named_subject_from_the_store_without_writing(
        tmp_path, mode_a_config, monkeypatch, with_resolver) -> None:
    monkeypatch.setattr(rp, "apply_ranking", lambda resp, *a, **k: resp)
    db, atlas, _varun = _store(tmp_path)
    before = _counts(db)
    engine = _engine(lambda: _facts((atlas,)), db=db,
                     resolver=EntityResolver(db) if with_resolver else None)
    out = _run(mode_a_config, engine, db=db)
    ids = [r.fact.fact_id for r in out.results]
    assert ids.index("atlas-new") < ids.index("atlas-old")
    old = next(r for r in out.results if r.fact.fact_id == "atlas-old")
    assert "newer:atlas-new" in old.evidence_chain
    assert _counts(db) == before


def test_a_shared_entity_the_query_does_not_name_moves_nothing(
        tmp_path, mode_a_config, monkeypatch) -> None:
    monkeypatch.setattr(rp, "apply_ranking", lambda resp, *a, **k: resp)
    db, _atlas, varun = _store(tmp_path)
    engine = _engine(lambda: _facts((varun,)), db=db, resolver=EntityResolver(db))
    out = _run(mode_a_config, engine, db=db)
    ids = [r.fact.fact_id for r in out.results]
    assert ids.index("atlas-old") < ids.index("atlas-new")
    assert not any(n.startswith("newer:") for r in out.results for n in r.evidence_chain or [])


def test_recall_uses_the_configured_boost(mode_a_config, monkeypatch) -> None:
    monkeypatch.setattr(rp, "apply_ranking", lambda resp, *a, **k: resp)
    monkeypatch.setattr(kind_aware, "query_subject", lambda *a, **k: frozenset())
    engine = _engine(_facts)
    mode_a_config.retrieval.kind_aware_boost = 0.0
    zero = [r.fact.fact_id for r in _run(mode_a_config, engine).results]
    mode_a_config.retrieval.kind_aware_boost = 0.15
    lifted = [r.fact.fact_id for r in _run(mode_a_config, engine).results]
    assert zero.index("kind-match") > lifted.index("kind-match")
