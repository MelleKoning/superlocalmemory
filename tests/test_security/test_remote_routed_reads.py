# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Every remote read tool answers for the key's profile, not the host's (4.1.21).

The host computer is using "personal"; the remote key is bound to "work". Each
tool is called the way remote access calls it (``profile_id="work"``) and must
answer with "work"'s memories only, while the host's active profile never
moves. Each test seeds both profiles differently, so a tool that ignored the
routed profile would answer with "personal"'s and fail.
"""

from __future__ import annotations

import datetime
import uuid

import pytest

from tests.test_security._routed_host import HOST, OTHER, WORK, open_host


@pytest.fixture
def host(tmp_path, monkeypatch, mock_embedder):
    with open_host(tmp_path, monkeypatch, mock_embedder) as opened:
        yield opened


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _two_profiles(host) -> tuple[list[str], list[str]]:
    """Three memories in "work", one in "personal"; their fact ids."""
    work = [host.save(f"The work rota for team {n} starts on Monday at 08:0{n}.", WORK,
                      f"w{n}") for n in range(3)]
    personal = [host.save("The personal dentist visit is on Friday at noon.", HOST, "p1")]
    return work, personal


# -- counts and status -------------------------------------------------------------------


def test_get_status_counts_the_keys_profile(host) -> None:
    work, _personal = _two_profiles(host)
    out = host.call("get_status")
    assert out["success"] is True, out
    assert out["profile"] == WORK and out["fact_count"] == len(work)


def test_health_counts_the_keys_profile(host) -> None:
    work, _personal = _two_profiles(host)
    out = host.call("health")
    assert out["profile"] == WORK
    assert out["components"]["database"]["fact_count"] == len(work)


def test_memory_used_counts_the_keys_profile(host) -> None:
    work, _personal = _two_profiles(host)
    out = host.call("memory_used")
    assert (out["profile"], out["total_facts"]) == (WORK, len(work))


def test_get_lifecycle_status_samples_only_the_keys_profile(host) -> None:
    work, personal = _two_profiles(host)
    out = host.call("get_lifecycle_status")
    sampled = {s["fact_id"] for state in out["samples"].values() for s in state}
    assert sampled and sampled <= set(work) and not sampled & set(personal)


def test_get_retention_stats_counts_the_keys_profile(host) -> None:
    work, personal = _two_profiles(host)
    for fact_id, profile in [(f, WORK) for f in work] + [(f, HOST) for f in personal]:
        host.engine._db.execute(
            "INSERT OR REPLACE INTO fact_retention (fact_id, profile_id, lifecycle_zone) "
            "VALUES (?, ?, ?)", (fact_id, profile, "warm" if profile == WORK else "cold"))
    out = host.call("get_retention_stats")
    assert (out["profile"], out["total"], out["warm"], out["cold"]) == (WORK, 3, 3, 0)


def test_get_brain_evidence_status_reads_the_keys_profile(host) -> None:
    out = host.call("get_brain_evidence_status")
    assert out["success"] is True, out
    assert out["profile_id"] == WORK and out["brain_truth"]["profile_id"] == WORK


# -- learned behaviour ---------------------------------------------------------------------


def _assertion(host, profile: str, trigger: str) -> str:
    assertion_id = uuid.uuid4().hex
    host.engine._db.execute(
        "INSERT INTO behavioral_assertions (id, profile_id, trigger_condition, action, "
        "category, confidence, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (assertion_id, profile, trigger, "do it", "skill_performance", 0.6, _now(), _now()))
    return assertion_id


def test_get_assertions_returns_only_the_keys_profile(host) -> None:
    mine = _assertion(host, WORK, "when deploying work")
    _assertion(host, HOST, "when cooking at home")
    out = host.call("get_assertions")
    assert [a["id"] for a in out["assertions"]] == [mine], out


def _pattern(host, profile: str, key: str) -> None:
    from superlocalmemory.learning.behavioral import BehavioralPatternStore

    BehavioralPatternStore(host.engine._db.db_path).record_pattern(
        profile, "interest", {"topic": key}, confidence=0.5)


def test_get_behavioral_patterns_returns_only_the_keys_profile(host) -> None:
    _pattern(host, WORK, "kubernetes")
    _pattern(host, HOST, "gardening")
    out = host.call("get_behavioral_patterns")
    assert [p["pattern_key"] for p in out["patterns"]] == ["kubernetes"], out


def test_get_learned_patterns_returns_only_the_keys_profile(host) -> None:
    _pattern(host, WORK, "kubernetes")
    _pattern(host, HOST, "gardening")
    out = host.call("get_learned_patterns")
    assert [p["pattern_key"] for p in out["patterns"]] == ["kubernetes"], out


def test_get_soft_prompts_returns_only_the_keys_profile(host) -> None:
    for profile, text in ((WORK, "Prefers terse status updates."),
                          (HOST, "Likes long letters.")):
        host.engine._db.execute(
            "INSERT INTO soft_prompt_templates (prompt_id, profile_id, category, content, "
            "confidence) VALUES (?, ?, 'communication_style', ?, 0.8)",
            (uuid.uuid4().hex, profile, text))
    out = host.call("get_soft_prompts")
    assert out["profile"] == WORK
    assert [p["content"] for p in out["prompts"]] == ["Prefers terse status updates."]


def _skill_event(host, profile: str, skill: str) -> None:
    host.engine._db.execute(
        "INSERT INTO tool_events (session_id, profile_id, tool_name, event_type, "
        "input_summary, created_at) VALUES ('s', ?, 'Skill', 'complete', ?, ?)",
        (profile, '{"skill": "%s"}' % skill, _now()))


def test_skill_health_reads_the_keys_profile(host) -> None:
    _skill_event(host, WORK, "deploy")
    _skill_event(host, HOST, "bake")
    out = host.call("skill_health")
    assert out["profile_id"] == WORK
    assert [s["name"] for s in out["skills"]] == ["deploy"], out


def test_skill_lineage_reads_the_keys_profile(host) -> None:
    from superlocalmemory.evolution.evolution_store import EvolutionStore

    EvolutionStore(host.engine._db.db_path)  # creates its tables
    for profile, skill in ((WORK, "deploy"), (HOST, "bake")):
        host.engine._db.execute(
            "INSERT INTO skill_evolution_log (id, profile_id, skill_name, evolution_type, "
            "trigger_type, status, created_at) VALUES (?, ?, ?, 'fix', 'manual', 'done', ?)",
            (uuid.uuid4().hex, profile, skill, _now()))
    out = host.call("skill_lineage")
    assert [entry["skill_name"] for entry in out["lineage"]] == ["deploy"], out


# -- loops ---------------------------------------------------------------------------------


def _lap(host, profile: str, run_id: str, name: str) -> None:
    from superlocalmemory.loops.ledger import LedgerEntry
    from superlocalmemory.storage.models import MemoryRecord

    entry = LedgerEntry(run_id=run_id, name=name, lap=1, ts=_now(), decision="done",
                        passed=True, detail="gate passed")
    host.engine._db.store_memory(MemoryRecord(profile_id=profile, content=entry.to_json(),
                                              session_id=f"loop:{run_id}"))


def test_slm_loop_history_and_show_read_the_keys_profile(host) -> None:
    _lap(host, WORK, "run-work", "nightly")
    _lap(host, HOST, "run-personal", "nightly")
    history = host.call("slm_loop_history", name="nightly")
    assert [r["run_id"] for r in history["runs"]] == ["run-work"], history
    shown = host.call("slm_loop_show", run_id="run-work")
    assert shown["count"] == 1, shown
    assert host.call("slm_loop_show", run_id="run-personal")["count"] == 0


# -- recall-shaped reads --------------------------------------------------------------------


def test_recall_trace_recalls_from_the_keys_profile(host) -> None:
    work, personal = _two_profiles(host)
    out = host.call("recall_trace", query="When does the rota start?")
    ids = {r["fact_id"] for r in out["results"]}
    assert ids and ids <= set(work) and not ids & set(personal), out


def test_session_init_loads_only_the_keys_profile(host) -> None:
    work, personal = _two_profiles(host)
    # Matches the query as well as the work memories do, so only the profile
    # keeps it out of the recall half of the context.
    personal.append(host.save("The personal rota for team chores starts on Monday at 08:00.",
                              HOST, "p2"))
    host.engine._db.set_pinned(work[0], True)
    host.engine._db.set_pinned(personal[0], True)
    out = host.call("session_init", query="rota", max_age_days=0)
    ids = {m["fact_id"] for m in out["memories"]}
    assert out["success"] is True and out["degraded_mode"] is False, out
    assert work[0] in ids and not ids & set(personal), out
    assert "dentist" not in out["context"] and "chores" not in out["context"]


def test_get_memory_summary_summarises_the_keys_profile(host) -> None:
    work, personal = _two_profiles(host)
    out = host.call("get_memory_summary", kind="day", target="today")
    sources = set(out.get("source_fact_ids", []))
    assert out.get("success", True) is not False, out
    assert sources and not sources & set(personal), out


def test_memory_kinds_status_and_review_read_the_keys_profile(host) -> None:
    work, personal = _two_profiles(host)
    db = host.engine._db
    db.execute("UPDATE atomic_facts SET memory_kind='decision', memory_kind_source='rules', "
               "memory_kind_confidence=0.9 WHERE fact_id=?", (work[0],))
    db.execute("UPDATE atomic_facts SET memory_kind='rule', memory_kind_source='rules', "
               "memory_kind_confidence=0.9 WHERE fact_id=?", (personal[0],))
    status = host.call("memory_kinds_status")
    assert status["success"] is True, status
    kinds = status["counts"]["kind"]
    assert sum(kinds["decision"].values()) == 1, status
    assert sum(kinds["rule"].values()) == 0, status
    review = host.call("review_memory_kinds", limit=50)
    assert review["success"] is True, review
    assert {i["fact_id"] for i in review["items"]} == {work[0]}, review


def test_run_view_lists_and_runs_the_keys_profile(host) -> None:
    from superlocalmemory.views import default_store

    work, personal = _two_profiles(host)
    store = default_store()
    store.create(WORK, name="Rota", query="When does the rota start?")
    store.create(HOST, name="Dentist", query="When is the dentist?")
    listed = host.call("run_view")
    assert [v["name"] for v in listed["views"]] == ["Rota"], listed
    ran = host.call("run_view", name="Rota")
    ids = {r["fact_id"] for r in ran.get("results", [])}
    assert ids and ids <= set(work), ran
    refused = host.call("run_view", name="Dentist")
    assert refused.get("success") is False, refused


def test_prestage_context_reads_the_keys_profile(host) -> None:
    work, personal = _two_profiles(host)
    out = host.call("prestage_context", query="When does the rota start?")
    ids = {m["id"] for m in out["memories"]}
    assert ids and ids <= set(work), out


# -- a profile that does not exist ---------------------------------------------------------


@pytest.mark.parametrize("tool, args", [
    ("get_status", {}), ("health", {}), ("memory_used", {}), ("get_assertions", {}),
    ("get_soft_prompts", {}), ("skill_health", {}), ("get_brain_evidence_status", {}),
    ("get_memory_summary", {}), ("slm_loop_history", {"name": "n"}), ("session_init", {}),
])
def test_a_profile_that_does_not_exist_is_refused_not_read(host, tool, args) -> None:
    import asyncio

    out = asyncio.run(host.tools[tool](**args, profile_id=OTHER))
    assert out.get("code") == "unknown_profile", (tool, out)
    assert not host.rows("SELECT 1 FROM profiles WHERE profile_id=?", (OTHER,))
