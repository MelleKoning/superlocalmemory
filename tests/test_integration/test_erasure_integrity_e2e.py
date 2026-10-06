# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Erasure and refusal through a REAL daemon on a synthetic store (4.1.22 G07).

* A fully erased memory leaves its words nowhere: not in any table of any SLM
  database in the data root (ingestion journal, entity summaries, event
  previews and the keyword index's own blocks included), not in the daemon's
  logs, and not in what recall returns by keyword, meaning, date or entity.
* Deleting a memory that correction history protects is refused before
  anything is touched, and ``slm delete`` prints that reason instead of
  claiming the daemon is unavailable.

Meaning search needs the embedding model. Set ``SLM_TEST_MODEL_CACHE`` to a
Hugging Face cache that holds it (offline) to prove that channel too; without
it the semantic assertions are skipped, never faked.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
import uuid

import pytest

from tests.test_integration.test_per_request_profile_e2e import (
    PRODUCTION_PORTS,
    REPO_ROOT,
    RealDaemon,
    _child_env,
    _foreign_daemon_pids,
    _reserve_private_port,
)


ERASE_PROFILE = "g07erase"
MODEL_CACHE = os.environ.get("SLM_TEST_MODEL_CACHE", "")


@pytest.fixture(scope="module")
def daemon(tmp_path_factory):
    root = tmp_path_factory.mktemp("g07-erasure")
    data_root = root / "data"
    data_root.mkdir()
    port = _reserve_private_port()
    assert port not in PRODUCTION_PORTS
    # Mode A, local embedding only: no local or hosted language model is asked
    # anything, so extraction is deterministic and the store stays synthetic.
    (data_root / "config.json").write_text(json.dumps({
        "mode": "a", "active_profile": "default", "daemon_port": port,
        "daemon_enable_legacy_port": False, "mesh_enabled": False,
        "scale_auto_promote_enabled": False,
        "embedding": {"model_name": "nomic-ai/nomic-embed-text-v1.5", "dimension": 768},
        "retrieval": {"sufficiency_judge": "off"},
    }), encoding="utf-8")
    env = _child_env(data_root, port, root / "home", root / "cache")
    if MODEL_CACHE:
        env["HF_HOME"] = MODEL_CACHE
        env.pop("SENTENCE_TRANSFORMERS_HOME", None)
    foreign = _foreign_daemon_pids()
    log = root / "daemon-stdout.log"
    with log.open("wb") as handle:
        proc = subprocess.Popen(
            [sys.executable, "-m", "superlocalmemory.server.unified_daemon",
             "--start", f"--port={port}"],
            stdout=handle, stderr=handle, env=env, cwd=str(REPO_ROOT),
            start_new_session=os.name == "posix",
        )
    real = RealDaemon(proc, port, data_root, log, env)
    try:
        real.wait_ready()
        real.precreate_profiles((ERASE_PROFILE,))
        yield real
    finally:
        real.stop(foreign)


# -- read-only probes -------------------------------------------------------

def _ro(daemon: RealDaemon, sql: str, args: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(f"file:{daemon.data_root / 'memory.db'}?mode=ro",
                           uri=True, timeout=30)
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


def _count(daemon: RealDaemon, sql: str, args: tuple = ()) -> int:
    try:
        return int(_ro(daemon, sql, args)[0][0])
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return 0
        raise


def _remember_complete(daemon: RealDaemon, text: str, profile: str) -> list[str]:
    key = f"g07-{uuid.uuid4().hex[:10]}"
    daemon.remember(text, profile, key)
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        rows = _ro(daemon, "SELECT state, final_fact_ids_json FROM ingestion_operations "
                   "WHERE profile_id = ? AND idempotency_key = ?", (profile or "default", key))
        if rows and rows[0][0] == "complete":
            facts = [str(f) for f in json.loads(rows[0][1] or "[]")]
            live = [f for f in facts if _count(
                daemon, "SELECT COUNT(*) FROM atomic_facts WHERE fact_id = ?", (f,))]
            assert live, "the memory produced no stored fact"
            return live
        time.sleep(0.5)
    raise AssertionError("remember never completed enrichment")


def _text_copies(daemon: RealDaemon, needle: str) -> dict[str, int]:
    """Every column of every table of every database that holds the needle."""
    hits: dict[str, int] = {}
    for db in sorted(daemon.data_root.rglob("*.db")):
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
        except sqlite3.Error:
            continue
        try:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'")]
            for table in tables:
                try:
                    cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
                except sqlite3.Error:
                    continue  # a virtual table whose module is not loaded here
                for col in cols:
                    try:
                        n = conn.execute(
                            f'SELECT COUNT(*) FROM "{table}" WHERE instr(lower('
                            f'CAST("{col}" AS TEXT)), ?) > 0', (needle.lower(),)).fetchone()[0]
                    except sqlite3.Error:
                        continue
                    if n:
                        hits[f"{db.name}:{table}.{col}"] = n
        finally:
            conn.close()
    return hits


def _log_copies(daemon: RealDaemon, needle: str) -> list[str]:
    files = [daemon.stdout_log, *(daemon.data_root / "logs").glob("*")]
    return [p.name for p in files if p.is_file()
            and needle.lower() in p.read_text(errors="replace").lower()]


def _recall(daemon: RealDaemon, query: str, profile: str) -> list[dict]:
    return list(daemon.recall(query, profile).get("results") or [])


def _semantic_ready(daemon: RealDaemon) -> bool:
    return bool(MODEL_CACHE) and _count(
        daemon, "SELECT COUNT(*) FROM embedding_metadata") > 0


# -- erasure ------------------------------------------------------------------

FACT_KEYED = (
    ("bm25_tokens", "fact_id"), ("embedding_metadata", "fact_id"),
    ("vector_row_map", "fact_id"), ("fact_outcome_score", "fact_id"),
    ("activation_cache", "node_id"), ("temporal_events", "fact_id"),
    ("fact_entity_associations", "fact_id"), ("fact_retention", "fact_id"),
    ("fact_temporal_validity", "fact_id"), ("polar_embeddings", "fact_id"),
)


def _protected_count(daemon: RealDaemon, facts: list[str]) -> int:
    ph = ",".join("?" * len(facts))
    return _count(daemon, f"SELECT COUNT(*) FROM correction_cases WHERE predecessor_fact_id "
                  f"IN ({ph}) OR successor_fact_id IN ({ph})", tuple(facts) * 2)


#: A memory whose facts are extracted from a full sentence usually proposes a
#: correction of itself (machine consolidation between its own facts), and a
#: fact named by any correction case can never be deleted (reported by G07 for
#: Varun's decision). Erasure is proven on the first memory that did not; the
#: later phrasings are noun phrases, which extract to one fact.
PHRASINGS = (
    "{m} Haverlin audits the {p} depot on 3 March 2026 with the synthetic ledger.",
    "{m} {p} ledger audit, 3 March 2026",
    "{m} {p} synthetic depot ledger note",
    "{m} {p} audit note",
)


def test_full_erasure_leaves_the_words_nowhere(daemon: RealDaemon) -> None:
    for phrasing in PHRASINGS:
        tag = uuid.uuid4().hex[:6]
        marker, place = f"zorvexa{tag}", f"Quillmarsh{tag}"
        facts = _remember_complete(daemon, phrasing.format(m=marker, p=place), ERASE_PROFILE)
        if _protected_count(daemon, facts) == 0:
            break
    else:
        pytest.fail("every fresh memory protected itself by correction history")
    ph = ",".join("?" * len(facts))

    before = _recall(daemon, marker, ERASE_PROFILE)
    keyword_hits = [r for r in before
                    if r.get("fact_id") in facts and "bm25" in (r.get("channel_scores") or {})]
    assert keyword_hits, "precondition: the marker is found by keyword before erasure"
    copies_before = _text_copies(daemon, marker)
    assert "memory.db:ingestion_operations.raw_content" in copies_before, copies_before
    assert "memory.db:atomic_facts_fts_data.block" in copies_before, copies_before
    semantic = _semantic_ready(daemon)
    if semantic:
        assert any("semantic" in (r.get("channel_scores") or {}) for r in before
                   if r.get("fact_id") in facts), "precondition: found by meaning"

    for fact_id in facts:
        code, body = daemon.request("DELETE", f"/api/memories/{fact_id}",
                                    params={"profile_id": ERASE_PROFILE})
        assert code == 200, body
        assert body["erasure_verified"] is True and body["erasure_state"] == "COMPLETE", body

    for query in (marker, place, f"{marker} Haverlin depot ledger",
                  "who audits the depot on 3 March 2026"):
        for result in _recall(daemon, query, ERASE_PROFILE):
            assert result.get("fact_id") not in facts, (query, result.get("channel_scores"))
            assert marker not in str(result.get("content", "")).lower(), query
            assert place.lower() not in str(result.get("content", "")).lower(), query

    assert _text_copies(daemon, marker) == {}
    assert _text_copies(daemon, place) == {}
    assert _log_copies(daemon, marker) == []
    for table, column in FACT_KEYED:
        left = _count(daemon, f"SELECT COUNT(*) FROM {table} WHERE {column} IN ({ph})",
                      tuple(facts))
        assert left == 0, (table, left)
    assert _count(daemon, f"SELECT COUNT(*) FROM graph_edges WHERE source_id IN ({ph}) "
                  f"OR target_id IN ({ph})", tuple(facts) * 2) == 0
    if not semantic:
        pytest.skip("erasure proven for keyword/date/entity and every table; set "
                    "SLM_TEST_MODEL_CACHE to also prove the meaning channel")


# -- refusal ------------------------------------------------------------------

def _footprint(daemon: RealDaemon, fact_id: str) -> dict[str, int]:
    out = {t: _count(daemon, f"SELECT COUNT(*) FROM {t} WHERE {c} = ?", (fact_id,))
           for t, c in FACT_KEYED + (("atomic_facts", "fact_id"),)}
    out["tombstones"] = _count(daemon, "SELECT COUNT(*) FROM projection_tombstones")
    out["receipts"] = _count(daemon, "SELECT COUNT(*) FROM erasure_receipts")
    return out


def _protected_fact(daemon: RealDaemon) -> str:
    tag = uuid.uuid4().hex[:6]
    fact_id = _remember_complete(
        daemon, f"Brannoc{tag} keeps the synthetic teal kettle in the north pantry.", "")[0]
    for _ in range(45):  # 503 = the writer is busy; the route says retry
        code, body = daemon.request(
            "PATCH", f"/api/memories/{fact_id}",
            {"content": f"Brannoc{tag} keeps the teal kettle in the south pantry."})
        if code != 503:
            break
        time.sleep(2)
    assert code in (200, 202), body
    assert _count(daemon, "SELECT COUNT(*) FROM correction_cases WHERE "
                  "predecessor_fact_id = ?", (fact_id,)) == 1
    return fact_id


def test_protected_delete_is_refused_before_anything_changes(daemon: RealDaemon) -> None:
    fact_id = _protected_fact(daemon)
    before = _footprint(daemon, fact_id)
    assert before["atomic_facts"] == 1 and before["bm25_tokens"] >= 1, before

    code, body = daemon.request("DELETE", f"/api/memories/{fact_id}")

    assert code == 409, body
    assert "protected by correction history" in str(body.get("detail", "")), body
    assert "Nothing was changed" in str(body.get("detail", "")), body
    assert _footprint(daemon, fact_id) == before


def test_cli_delete_prints_the_refusal_not_an_outage(daemon: RealDaemon) -> None:
    fact_id = _protected_fact(daemon)
    cmd = [sys.executable, "-m", "superlocalmemory.cli.main", "delete", fact_id, "--yes"]

    plain = subprocess.run(cmd, env=daemon.env, cwd=str(REPO_ROOT),
                           capture_output=True, text=True, timeout=180)
    as_json = subprocess.run([*cmd, "--json"], env=daemon.env, cwd=str(REPO_ROOT),
                             capture_output=True, text=True, timeout=180)

    assert plain.returncode == 1, (plain.stdout, plain.stderr)
    assert "Refused: fact is protected by correction history" in plain.stderr, plain.stderr
    assert "DAEMON_UNAVAILABLE" not in plain.stdout + plain.stderr
    assert as_json.returncode == 1, as_json.stdout
    envelope = json.loads(as_json.stdout)
    assert envelope["success"] is False
    assert envelope["error"]["code"] == "CONFLICT"
    assert envelope["error"]["retryable"] is False
    assert _count(daemon, "SELECT COUNT(*) FROM atomic_facts WHERE fact_id = ?", (fact_id,)) == 1

