# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""A measurement stays a measurement, through the real daemon, in both modes.

The 4.1.21 regression: a memory said a checkpoint "was recalled at rank 1 in
2004.6 ms"; the local model's extraction stored "The recall happened on June
1st, 2004" and dated two facts 2004-06-01. That was reproduced against a real
local model on 4.1.21 before this test was written; the stub model below
answers with exactly that kind of output, so the composed path is exercised
deterministically on any machine:

    real daemon (isolated data root, private port)
      -> POST /remember          first-queryable fact
      -> background enrichment   model extraction (Mode B) / rules (Mode A)
      -> lineage checkpoint      source-fidelity check, withholding
      -> GET /recall             what a user is actually shown
      -> slm db fidelity         what is offered for review

Every fixture sentence is synthetic and de-identified.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.test_integration.test_per_request_profile_e2e import (
    PRODUCTION_PORTS,
    REPO_ROOT,
    RealDaemon,
    _child_env,
    _foreign_daemon_pids,
    _reserve_private_port,
)

KESTREL = "The Kestrel checkpoint was recalled at rank 1 in 2004.6 ms."
POLICY = (
    "Never publish without approval from the release owner. "
    "Ship the iOS build before the Android build. "
    "Previously the team used Jenkins, currently it uses GitHub Actions."
)

#: What the 4.1.21 local model answered for KESTREL (shape reproduced live).
KESTREL_EXTRACTION = [
    {"text": "The Kestrel checkpoint was recalled at rank 1", "fact_type": "episodic",
     "entities": ["Kestrel"], "referenced_date": "2004-06-01", "importance": 6,
     "confidence": 0.9},
    {"text": "The recall happened on June 1st, 2004", "fact_type": "episodic",
     "entities": ["Kestrel"], "referenced_date": "2004-06-01", "importance": 5,
     "confidence": 0.9},
    {"text": "The Kestrel checkpoint recall took 2004.6 ms", "fact_type": "semantic",
     "entities": ["Kestrel"], "referenced_date": None, "importance": 6, "confidence": 0.9},
]
#: Polarity, order and status damage of the kind a paraphrasing model makes.
POLICY_EXTRACTION = [
    {"text": "Publish with approval from the release owner", "fact_type": "semantic",
     "entities": [], "importance": 8, "confidence": 0.9},
    {"text": "Ship the Android build before the iOS build", "fact_type": "prospective",
     "entities": ["Android"], "importance": 7, "confidence": 0.9},
    {"text": "The team uses Jenkins", "fact_type": "semantic",
     "entities": ["Jenkins"], "importance": 6, "confidence": 0.9},
    {"text": "The team uses GitHub Actions", "fact_type": "semantic",
     "entities": ["GitHub Actions"], "importance": 6, "confidence": 0.9},
]
BAD_TEXTS = {
    "The recall happened on June 1st, 2004",
    "Publish with approval from the release owner",
    "Ship the Android build before the iOS build",
    "The team uses Jenkins",
}
GOOD_TEXTS = {"The Kestrel checkpoint recall took 2004.6 ms", "The team uses GitHub Actions"}


def _vector(text: str, dim: int = 768) -> list[float]:
    """Deterministic unit vector per text (a stand-in embedding model)."""
    raw = [b for i in range(0, dim, 32)
           for b in hashlib.sha256(f"{i}:{text}".encode()).digest()][:dim]
    values = [(b - 127.5) / 127.5 for b in raw]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


class _StubModels(BaseHTTPRequestHandler):
    """A local model server: Ollama chat/generate plus OpenAI-style embeddings."""

    def log_message(self, *_args) -> None:  # keep test output clean
        return

    def _send(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self._send({"models": [{"name": "stub-llm:latest"}, {"name": "nomic-embed-text"}]})

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        if self.path.rstrip("/").endswith("/embeddings"):
            texts = request.get("input") or []
            texts = [texts] if isinstance(texts, str) else texts
            self._send({"data": [{"index": i, "embedding": _vector(str(t))}
                                 for i, t in enumerate(texts)]})
            return
        messages = request.get("messages") or []
        system = " ".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
        user = " ".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")
        answer = "[]"
        if "fact extraction engine" in system:
            if "Kestrel" in user:
                answer = json.dumps(KESTREL_EXTRACTION)
            elif "release owner" in user:
                answer = json.dumps(POLICY_EXTRACTION)
        if self.path.endswith("/api/generate"):
            self._send({"response": answer, "done": True})
        else:
            self._send({"message": {"role": "assistant", "content": answer}, "done": True})


@pytest.fixture(scope="module")
def stub_models():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubModels)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _start_daemon(root: Path, mode: str, models_url: str) -> tuple[RealDaemon, set[int]]:
    data_root = root / "data"
    data_root.mkdir(parents=True)
    port = _reserve_private_port()
    assert port not in PRODUCTION_PORTS
    config = {
        "mode": mode, "active_profile": "default", "daemon_port": port,
        "daemon_enable_legacy_port": False, "mesh_enabled": False,
        "scale_auto_promote_enabled": False,
        # A local OpenAI-compatible embedder, so enrichment runs on any machine.
        "embedding": {"provider": "openai", "api_endpoint": f"{models_url}/v1",
                      "model_name": "stub-embed", "dimension": 768, "api_key": ""},
        "retrieval": {"sufficiency_judge": "off"},
    }
    if mode == "b":
        config["llm"] = {"provider": "ollama", "model": "stub-llm", "base_url": models_url}
    (data_root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    env = _child_env(data_root, port, root / "home", root / "cache")
    env["OLLAMA_HOST"] = models_url
    foreign_before = _foreign_daemon_pids()
    log_path = root / "daemon-stdout.log"
    with log_path.open("wb") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "superlocalmemory.server.unified_daemon",
             "--start", f"--port={port}"],
            stdout=log_file, stderr=log_file, env=env, cwd=str(REPO_ROOT),
            start_new_session=os.name == "posix",
        )
    daemon = RealDaemon(proc, port, data_root, log_path, env)
    daemon.wait_ready()
    return daemon, foreign_before


def _rows(daemon: RealDaemon, sql: str, args: tuple = ()) -> list[dict]:
    conn = sqlite3.connect(f"file:{daemon.data_root / 'memory.db'}?mode=ro", uri=True,
                           timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, args)]
    finally:
        conn.close()


def _remember_and_enrich(daemon: RealDaemon, content: str, key: str) -> tuple[dict, list[dict]]:
    payload = daemon.remember(content, profile_id="", idempotency_key=key)
    first = _rows(daemon, "SELECT fact_id, content, referenced_date FROM atomic_facts "
                          "WHERE fact_id IN (SELECT value FROM json_each(?))",
                  (json.dumps(payload["fact_ids"]),))
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        state = _rows(daemon, "SELECT state, last_error FROM ingestion_operations "
                              "WHERE idempotency_key=?", (key,))
        if state and state[0]["state"] in ("complete", "failed"):
            assert state[0]["state"] == "complete", state
            return payload, first
        time.sleep(0.5)
    raise AssertionError(f"enrichment of {key!r} never completed: {state}")


def _memory_facts(daemon: RealDaemon, content: str) -> list[dict]:
    return _rows(
        daemon,
        "SELECT f.fact_id, f.content, f.referenced_date, f.interval_start, f.interval_end, "
        "COALESCE(f.quarantined,0) AS quarantined, l.unresolved_reason "
        "FROM atomic_facts f JOIN memories m ON m.memory_id = f.memory_id "
        "LEFT JOIN derivation_lineage l ON l.object_type='fact' AND l.object_id=f.fact_id "
        "WHERE m.content = ?",
        (content,),
    )


def _no_invented_2004_date(fact: dict) -> bool:
    fields = " ".join(str(fact.get(k) or "") for k in
                      ("content", "referenced_date", "interval_start", "interval_end"))
    return "2004-06" not in fields and not re.search(r"June\s+1(st)?,?\s+2004", fields)


def _recall_contents(daemon: RealDaemon, query: str) -> list[str]:
    payload = daemon.recall(query, profile_id="default")
    return [str(item.get("content", "")) for item in payload.get("results", [])]


def _fidelity_listing(daemon: RealDaemon) -> dict:
    out = subprocess.run(
        [sys.executable, "-m", "superlocalmemory.cli.main", "db", "fidelity", "--json"],
        env=daemon.env, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr[-1500:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def mode_b_daemon(tmp_path_factory, stub_models):
    daemon, foreign = _start_daemon(tmp_path_factory.mktemp("g03-b"), "b", stub_models)
    try:
        yield daemon
    finally:
        daemon.stop(foreign)


@pytest.fixture(scope="module")
def mode_a_daemon(tmp_path_factory, stub_models):
    daemon, foreign = _start_daemon(tmp_path_factory.mktemp("g03-a"), "a", stub_models)
    try:
        yield daemon
    finally:
        daemon.stop(foreign)


class TestModeBModelExtraction:
    def test_measurement_is_never_stored_as_a_trusted_date(self, mode_b_daemon) -> None:
        payload, first = _remember_and_enrich(mode_b_daemon, KESTREL, "g03-b-kestrel")
        assert [f["content"] for f in first] == [KESTREL], "first-queryable is the source"
        assert all(_no_invented_2004_date(f) for f in first)

        facts = _memory_facts(mode_b_daemon, KESTREL)
        live = [f for f in facts if not f["quarantined"]]
        withheld = [f for f in facts if f["quarantined"]]
        assert any(f["content"] == KESTREL for f in live), "the user's own words stay live"
        assert all(_no_invented_2004_date(f) for f in live), live
        assert {f["content"] for f in withheld} >= {"The recall happened on June 1st, 2004"}
        assert all(f["unresolved_reason"].startswith("source_fidelity:") for f in withheld)
        assert any("2004.6 ms" in f["content"] for f in live if f["content"] != KESTREL), (
            "a faithful model fact is kept"
        )

        contents = _recall_contents(mode_b_daemon, "When was the Kestrel checkpoint recalled")
        assert contents, "the memory is still found"
        assert not any("2004-06" in c or "June 1st, 2004" in c for c in contents), contents

    def test_negation_order_and_status_are_preserved(self, mode_b_daemon) -> None:
        _remember_and_enrich(mode_b_daemon, POLICY, "g03-b-policy")
        facts = _memory_facts(mode_b_daemon, POLICY)
        by_text = {f["content"]: f for f in facts}
        for text in BAD_TEXTS & set(by_text):
            assert by_text[text]["quarantined"] == 1, text
        assert BAD_TEXTS & set(by_text), "the stub's damaged facts reached the store"
        assert by_text["The team uses GitHub Actions"]["quarantined"] == 0
        assert by_text[POLICY]["quarantined"] == 0, "the verbatim memory is never withheld"
        contents = _recall_contents(mode_b_daemon, "publish approval release owner")
        assert "Publish with approval from the release owner" not in contents
        assert any("Never publish without approval" in c for c in contents), contents

    def test_withheld_facts_are_offered_for_review(self, mode_b_daemon) -> None:
        listing = _fidelity_listing(mode_b_daemon)
        withheld_ids = {f["fact_id"] for f in listing["withheld"]["facts"]}
        stored = _rows(mode_b_daemon, "SELECT fact_id FROM atomic_facts WHERE quarantined = 1")
        assert withheld_ids == {r["fact_id"] for r in stored}
        assert listing["withheld"]["total"] >= 4
        reasons = {r for f in listing["withheld"]["facts"] for r in f["reasons"]}
        assert {"number_became_date", "negation_lost", "order_reversed",
                "status_lost"} <= reasons, reasons


    def test_correction_apply_and_rollback_keep_the_measurement(self, mode_b_daemon) -> None:
        facts = _memory_facts(mode_b_daemon, KESTREL)
        target = next(f for f in facts if f["content"] == "The Kestrel checkpoint recall took "
                      "2004.6 ms")
        corrected = "The Kestrel checkpoint recall took 2004.6 ms at rank 1"
        code, proposed = mode_b_daemon.request(
            "PATCH", f"/api/memories/{target['fact_id']}", {"content": corrected})
        assert code in (200, 202) and proposed.get("success"), proposed
        case = proposed["correction_case"]
        successor = proposed["successor_fact_id"]

        def successor_row() -> dict:
            return _rows(mode_b_daemon, "SELECT content, referenced_date, interval_start, "
                                        "interval_end FROM atomic_facts WHERE fact_id=?",
                         (successor,))[0]

        assert successor_row()["content"] == corrected
        assert _no_invented_2004_date(successor_row())
        for action in ("apply", "rollback"):
            code, reviewed = mode_b_daemon.request(
                "POST", f"/api/corrections/{case['case_id']}/{action}",
                {"expected_version": case["version"]})
            assert code == 200 and reviewed.get("success"), (action, reviewed)
            case = reviewed["correction_case"]
            live = [f for f in _memory_facts(mode_b_daemon, KESTREL) if not f["quarantined"]]
            assert all(_no_invented_2004_date(f) for f in live), (action, live)
            assert _no_invented_2004_date(successor_row())
            contents = _recall_contents(mode_b_daemon, "Kestrel checkpoint recall time")
            assert not any("2004-06" in c or "June 1st, 2004" in c for c in contents)


class TestModeARules:
    def test_rules_path_keeps_measurements_and_polarity(self, mode_a_daemon) -> None:
        for key, content in (("g03-a-kestrel", KESTREL), ("g03-a-policy", POLICY)):
            _, first = _remember_and_enrich(mode_a_daemon, content, key)
            assert [f["content"] for f in first] == [content]
            facts = _memory_facts(mode_a_daemon, content)
            assert facts and all(_no_invented_2004_date(f) for f in facts), facts
            assert all(f["quarantined"] == 0 for f in facts), "nothing faithful is withheld"
        policy = " ".join(f["content"] for f in _memory_facts(mode_a_daemon, POLICY))
        assert "Never publish without approval" in policy
        assert policy.index("iOS") < policy.index("Android")
        assert "Previously the team used Jenkins" in policy
