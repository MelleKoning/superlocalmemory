# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A kind-filtered recall finds the memory of that kind behind many closer ones.

4.1.21 applied ``kind`` only after fusion. Fusion takes each channel's top 50,
so when more than 50 memories of OTHER kinds match the question better, the one
memory of the asked-for kind never became a candidate and the recall came back
empty, although the memory was stored and confirmed. Measured on a copy of a
21,739-fact store: 3 of 10 everyday ``kind=decision`` questions returned
nothing.

Composed path only: a real daemon subprocess on a private port with its own
data folder, saves and recalls over its HTTP API, and the database read back
directly. Synthetic, de-identified fixture text.
"""
from __future__ import annotations

import uuid

import pytest

from tests.test_integration.test_mcp_declared_kind_transport import _start_daemon
from tests.test_integration.test_per_request_profile_e2e import (
    PRODUCTION_PORTS,
    _foreign_daemon_pids,
    _reserve_private_port,
)

RUN = uuid.uuid4().hex[:6].upper()
#: More than any channel's candidate width (semantic 50, BM25 50).
CROWD = 90
QUESTION = f"kestrel checkpoint deployment parameters window {RUN}"


@pytest.fixture(scope="module")
def daemon(tmp_path_factory):
    root = tmp_path_factory.mktemp("kind-crowd")
    data_root = root / "data"
    data_root.mkdir()
    port = _reserve_private_port()
    assert port not in PRODUCTION_PORTS
    foreign = _foreign_daemon_pids()
    d = _start_daemon(data_root, port, root)
    try:
        yield d
    finally:
        d.stop(foreign)


def _save(daemon, content: str, kind: str, key: str) -> dict:
    code, out = daemon.request("POST", "/remember", {
        "content": content, "kind": kind, "idempotency_key": key,
    })
    assert code in (200, 202) and out.get("ok") is True, out
    if code == 202:
        daemon._wait_committed(out["profile"], key)
    return out


def test_the_decision_behind_a_crowd_of_closer_memories_is_found(daemon) -> None:
    for i in range(CROWD):
        _save(daemon, f"Kestrel checkpoint deployment parameters window {RUN}, "
                      f"synthetic status note {i}.", "status", f"crowd-{RUN}-{i}")
    decided = _save(daemon, f"Kestrel checkpoint {RUN}: the team chose a manual "
                            "approval gate.", "decision", f"decision-{RUN}")
    code, unfiltered = daemon.request("GET", "/recall", params={"q": QUESTION, "limit": 50})
    assert code == 200, unfiltered
    code, out = daemon.request("GET", "/recall",
                               params={"q": QUESTION, "kind": "decision", "limit": 10})
    assert code == 200, out
    shown = {r["fact_id"] for r in out.get("results", [])}
    assert set(decided["fact_ids"]) & shown, {
        "decision": decided["fact_ids"], "shown": sorted(shown),
        "unfiltered_has_it": bool(set(decided["fact_ids"])
                                  & {r["fact_id"] for r in unfiltered.get("results", [])}),
    }
    assert all(r.get("memory_kind") == "decision" for r in out["results"]), out["results"]
