# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Project-filtered recall: strict mode, crowding, identity and names (G09).

4.1.21 applied ``project`` only after fusion, like ``kind`` was. With more
than one channel's width of better matches saved under OTHER projects, the
project's own memory was never a candidate: the filter fell back to the
unfiltered results and said nothing was saved under the project, which was
untrue. These tests drive a real daemon over HTTP with its own data folder and
private port, synthetic de-identified text.
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
QUESTION = f"harbor relay rollout parameters window {RUN}"


@pytest.fixture(scope="module")
def daemon(tmp_path_factory):
    root = tmp_path_factory.mktemp("project-crowd")
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


def _save(daemon, content: str, project: str, key: str) -> dict:
    # The HTTP route takes the project in ``metadata``, as the MCP tool sends it.
    code, out = daemon.request("POST", "/remember", {
        "content": content, "metadata": {"project": project}, "idempotency_key": key,
    })
    assert code in (200, 202) and out.get("ok") is True, out
    if code == 202:
        daemon._wait_committed(out["profile"], key)
    return out


def _recall(daemon, **params) -> dict:
    code, out = daemon.request("GET", "/recall", params={"q": QUESTION, "limit": 10,
                                                         **params})
    assert code == 200, out
    return out


def _ids(out: dict) -> set[str]:
    return {r["fact_id"] for r in out.get("results", [])}


@pytest.fixture(scope="module")
def saved(daemon):
    for i in range(CROWD):
        _save(daemon, f"Harbor relay rollout parameters window {RUN}, synthetic "
                      f"status note {i}.", "/srv/other/lighthouse", f"crowd-{RUN}-{i}")
    mine = _save(daemon, f"Harbor relay {RUN}: the rollout uses a staged canary.",
                 "/Users/someone/work/a/Beacon", f"mine-{RUN}")
    twin = _save(daemon, f"Harbor relay {RUN}: the twin repository pins version 2.",
                 "/Volumes/ext/b/beacon", f"twin-{RUN}")
    return {"mine": set(mine["fact_ids"]), "twin": set(twin["fact_ids"])}


def test_a_project_memory_behind_a_crowd_of_closer_ones_is_found(daemon, saved):
    out = _recall(daemon, project="Beacon")
    shown = _ids(out)
    assert saved["mine"] & shown, {"mine": sorted(saved["mine"]), "shown": sorted(shown)}
    report = out["project_scope"]["filter"]
    assert report["applied"] is True and report["matched"] >= 1, report


def test_strict_keeps_only_the_project_and_says_so(daemon, saved):
    out = _recall(daemon, project="beacon", project_strict="true")
    shown = _ids(out)
    assert shown and shown <= (saved["mine"] | saved["twin"]), shown
    assert out["project_scope"]["filter"]["strict"] is True


def test_strict_with_no_memory_under_the_project_returns_nothing(daemon, saved):
    out = _recall(daemon, project=f"no-such-project-{RUN}", project_strict="true")
    assert out.get("results") == [], out.get("results")
    report = out["project_scope"]["filter"]
    assert report["strict"] is True and report["matched"] == 0
    assert report["reason"] == "no_match" and "strict" in report["note"]


def test_without_strict_the_fall_back_is_unchanged(daemon, saved):
    out = _recall(daemon, project=f"no-such-project-{RUN}")
    assert out.get("results"), "the 4.1.21 fall-back must still return results"
    report = out["project_scope"]["filter"]
    assert report["applied"] is False and report["strict"] is False
    assert report["reason"] == "no_match"


def test_a_name_shared_by_two_saved_paths_is_reported_as_ambiguous(daemon, saved):
    out = _recall(daemon, project="BEACON")
    identity = out["project_scope"]["filter"]["identity"]
    assert identity["name"] == "BEACON" and identity["rule"]
    assert identity.get("ambiguous") is True, identity
    assert identity["stored_as_count"] == 2


def test_a_full_path_and_any_case_name_the_same_project(daemon, saved):
    by_path = _ids(_recall(daemon, project="/Users/someone/work/a/Beacon/",
                           project_strict="true"))
    by_name = _ids(_recall(daemon, project="bEaCoN", project_strict="true"))
    assert by_path == by_name and saved["mine"] & by_path


def test_a_named_profile_is_routed_and_the_active_profile_is_kept(daemon, saved):
    code, before = daemon.request("GET", "/api/profiles/active")
    code2, out = daemon.request("GET", "/recall", params={
        "q": QUESTION, "project": "beacon", "project_strict": "true",
        "profile_id": "default"})
    assert code2 == 200, out
    if code == 200:
        _, after = daemon.request("GET", "/api/profiles/active")
        assert after == before
    code3, refused = daemon.request("GET", "/recall", params={
        "q": QUESTION, "project": "beacon", "project_strict": "true",
        "profile_id": f"absent-{RUN}"})
    assert code3 != 200 or refused.get("ok") is False or not refused.get("results"), refused
