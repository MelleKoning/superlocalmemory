# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Saving the same words again with a kind never drops that kind silently.

Identical words are stored once. 4.1.21 then kept whatever kind the stored
fact had and reported the second save as a success, on every door. Now an
unconfirmed stored fact takes the declared kind (auditable in the kind
history), and a fact already confirmed as another kind keeps it while the
save reports ``kind_conflict``.

Composed path only: a REAL daemon, a REAL ``slm mcp`` child, plain HTTP and
the real ``slm`` command line, all against one store.
"""
from __future__ import annotations

import json
import subprocess
import sys
import uuid

import pytest

from tests.test_integration.test_mcp_declared_kind_transport import _Lane
from tests.test_integration.test_per_request_profile_e2e import REPO_ROOT

RUN = uuid.uuid4().hex[:6].upper()


def _words(tag: str) -> str:
    return f"Fixture record KS{tag}{RUN}: the shared synthetic code is S{tag}."


@pytest.fixture(scope="module")
def lane(tmp_path_factory):
    lane = _Lane(tmp_path_factory.mktemp("kind-on-resave"))
    try:
        lane.open_mcp()
        yield lane
    finally:
        lane.close_mcp()
        lane.daemon.stop(lane.foreign)


def _http_remember(lane, words: str, kind: str | None = None) -> dict:
    body = {"content": words, "idempotency_key": uuid.uuid4().hex}
    if kind:
        body["kind"] = kind
    code, out = lane.daemon.request("POST", "/remember", body)
    assert code == 200, out
    return out


def _cli_remember(lane, words: str, kind: str) -> dict:
    done = subprocess.run(
        [sys.executable, "-m", "superlocalmemory.cli.main", "remember", words,
         "--kind", kind, "--json"],
        env=lane.daemon.env, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180,
    )
    assert done.returncode == 0, (done.stdout[-1500:], done.stderr[-1500:])
    return json.loads(done.stdout[done.stdout.index("{"):])["data"]


def _history(lane, fact_id: str) -> list[tuple]:
    return lane.rows("SELECT origin, new_kind, new_source, actor FROM memory_kind_history "
                     "WHERE fact_id=? ORDER BY history_id", (fact_id,))


def test_an_untyped_memory_takes_the_kind_declared_when_its_words_are_saved_again(lane):
    words = _words("A")
    first = lane.tool("remember", content=words)
    assert lane.kind_of(first["fact_ids"][0])[1] not in ("caller", "user")

    second = lane.tool("remember", content=words, kind="rule")

    assert second["success"] is True and "kind_conflict" not in second, second
    assert second["fact_ids"] == first["fact_ids"]
    assert lane.kind_of(first["fact_ids"][0]) == ("rule", "caller")
    [(origin, kind, source, actor)] = _history(lane, first["fact_ids"][0])
    assert (origin, kind, source) == ("user_edit", "rule", "caller")
    assert actor.startswith("daemon-capability:")


def test_over_http_a_different_confirmed_kind_is_kept_and_reported(lane):
    words = _words("B")
    first = _http_remember(lane, words, "decision")

    second = _http_remember(lane, words, "opinion")

    assert second["fact_ids"] == first["fact_ids"]
    assert second["kind_conflict"] == {"kept": "decision", "requested": "opinion"}
    assert lane.kind_of(first["fact_ids"][0]) == ("decision", "caller")
    assert _history(lane, first["fact_ids"][0]) == []


def test_the_same_kind_again_reports_nothing_and_changes_nothing(lane):
    words = _words("C")
    first = _http_remember(lane, words, "procedure")
    second = _http_remember(lane, words, "procedure")
    assert "kind_conflict" not in second, second
    assert lane.kind_of(first["fact_ids"][0]) == ("procedure", "caller")


def test_the_command_line_reports_the_kept_kind(lane):
    words = _words("D")
    _http_remember(lane, words, "status")

    out = _cli_remember(lane, words, "rule")

    assert out["kind_conflict"] == {"kept": "status", "requested": "rule"}, out
