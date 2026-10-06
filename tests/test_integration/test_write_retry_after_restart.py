# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Saving the same memory again after SLM restarts returns the first save.

4.1.21 refused it: the daemon bound every idempotency key to the actor id of
its per-start private capability, which changes on every restart, so a retry
with the same key - an assistant's retry, or simply re-running the same
``slm remember`` command, which derives the same key - was refused as "a
different request". Without a declared kind the tool then reported that
refusal as a retryable DAEMON_UNAVAILABLE, so a client retried it forever.

Composed path only: a REAL daemon on a private port with its own data folder,
a REAL ``slm mcp`` stdio child, the real ``slm`` command line, and a real
restart of the daemon on the same store.
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
PLAIN = f"Fixture record RR{RUN}: the synthetic restart code is R771."
KINDED = f"Fixture record RK{RUN}: the synthetic restart code is R772."
CLI_WORDS = f"Fixture record RC{RUN}: the synthetic command line code is R773."


def _cli_remember(lane: _Lane, words: str) -> dict:
    done = subprocess.run(
        [sys.executable, "-m", "superlocalmemory.cli.main", "remember", words, "--json"],
        env=lane.daemon.env, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180,
    )
    assert done.returncode == 0, (done.stdout[-1500:], done.stderr[-1500:])
    text = done.stdout
    return json.loads(text[text.index("{"):])  # one (indented) JSON document


@pytest.fixture(scope="module")
def lane(tmp_path_factory):
    lane = _Lane(tmp_path_factory.mktemp("retry-after-restart"))
    try:
        lane.open_mcp()
        lane.saved["plain"] = lane.tool("remember", content=PLAIN,
                                        idempotency_key=f"fixture-plain-{RUN}")
        lane.saved["kinded"] = lane.tool("remember", content=KINDED, kind="decision",
                                         idempotency_key=f"fixture-kinded-{RUN}")
        lane.saved["cli"] = _cli_remember(lane, CLI_WORDS)
        for name, out in lane.saved.items():
            assert out.get("success", True) is True and (
                out.get("fact_ids") or out.get("data", {}).get("fact_ids")), (name, out)
        yield lane
    finally:
        lane.close_mcp()
        lane.daemon.stop(lane.foreign)


def _cli_fact_ids(out: dict) -> list:
    return out.get("fact_ids") or out.get("data", {}).get("fact_ids") or []


def test_a_key_reused_for_other_words_is_refused_and_not_retryable(lane):
    # No kind declared: 4.1.21 turned this 422 into a retryable outage.
    out = lane.tool("remember", content=f"Other words RX{RUN}, same key.",
                    idempotency_key=f"fixture-plain-{RUN}")
    assert out["success"] is False, out
    assert out["retryable"] is False, out
    assert out["code"] == "IDEMPOTENCY_CONFLICT", out


def test_after_a_restart_the_same_save_returns_the_first_memory(lane):
    lane.restart()
    again = lane.tool("remember", content=PLAIN, idempotency_key=f"fixture-plain-{RUN}")
    assert again.get("success") is True, again
    assert again["fact_ids"] == lane.saved["plain"]["fact_ids"], again


def test_after_a_restart_the_same_save_with_a_kind_returns_the_first_memory(lane):
    again = lane.tool("remember", content=KINDED, kind="decision",
                      idempotency_key=f"fixture-kinded-{RUN}")
    assert again.get("success") is True, again
    assert again["fact_ids"] == lane.saved["kinded"]["fact_ids"], again
    assert lane.kind_of(again["fact_ids"][0]) == ("decision", "caller")


def test_after_a_restart_rerunning_the_same_command_returns_the_first_memory(lane):
    again = _cli_remember(lane, CLI_WORDS)
    assert _cli_fact_ids(again) == _cli_fact_ids(lane.saved["cli"]), again


def test_after_a_restart_a_reused_key_is_still_refused(lane):
    out = lane.tool("remember", content=f"Other words RY{RUN}, same key.",
                    idempotency_key=f"fixture-plain-{RUN}")
    assert out["success"] is False and out["retryable"] is False, out
    assert out["code"] == "IDEMPOTENCY_CONFLICT", out


def test_only_one_memory_was_stored_for_each_save(lane):
    for words in (PLAIN, KINDED, CLI_WORDS):
        rows = lane.rows("SELECT COUNT(*) FROM memories WHERE content=?", (words,))
        assert rows[0][0] == 1, words
