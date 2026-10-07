# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``slm embedder``: the terminal client of /api/v3/embedding/reindex."""

from __future__ import annotations

from argparse import Namespace

from superlocalmemory.cli import embedder_cmd


def _capture(monkeypatch):
    calls = []

    def fake(method, path, body=None, **kw):
        calls.append((method, path, body, kw))
        return {"success": True, "job": {"job_id": 3, "kind": "rollback", "state": "queued",
                                         "from": "a::1", "to": "b::2", "done": 0, "total": 5}}

    monkeypatch.setattr(embedder_cmd, "daemon_request", fake)
    return calls


def test_rollback_waits_long_enough_for_the_previous_model_to_start(monkeypatch, capsys):
    calls = _capture(monkeypatch)
    embedder_cmd.cmd_embedder(Namespace(embedder_command="rollback", json=False))
    method, path, _body, kw = calls[0]
    assert (method, path) == ("POST", "/api/v3/embedding/reindex/rollback")
    assert kw["timeout_seconds"] >= 300, "a rollback probes the old model inside the request"
    assert "Job 3 (rollback)" in capsys.readouterr().out


def test_switch_sends_the_model_and_width_and_never_a_key(monkeypatch):
    calls = _capture(monkeypatch)
    embedder_cmd.cmd_embedder(Namespace(embedder_command="switch", json=False, model="m",
                                        dimension=384, provider="", endpoint="", no_wait=True))
    _method, path, body, _kw = calls[0]
    assert path == "/api/v3/embedding/reindex"
    assert body == {"model_name": "m", "dimension": 384}
