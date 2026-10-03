# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""L3-11: ``slm kinds`` reports a 422 as invalid input (exit 2, the real
message) and a 404 as what was actually asked (not a fixed phrase written for
one specific caller). Routed through the REAL ``cli.daemon.daemon_request`` —
only ``urllib.request.urlopen`` is faked, so this exercises daemon.py's own
status-code-to-exception mapping, not a reimplementation of it.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from argparse import Namespace
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from superlocalmemory.cli import kinds_cmd
from tests.test_server.test_memory_kinds_routes import _client, _fact


def _bind_real_daemon_request(monkeypatch, tc) -> None:
    from superlocalmemory.cli import daemon as d

    desc = SimpleNamespace(port=59999, capability="cap", instance_id="inst")
    monkeypatch.setattr(d, "read_descriptor", lambda: desc)
    monkeypatch.setattr(d, "_fetch_health", lambda port: {"ok": True})
    monkeypatch.setattr(d, "descriptor_matches_health", lambda a, b: True)
    monkeypatch.setattr(d, "is_daemon_running", lambda: True)

    def fake_urlopen(req, timeout=None):
        parts = urlsplit(req.full_url)
        path = parts.path + (("?" + parts.query) if parts.query else "")
        headers = {k: v for k, v in req.header_items()}
        resp = tc.request(req.get_method(), path, content=req.data, headers=headers)
        if resp.status_code >= 400:
            raise urllib.error.HTTPError(
                req.full_url, resp.status_code, "err", None, io.BytesIO(resp.content),
            )
        return io.BytesIO(resp.content)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def _args(**kw) -> Namespace:
    base = {"json": False, "kinds_command": None, "backfill_command": None}
    base.update(kw)
    return Namespace(**base)


def test_set_on_an_unknown_fact_reports_what_was_asked(tmp_path, monkeypatch, capsys) -> None:
    tc, app, db = _client(tmp_path, monkeypatch)
    _bind_real_daemon_request(monkeypatch, tc)
    with pytest.raises(SystemExit) as exited:
        kinds_cmd.cmd_kinds(_args(kinds_command="set", fact_id="no-such-fact", kind="rule"))
    out = capsys.readouterr().out
    assert exited.value.code == 1
    assert "Memory not found" in out
    assert "classification run" not in out


def test_confirm_overlong_fact_id_does_not_abort_the_good_item(
    tmp_path, monkeypatch, capsys,
) -> None:
    tc, app, db = _client(tmp_path, monkeypatch)
    good_fid = _fact(db, "We went with SQLite.")
    _bind_real_daemon_request(monkeypatch, tc)
    bad_fid = "a" * 500
    kinds_cmd.cmd_kinds(_args(
        json=True, kinds_command="confirm",
        items=[f"{good_fid}=decision", f"{bad_fid}=decision"],
    ))
    out = json.loads(capsys.readouterr().out)
    assert out["success"] is True
    items = out["data"]["items"]
    assert {i["fact_id"]: i["ok"] for i in items} == {good_fid: True, bad_fid: False}
    row = dict(db.execute(
        "SELECT memory_kind FROM atomic_facts WHERE fact_id=?", (good_fid,))[0])
    assert row["memory_kind"] == "decision"


def test_confirm_overlong_fact_id_http_door_is_not_a_batch_422(tmp_path, monkeypatch) -> None:
    """The HTTP door this CLI command talks to must itself answer per-item,
    not reject the whole request because one id is unusually long."""
    tc, app, db = _client(tmp_path, monkeypatch)
    good_fid = _fact(db, "We went with SQLite.")
    bad_fid = "a" * 500
    r = tc.post("/api/memory-kinds/confirm", json={"items": [
        {"fact_id": good_fid, "kind": "decision"},
        {"fact_id": bad_fid, "kind": "decision"},
    ]})
    assert r.status_code == 200, r.text
    items = r.json()["items"]
    assert items[0]["ok"] is True
    assert items[1]["ok"] is False
