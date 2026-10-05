# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm view`` is a thin client of the dashboard's routes (issue #113).

``daemon_request`` is bound to a TestClient of the real routes, so these tests
run the terminal and the HTTP surface together: what the terminal says is what
the dashboard would say.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.cli import view_cmd
from superlocalmemory.cli.daemon import DaemonConflict, DaemonNotFound, DaemonUnprocessable
from superlocalmemory.server.routes import views as routes

from ._store import learning_db


def _bind(monkeypatch, tc: TestClient) -> list[str]:
    seen: list[str] = []

    def fake(method, path, body=None, *, preserve_conflict=False, preserve_not_found=False,
             preserve_unprocessable=False, **_kw):
        seen.append(f"{method} {path}")
        res = tc.request(method, path, json=body)
        detail = res.json().get("detail") if res.headers.get("content-type", "").startswith(
            "application/json") else None
        if res.status_code == 409 and preserve_conflict:
            raise DaemonConflict(str(detail or ""))
        if res.status_code == 404 and preserve_not_found:
            raise DaemonNotFound(404, detail.get("code", ""), detail.get("message", ""), path)
        if res.status_code == 422 and preserve_unprocessable:
            raise DaemonUnprocessable(detail.get("code", ""), detail.get("message", ""))
        return res.json() if res.status_code < 400 else None

    monkeypatch.setattr(view_cmd, "daemon_request", fake)
    return seen


@pytest.fixture()
def cli(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    learning_db(tmp_path)
    monkeypatch.setattr(routes, "_profile", lambda: "default")
    app = FastAPI()
    app.include_router(routes.router)
    app.state.engine = SimpleNamespace(
        _config=SimpleNamespace(), profile_id="default",
        _db=SimpleNamespace(get_memory_content_batch=lambda *a, **k: {}),
        recall=lambda query, **kw: SimpleNamespace(
            results=[_result("f-2", "Second memory"), _result("f-1", "First memory")],
            query_type="semantic", retrieval_time_ms=2.0, channel_weights={},
            no_confident_match=False))
    tc = TestClient(app)
    from superlocalmemory.core.security_primitives import ensure_install_token

    tc.headers["X-Install-Token"] = ensure_install_token()
    return _bind(monkeypatch, tc)


def _result(fact_id: str, content: str):
    fact = SimpleNamespace(fact_id=fact_id, memory_id=f"m-{fact_id}", content=content,
                           created_at="2026-10-05T00:00:00+00:00", fact_type=None,
                           lifecycle=None, access_count=0)
    return SimpleNamespace(fact=fact, score=0.8, confidence=0.7, trust_score=0.5,
                           channel_scores={}, evidence_chain=[])


def _parse(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="slm")
    parser.add_argument("--json", action="store_true")
    sub = parser.add_subparsers(dest="command")
    view_cmd.register_view_parser(sub)
    return parser.parse_args(list(argv))


def _run(capsys, *argv: str) -> tuple[str, str]:
    view_cmd.cmd_view(_parse(*argv))
    out = capsys.readouterr()
    return out.out, out.err


def test_create_list_run_rename_delete(cli, capsys) -> None:
    out, _ = _run(capsys, "view", "create", "Work log", "what shipped", "--window", "7d",
                  "--limit", "5")
    assert "Saved the view 'Work log'" in out and "[window=7d]" in out
    out, _ = _run(capsys, "view", "list")
    assert "Work log" in out and "up to 5 results" in out
    out, _ = _run(capsys, "view", "run", "work log")
    assert out.index("Second memory") < out.index("First memory"), "recall's order kept"
    assert "memory id: f-2" in out and "memory id: f-1" in out
    out, _ = _run(capsys, "view", "show", "Work log")
    assert "memory id: f-2" in out
    out, _ = _run(capsys, "view", "rename", "Work log", "Shipped")
    assert "Renamed the view to 'Shipped'" in out
    out, _ = _run(capsys, "view", "delete", "Shipped")
    assert "No memory was changed" in out
    out, _ = _run(capsys, "view", "list")
    assert "No saved views yet" in out


def test_names_with_slashes_and_spaces_reach_the_right_view(cli, capsys) -> None:
    _run(capsys, "view", "create", "a/b c?&", "q")
    out, _ = _run(capsys, "view", "run", "a/b c?&")
    assert "a/b c?&" in out
    assert any("/run?name=a%2Fb%20c%3F%26&via=cli" in call for call in cli)


def test_json_envelope(cli, capsys) -> None:
    _run(capsys, "view", "create", "W", "q")
    out, _ = _run(capsys, "view", "run", "W", "--json")
    data = json.loads(out)
    assert data["success"] is True and data["data"]["result_ids"] == ["f-2", "f-1"]
    out, _ = _run(capsys, "view", "--json", "list")
    assert json.loads(out)["data"]["count"] == 1


def test_refusals_exit_with_the_right_code(cli, capsys) -> None:
    with pytest.raises(SystemExit) as bad:
        _run(capsys, "view", "create", "W", "q", "--window", "fortnight")
    assert bad.value.code == 2
    assert "window" in capsys.readouterr().err
    with pytest.raises(SystemExit) as missing:
        _run(capsys, "view", "run", "nope")
    assert missing.value.code == 1
    assert "no saved view called 'nope'" in capsys.readouterr().err
    _run(capsys, "view", "create", "W", "q")
    with pytest.raises(SystemExit) as dup:
        _run(capsys, "view", "create", "w", "other")
    assert dup.value.code == 1 and "already exists" in capsys.readouterr().err


def test_json_refusal_is_still_json(cli, capsys) -> None:
    with pytest.raises(SystemExit):
        _run(capsys, "view", "run", "nope", "--json")
    data = json.loads(capsys.readouterr().out)
    assert data["success"] is False and data["error"]["code"] == "view_not_found"


def test_no_subcommand_prints_usage(cli, capsys) -> None:
    out, _ = _run(capsys, "view")
    assert "slm view create" in out


def test_daemon_down_says_so(monkeypatch, capsys) -> None:
    monkeypatch.setattr(view_cmd, "daemon_request", lambda *a, **k: None)
    with pytest.raises(SystemExit) as info:
        _run(capsys, "view", "list")
    assert info.value.code == 1 and "slm serve" in capsys.readouterr().err


def test_view_is_in_the_help_groups() -> None:
    from superlocalmemory.cli.commands import _COMMAND_GROUPS

    names = {name for _group, items in _COMMAND_GROUPS for name, _text in items}
    assert "view" in names
