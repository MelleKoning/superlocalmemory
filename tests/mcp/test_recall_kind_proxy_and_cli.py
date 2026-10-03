# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The ``kind`` filter reaches the daemon identically from the MCP proxy and
the CLI (LLD/WP8 4.1.19) — mirrors test_retrieval/test_recall_facets.py's
proxy/CLI tests for project/saved_by/about.
"""

from __future__ import annotations

from argparse import Namespace


def test_mcp_proxy_sends_kind_only_when_set(monkeypatch) -> None:
    from superlocalmemory.cli import daemon
    from superlocalmemory.mcp import _daemon_proxy

    paths = []
    monkeypatch.setattr(daemon, "daemon_request",
                        lambda method, path, *a, **k: paths.append(path) or {"results": []})
    proxy = _daemon_proxy.DaemonPoolProxy(port=8765)
    proxy.recall("q", kind="decision")
    assert "kind=decision" in paths[0]
    paths.clear()
    proxy.recall("q")
    assert "kind=" not in paths[0]


def _recall_args(**overrides) -> Namespace:
    base = dict(query="what did we decide", limit=5, json=True, project="", saved_by="",
               about="", fast=None, window="", as_of="", known_as_of="", valid_at="",
               include_unknown=False, include_global=None, include_shared=None,
               session_id="", profile_id="", kind="")
    base.update(overrides)
    return Namespace(**base)


def test_cli_recall_adds_kind_to_the_request(monkeypatch) -> None:
    from superlocalmemory.cli import commands, daemon

    paths = []
    monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)
    monkeypatch.setattr(daemon, "daemon_request",
                        lambda method, path, *a, **k: paths.append(path) or {"results": []})
    try:
        commands.cmd_recall(_recall_args(kind="decision"))
    except SystemExit:
        pass
    assert paths, "the CLI did not call the daemon"
    assert "&kind=decision" in paths[0]


def test_cli_recall_refuses_an_unknown_kind_before_any_request(monkeypatch, capsys) -> None:
    import pytest

    from superlocalmemory.cli import commands, daemon

    called = []
    monkeypatch.setattr(daemon, "is_daemon_running", lambda: called.append(1) or True)
    with pytest.raises(SystemExit) as exited:
        commands.cmd_recall(_recall_args(kind="not-a-real-kind"))
    assert exited.value.code == 2
    assert not called, "the daemon must not be probed for a kind that never parsed"
