"""GB6 (4.1.22): only the default data root's daemon may take legacy port 8767.

Pre-descriptor clients dial 8767 and know nothing about data roots. Before
4.1.22 any daemon took it unless SLM_DISABLE_LEGACY_PORT=1, so a second data
root started first redirected old clients into the wrong store.

These tests never touch the real 8767 (the owner's daemon may hold it): the
redirect is exercised on a free loopback port passed in explicitly.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import socket
from pathlib import Path

import pytest

from superlocalmemory.infra.data_root import DATA_ROOT_ALIASES, is_implicit_default_root


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for name in DATA_ROOT_ALIASES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("SLM_DISABLE_LEGACY_PORT", raising=False)
    return home


def test_no_environment_selection_is_the_default_root(clean_env):
    assert is_implicit_default_root(home=clean_env)


def test_another_selected_root_is_not_the_default(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "team-store"))
    assert not is_implicit_default_root(home=clean_env)


def test_selecting_the_default_folder_explicitly_still_counts(clean_env, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(clean_env / ".superlocalmemory"))
    assert is_implicit_default_root(home=clean_env)


def test_a_legacy_relocation_is_what_old_clients_mean(clean_env, tmp_path, monkeypatch):
    relocated = tmp_path / "relocated-store"
    default = clean_env / ".superlocalmemory"
    default.mkdir()
    (default / "config.json").write_text(json.dumps({"base_dir": str(relocated)}), encoding="utf-8")
    assert is_implicit_default_root(home=clean_env)  # no env: resolves to the relocation
    monkeypatch.setenv("SLM_DATA_DIR", str(relocated))
    assert is_implicit_default_root(home=clean_env)
    monkeypatch.setenv("SLM_DATA_DIR", str(default))
    assert not is_implicit_default_root(home=clean_env)  # the relocation is the default now


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _listening(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0


async def _attempt(legacy_port: int) -> tuple[bool, bool]:
    """(task_started, port_listening) for one redirect attempt."""
    from superlocalmemory.server.legacy_port import maybe_start_legacy_redirect

    task = maybe_start_legacy_redirect(_free_port(), legacy_port)
    await asyncio.sleep(0.3)
    listening = await asyncio.to_thread(_listening, legacy_port)
    if task is not None:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    return task is not None, listening


def test_the_default_root_daemon_takes_the_legacy_port(clean_env):
    port = _free_port()
    started, listening = asyncio.run(_attempt(port))
    assert started and listening


def test_a_second_data_root_never_takes_the_legacy_port(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "second-root"))
    port = _free_port()
    started, listening = asyncio.run(_attempt(port))
    assert not started and not listening


def test_the_opt_out_still_wins_on_the_default_root(clean_env, monkeypatch):
    monkeypatch.setenv("SLM_DISABLE_LEGACY_PORT", "1")
    port = _free_port()
    started, listening = asyncio.run(_attempt(port))
    assert not started and not listening


def test_the_daemon_lifespan_uses_the_ownership_rule():
    """The lifespan must go through the rule, not start the redirect directly."""
    from superlocalmemory.server import unified_daemon

    source = inspect.getsource(unified_daemon.lifespan)
    assert "_maybe_start_legacy_redirect(" in source
    assert "_start_legacy_redirect(identity" not in source
    assert "create_task(_start_legacy_redirect" not in source
