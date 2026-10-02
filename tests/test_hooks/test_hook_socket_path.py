# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The prompt hook's fast socket works when the data folder's path is long.

A Unix socket path is limited to about 104 bytes on macOS. With a long data
folder the daemon logged "AF_UNIX path too long" and every prompt hook fell
back to the slow path. A path that does not fit now moves to a short,
per-user, owner-only folder that the hook client computes the same way — and
a folder someone else owns is never used.
"""

from __future__ import annotations

import os
import shutil
import socket
import stat
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    getattr(socket, "AF_UNIX", None) is None or os.name != "posix",
    reason="Unix sockets",
)


@pytest.fixture
def short_base(monkeypatch):
    """A short private base standing in for /tmp/slm-<uid>."""
    from superlocalmemory.hooks import hook_daemon

    parent = Path(tempfile.mkdtemp(prefix="slmb", dir="/tmp"))
    base = parent / "u"
    monkeypatch.setattr(hook_daemon, "_short_socket_base", lambda: base)
    yield base
    shutil.rmtree(parent, ignore_errors=True)


@pytest.fixture
def long_data_dir(tmp_path, monkeypatch):
    deep = tmp_path / ("a-rather-long-folder-name-" * 4) / "superlocalmemory-data"
    deep.mkdir(parents=True)
    monkeypatch.setenv("SLM_DATA_DIR", str(deep))
    return deep


def test_the_daemon_starts_and_the_hook_reaches_it_with_a_long_data_folder(
    long_data_dir, short_base,
):
    from superlocalmemory.hooks.hook_daemon import HookDaemon, _default_sock_path, try_socket_recall

    assert len(os.fsencode(str(_default_sock_path()))) > 104
    daemon = HookDaemon(queue_db_path=long_data_dir / "recall_queue.db")
    daemon.start()
    try:
        # An acknowledgement prompt needs no recall: it proves the hook client
        # found the same socket the daemon bound.
        assert try_socket_recall(prompt="ok", session_id="s", timeout=5.0) == {}
    finally:
        daemon.stop()


def test_the_short_folder_and_socket_are_owner_only(long_data_dir, short_base):
    from superlocalmemory.hooks.hook_daemon import HookDaemon

    daemon = HookDaemon(queue_db_path=long_data_dir / "recall_queue.db")
    daemon.start()
    try:
        assert stat.S_IMODE(short_base.lstat().st_mode) == 0o700
        sockets = list(short_base.iterdir())
        assert len(sockets) == 1
        assert stat.S_IMODE(sockets[0].lstat().st_mode) == 0o600
    finally:
        daemon.stop()
    assert list(short_base.iterdir()) == []


def test_a_short_data_folder_keeps_the_socket_where_it_was(short_base, monkeypatch):
    from superlocalmemory.hooks.hook_daemon import resolve_sock_path

    short = Path(tempfile.mkdtemp(prefix="slmd", dir="/tmp"))
    try:
        path = short / "hook_daemon.sock"
        assert resolve_sock_path(path) == path
    finally:
        shutil.rmtree(short, ignore_errors=True)


def test_two_data_folders_never_share_a_socket(short_base):
    from superlocalmemory.hooks.hook_daemon import resolve_sock_path

    first = Path("/" + "x" * 120) / "one" / "hook_daemon.sock"
    second = Path("/" + "x" * 120) / "two" / "hook_daemon.sock"
    a = resolve_sock_path(first, create=True)
    b = resolve_sock_path(second, create=True)
    assert a is not None and b is not None and a != b
    assert len(os.fsencode(str(a))) <= 103


def test_a_short_folder_owned_by_someone_else_is_never_used(short_base, monkeypatch):
    from superlocalmemory.hooks import hook_daemon

    short_base.mkdir(mode=0o700)
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)
    path = Path("/" + "x" * 120) / "hook_daemon.sock"
    assert hook_daemon.resolve_sock_path(path, create=True) is None
    assert hook_daemon.try_socket_recall(sock_path=path, prompt="ok") is None


def test_a_short_folder_that_is_a_symlink_is_never_used(short_base, tmp_path):
    from superlocalmemory.hooks.hook_daemon import resolve_sock_path

    target = Path(tempfile.mkdtemp(prefix="slmt", dir="/tmp"))
    try:
        short_base.symlink_to(target)
        assert resolve_sock_path(Path("/" + "x" * 120) / "s.sock", create=True) is None
    finally:
        shutil.rmtree(target, ignore_errors=True)
