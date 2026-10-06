# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Checking whether a process is alive must not use ``os.kill(pid, 0)`` on Windows.

On POSIX, signal 0 only checks that the process exists. On Windows signal 0 is
``CTRL_C_EVENT``: ``os.kill(pid, 0)`` calls ``GenerateConsoleCtrlEvent``, which
either fails with ``WinError 87`` ("The parameter is incorrect") when ``pid``
is not a console process group, or SENDS CTRL+C to that group when it is. So on
Windows every liveness check built on it was wrong: the PID file cleanup raised
(seen on the Windows CI runner), a live embedding or reranker worker looked
dead (so a duplicate 1.6 GB worker could be started), and a lock left by a
dead process was never recovered.

Windows cannot run here. Each test makes this process look like Windows —
``sys.platform == "win32"`` and an ``os.kill`` that does what Windows does with
signal 0 — and checks the answer is still right.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory"


def _dead_pid() -> int:
    """The PID of a process that has exited (and been reaped)."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


@pytest.fixture()
def windows_signal_zero(monkeypatch):
    """This process behaves like Windows for a liveness check."""
    sent: list[int] = []
    real_kill = os.kill

    def kill(pid: int, sig: int) -> None:
        if sig == 0:
            sent.append(pid)
            raise OSError(22, "The parameter is incorrect")  # WinError 87
        real_kill(pid, sig)  # the test's own clean-up of its child

    import psutil

    real_pid_exists = psutil.pid_exists

    def pid_exists(pid: int) -> bool:
        # psutil's Windows build asks OpenProcess, not signal 0. Its POSIX build
        # (the one installed here) uses os.kill, so answer with the real one.
        with monkeypatch.context() as m:
            m.setattr(os, "kill", real_kill)
            return real_pid_exists(pid)

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "kill", kill)
    monkeypatch.setattr(psutil, "pid_exists", pid_exists)
    return sent


def test_is_pid_alive_answers_without_signal_zero(windows_signal_zero):
    from superlocalmemory.core.platform_utils import is_pid_alive

    assert is_pid_alive(os.getpid()) is True
    assert is_pid_alive(_dead_pid()) is False
    assert windows_signal_zero == [], "a CTRL_C_EVENT would have been sent"


def test_the_pid_file_cleanup_keeps_the_living_and_drops_the_dead(
        windows_signal_zero, tmp_path):
    from superlocalmemory.infra.pid_manager import PidManager

    manager = PidManager(tmp_path / "slm.pids")
    dead = _dead_pid()
    manager.register(os.getpid(), os.getppid())
    manager.register(dead, os.getpid())

    assert manager.cleanup_dead() == 1
    assert [record.pid for record in manager.read_all()] == [os.getpid()]
    assert windows_signal_zero == []


@pytest.mark.parametrize("module, check, pid_file", [
    ("superlocalmemory.core.embeddings", "_is_embedding_worker_alive", "_embedding_pid_file"),
    ("superlocalmemory.retrieval.reranker", "_is_reranker_worker_alive", "_reranker_pid_file"),
])
def test_a_live_worker_is_seen_as_alive(windows_signal_zero, monkeypatch, tmp_path,
                                        module, check, pid_file):
    import importlib

    mod = importlib.import_module(module)
    path = tmp_path / "worker.pid"
    monkeypatch.setattr(mod, pid_file, lambda: path)

    path.write_text(str(os.getpid()), encoding="utf-8")
    assert getattr(mod, check)() is True, "a live worker looked dead: a second would start"
    assert path.exists()

    path.write_text(str(_dead_pid()), encoding="utf-8")
    assert getattr(mod, check)() is False
    assert not path.exists(), "the stale PID file was not cleaned up"
    assert windows_signal_zero == []


def test_a_lock_left_by_a_dead_process_is_recovered(windows_signal_zero, tmp_path):
    from superlocalmemory.core.scale_engine import ScaleEngineManager as ScaleEngine

    lock = tmp_path / "adopt.lock"
    lock.write_text(json.dumps({"pid": _dead_pid()}), encoding="utf-8")
    assert ScaleEngine._clear_dead_legacy_adoption_lock(lock) is True
    assert not lock.exists()

    lock.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    assert ScaleEngine._clear_dead_legacy_adoption_lock(lock) is False
    assert lock.exists(), "a live holder's lock was removed"
    assert windows_signal_zero == []


def test_a_live_daemon_beside_the_store_is_noticed(windows_signal_zero, tmp_path):
    from superlocalmemory.storage.migration_runner import _foreign_live_daemon

    memory_db = tmp_path / "memory.db"
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        (tmp_path / "daemon.pid").write_text(str(other.pid), encoding="utf-8")
        assert _foreign_live_daemon(memory_db) == other.pid
    finally:
        other.kill()
        other.wait()
    (tmp_path / "daemon.pid").write_text(str(other.pid), encoding="utf-8")
    assert _foreign_live_daemon(memory_db) is None
    assert windows_signal_zero == []


def test_nothing_in_the_product_probes_a_process_with_signal_zero():
    """One cross-platform check (core.platform_utils.is_pid_alive) does this."""
    allowed = {
        SRC / "core" / "platform_utils.py": "the POSIX branch of is_pid_alive itself",
        # Its signal-0 probes sit in the POSIX implementation; on Windows the
        # module defines no-op stubs instead and never reaches them.
        SRC / "infra" / "process_reaper.py": "POSIX-only implementation",
    }
    found = []
    for path in sorted(SRC.rglob("*.py")):
        if path in allowed:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "kill" and len(node.args) == 2
                    and isinstance(node.args[1], ast.Constant) and node.args[1].value == 0):
                found.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not found, "use platform_utils.is_pid_alive instead: " + ", ".join(found)
