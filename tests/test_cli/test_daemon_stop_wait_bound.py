# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""4.1.22 polish: `slm serve stop` against a daemon wedged in "starting".

Before this fix the wait was a bare constant (``_STOP_START_WAIT_S = 90.0``),
not configurable, and `slm serve stop` gave no feedback for up to 90 s before
printing the misleading "Daemon was not running." for a daemon that was, in
fact, alive the whole time -- just stuck. This pins:

* the bound is now a function of ``SLM_DAEMON_STOP_WAIT_S``, clamped sanely;
* the CLI says what it is waiting for before the wait, and reports the truth
  (not "not running") when the wait times out on a daemon still alive;
* no code path in this file kills anything by process-name pattern -- only
  ``os.kill``/``psutil.Process`` against a PID read from the descriptor/lock.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", 90.0), ("30", 30.0), ("0", 0.0), ("-5", 0.0),
        ("999", 180.0), ("nan", 90.0), ("junk", 90.0),
    ],
)
def test_stop_wait_budget_is_clamped(monkeypatch, raw, expected):
    from superlocalmemory.cli.daemon_startup import stop_wait_budget

    monkeypatch.setenv("SLM_DAEMON_STOP_WAIT_S", raw)
    assert stop_wait_budget() == expected


def test_stop_wait_budget_defaults_without_the_env_var(monkeypatch):
    from superlocalmemory.cli.daemon_startup import stop_wait_budget

    monkeypatch.delenv("SLM_DAEMON_STOP_WAIT_S", raising=False)
    assert stop_wait_budget() == 90.0


def test_stop_daemon_uses_the_configurable_budget_not_a_bare_constant():
    from superlocalmemory.cli import daemon as _daemon

    source = inspect.getsource(_daemon.stop_daemon)
    assert "_startup.stop_wait_budget()" in source
    assert "_STOP_START_WAIT_S" not in source
    assert not hasattr(_daemon, "_STOP_START_WAIT_S")


class TestCmdServeStopMessaging:
    def _descriptor(self, *, state: str, pid: int = 4242):
        return SimpleNamespace(state=state, pid=pid)

    def test_prints_what_it_is_waiting_for_before_stopping_a_starting_daemon(
        self, capsys,
    ):
        from superlocalmemory.cli import commands

        starting = self._descriptor(state="starting")
        args = SimpleNamespace(action="stop")
        with patch("superlocalmemory.cli.daemon.read_descriptor", return_value=starting), \
             patch("superlocalmemory.cli.daemon._descriptor_process_is_alive", return_value=True), \
             patch("superlocalmemory.cli.daemon.stop_daemon", return_value=True), \
             patch("superlocalmemory.cli.daemon_startup.stop_wait_budget", return_value=42.0):
            commands.cmd_serve(args)

        out = capsys.readouterr().out
        assert "still starting" in out
        assert "42s" in out
        assert "Daemon stopped." in out

    def test_reports_the_truth_when_a_stuck_starting_daemon_cannot_be_stopped(
        self, capsys,
    ):
        """This must NOT say "Daemon was not running." -- it is."""
        from superlocalmemory.cli import commands

        stuck = self._descriptor(state="starting")
        args = SimpleNamespace(action="stop")
        with patch("superlocalmemory.cli.daemon.read_descriptor", return_value=stuck), \
             patch("superlocalmemory.cli.daemon._descriptor_process_is_alive", return_value=True), \
             patch("superlocalmemory.cli.daemon.stop_daemon", return_value=False), \
             patch("superlocalmemory.cli.daemon_startup.stop_wait_budget", return_value=1.0):
            commands.cmd_serve(args)

        out = capsys.readouterr().out
        assert "was not running" not in out.lower()
        assert "still starting" in out
        assert "may still be running" in out

    def test_says_not_running_when_it_truly_is_not(self, capsys):
        from superlocalmemory.cli import commands

        args = SimpleNamespace(action="stop")
        with patch("superlocalmemory.cli.daemon.read_descriptor", return_value=None), \
             patch("superlocalmemory.cli.daemon.stop_daemon", return_value=False):
            commands.cmd_serve(args)

        out = capsys.readouterr().out
        assert "Daemon was not running." in out

    def test_ready_daemon_stops_without_the_starting_message(self, capsys):
        from superlocalmemory.cli import commands

        ready = self._descriptor(state="ready")
        args = SimpleNamespace(action="stop")
        with patch("superlocalmemory.cli.daemon.read_descriptor", return_value=ready), \
             patch("superlocalmemory.cli.daemon.stop_daemon", return_value=True):
            commands.cmd_serve(args)

        out = capsys.readouterr().out
        assert "still starting" not in out
        assert "Daemon stopped." in out


class TestNoProcessNamePatternKilling:
    """LANE-RULES (permanent): never kill processes by pattern, only the PID
    the descriptor/lock names. Source-level guard against reintroducing it."""

    def test_stop_daemon_never_scans_processes_by_name(self):
        from superlocalmemory.cli import daemon as _daemon

        source = inspect.getsource(_daemon.stop_daemon)
        for banned in ("process_iter", "pkill", "killall", "pgrep"):
            assert banned not in source, f"{banned} must not appear in stop_daemon"

    def test_cmd_serve_stop_never_scans_processes_by_name(self):
        from superlocalmemory.cli import commands

        source = inspect.getsource(commands.cmd_serve)
        for banned in ("process_iter", "pkill", "killall", "pgrep"):
            assert banned not in source, f"{banned} must not appear in cmd_serve"
