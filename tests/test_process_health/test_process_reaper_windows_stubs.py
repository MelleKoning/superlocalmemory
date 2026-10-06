# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""On Windows the process reaper does nothing, and says so.

The reaper finds orphans with ``ps`` and stops them with POSIX signals, so
process_reaper.py defines no-op stubs on Windows. This loads the module the
way Windows does (``sys.platform == "win32"``) on any machine and checks the
stubs never find, signal or kill anything, and report it as unsupported.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

import superlocalmemory.infra.process_reaper as posix_reaper


@pytest.fixture()
def windows_reaper(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("the Windows reaper must not signal or run anything")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "kill", refuse)
    spec = importlib.util.spec_from_file_location("_reaper_as_on_windows", posix_reaper.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)  # dataclasses look it up
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.subprocess, "run", refuse)
    return module


def test_it_finds_nothing(windows_reaper):
    config = windows_reaper.ReaperConfig(orphan_age_threshold_hours=0.0)
    assert windows_reaper.find_slm_processes() == []
    assert windows_reaper.find_orphans(config) == []


@pytest.mark.parametrize("pid", [0, 1, os.getpid(), 4242])
def test_it_never_kills_and_says_it_is_unsupported(windows_reaper, pid):
    outcome = windows_reaper.kill_orphan(pid)
    assert outcome["killed"] is False
    assert outcome["method"] == "unsupported"
    assert outcome["error"]


def test_startup_and_cleanup_report_zero(windows_reaper, tmp_path):
    from superlocalmemory.infra.pid_manager import PidManager

    config = windows_reaper.ReaperConfig()
    started = windows_reaper.reap_stale_on_startup(config, PidManager(tmp_path / "slm.pids"))
    assert started["orphans_found"] == 0 and started["orphans_killed"] == 0
    cleaned = windows_reaper.cleanup_all_orphans(config, force=True)
    assert cleaned["killed"] == 0 and cleaned["processes"] == []
