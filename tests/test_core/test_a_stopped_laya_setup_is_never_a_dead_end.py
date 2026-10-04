# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A setup that stops part-way must leave the person a way forward.

From a real 4.1.19 machine: the weights download hung at 35% on a network that
blocks the model host; the dashboard showed "Downloading the model weights…"
for half an hour; once it was stopped, the folder it left behind (a lock, the
step stamps, a partial environment and download, never marked finished) read
as "not installed", and Remove answered "No managed install to remove." —
so nothing in the dashboard could clear it.

Every subprocess here is a local fake; nothing touches the network.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from superlocalmemory.core import laya_process
from superlocalmemory.core import laya_runtime as lr


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(lr, "_apple_silicon", lambda: True)
    for job in (lr.LayaInstallJob, lr.LayaAdoptJob, lr.LayaTestJob):
        job._instance = None
    laya_process.CANCEL.clear()
    yield
    laya_process.CANCEL.clear()
    for job in (lr.LayaInstallJob, lr.LayaAdoptJob, lr.LayaTestJob):
        job._instance = None


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) if not isinstance(data, str) else data)


def _stopped_setup() -> Path:
    """Exactly what the aborted download left on the real machine."""
    run_dir = lr.runtime_dir()
    _write(run_dir / ".install-steps.json", {"venv": True, "pip": True})
    _write(run_dir / "install.lock", "")
    (run_dir / "venv" / "lib").mkdir(parents=True)
    _write(run_dir / "hf-cache" / "hub" / "blob.incomplete", "x" * 1024)
    return run_dir


def _external_install(tmp_path: Path) -> tuple[Path, Path]:
    python = tmp_path / "ext" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    _write(tmp_path / "ext" / "pyvenv.cfg", "home = /usr/bin\n")
    model = tmp_path / "ext-model"
    model.mkdir()
    return python, model


def _fake_python(tmp_path: Path, body: str) -> Path:
    """An executable that stands in for the environment's python."""
    exe = tmp_path / "fakepy"
    exe.write_text("#!/bin/sh\n" + body + "\n")
    exe.chmod(0o755)
    return exe


# -- detect: truthful about a stopped setup --------------------------------------

def test_a_setup_that_never_finished_reads_as_unfinished_not_missing():
    _stopped_setup()
    status = lr.detect()
    assert status.state == lr.STATE_FAILED
    assert status.managed is True
    assert status.action == lr.ACTION_SETUP
    assert status.error == lr.UNFINISHED


def test_an_install_made_elsewhere_is_offered_a_check_never_a_download(tmp_path):
    python, model = _external_install(tmp_path)
    cfg = SimpleNamespace(sufficiency_python=str(python), sufficiency_model=str(model),
                          sufficiency_hf_home="")
    status = lr.detect(cfg)
    assert status.state == lr.STATE_FAILED
    assert status.action == lr.ACTION_CHECK
    assert status.error == lr.UNCHECKED
    assert "Set up" not in status.step


def test_an_adopted_install_that_moved_says_so(tmp_path):
    _write(lr.runtime_dir() / "adopted.json", {
        "python": str(tmp_path / "gone" / "bin" / "python"),
        "model_path": str(tmp_path / "gone-model"), "verified": True})
    status = lr.detect()
    assert status.state == lr.STATE_FAILED
    assert status.action == lr.ACTION_CHECK
    assert "moved" in status.error


def test_the_reason_a_setup_failed_is_shown_after_it_failed(monkeypatch):
    monkeypatch.setattr(lr, "_check_disk_space", lambda p: True)
    monkeypatch.setattr(lr, "_create_venv", lambda *a, **k: (True, "", ""))
    monkeypatch.setattr(lr, "_pip_install", lambda *a, **k: (True, "", ""))
    monkeypatch.setattr(lr, "_download_weights",
                        lambda *a, **k: (False, laya_process.KIND_STALLED, "stalled"))
    job = lr.LayaInstallJob.instance()
    assert job.start()
    deadline = time.monotonic() + 10
    while job.running and time.monotonic() < deadline:
        time.sleep(0.05)
    status = lr.detect()
    assert status.state == lr.STATE_FAILED
    assert status.action == lr.ACTION_SETUP
    assert "Can't reach the model download server" in status.error
    assert "Use an install you already have" in status.error


# -- remove: clears a stopped setup, keeps someone else's install ---------------

def test_remove_clears_a_setup_that_never_finished():
    run_dir = _stopped_setup()
    result = lr.remove()
    assert result.state == lr.STATE_NOT_INSTALLED
    assert not run_dir.exists()
    assert lr.detect().state == lr.STATE_NOT_INSTALLED


def test_remove_keeps_a_valid_adopted_record(tmp_path):
    run_dir = _stopped_setup()
    python, model = _external_install(tmp_path)
    adopted = {"python": str(python), "model_path": str(model), "verified": True}
    _write(run_dir / "adopted.json", adopted)
    assert lr.remove().state == lr.STATE_NOT_INSTALLED
    assert sorted(p.name for p in run_dir.iterdir()) == ["adopted.json"]
    assert json.loads((run_dir / "adopted.json").read_text()) == adopted
    assert lr.detect().state == lr.STATE_READY


def test_remove_drops_an_adopted_record_that_names_nothing(tmp_path):
    run_dir = _stopped_setup()
    _write(run_dir / "adopted.json", {"python": "/nowhere/pytest-of-x/venv/bin/python",
                                      "model_path": "/nowhere/model", "verified": True})
    assert lr.remove().state == lr.STATE_NOT_INSTALLED
    assert not run_dir.exists()


def test_forget_drops_only_the_record(tmp_path):
    python, model = _external_install(tmp_path)
    _write(lr.runtime_dir() / "adopted.json", {"python": str(python),
                                               "model_path": str(model), "verified": True})
    assert lr.forget_adopted() is True
    assert python.exists() and model.is_dir()
    assert lr.detect().state == lr.STATE_NOT_INSTALLED


# -- resuming a stopped setup -----------------------------------------------------

def test_a_new_environment_gets_its_package_again(monkeypatch):
    """The stamp said the package was installed — into an environment that is gone."""
    _stopped_setup()  # stamps venv+pip, but venv/bin/python does not exist
    pip_calls = []
    monkeypatch.setattr(lr, "_check_disk_space", lambda p: True)
    monkeypatch.setattr(lr, "_create_venv", lambda *a, **k: (True, "", ""))
    monkeypatch.setattr(lr, "_pip_install",
                        lambda *a, **k: pip_calls.append(1) or (True, "", ""))
    monkeypatch.setattr(lr, "_download_weights", lambda *a, **k: (True, "", ""))
    monkeypatch.setattr(lr, "verify", lambda *a, **k: (True, "ok"))
    assert lr.install().state == lr.STATE_READY
    assert pip_calls == [1]


# -- the download wait: stall, cancel, a chatty process ---------------------------

def test_a_download_that_stops_growing_is_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(lr, "_STALL_S", 1.5)
    exe = _fake_python(tmp_path, "sleep 60")
    started = time.monotonic()
    ok, kind, _ = lr._download_weights(exe, "r", "v", tmp_path / "hub",
                                       progress=None, timeout_s=60)
    assert (ok, kind) == (False, laya_process.KIND_STALLED)
    assert time.monotonic() - started < 10
    assert lr._failure_message(kind, doing="d", do="x") == lr._STALLED_MESSAGE


def test_a_growing_download_is_not_called_stalled(tmp_path, monkeypatch):
    monkeypatch.setattr(lr, "_STALL_S", 4.0)   # bytes arrive every 0.5 s: never stalled
    hub = tmp_path / "hub"
    exe = _fake_python(tmp_path, f'for i in 1 2 3 4 5 6 7 8; do printf xxxx >> "{hub}/part"; '
                                 "sleep 0.5; done; exit 0")
    seen = []
    ok, kind, _ = lr._download_weights(exe, "r", "v", hub,
                                       progress=lambda f, s: seen.append(s), timeout_s=60)
    assert ok is True, kind
    assert any("of 807 MB" in s for s in seen)


def test_cancel_stops_a_download_at_once(tmp_path):
    exe = _fake_python(tmp_path, "sleep 60")
    timer = threading.Timer(0.5, laya_process.CANCEL.set)
    timer.start()
    started = time.monotonic()
    ok, kind, _ = lr._download_weights(exe, "r", "v", tmp_path / "hub",
                                       progress=None, timeout_s=60)
    assert (ok, kind) == (False, laya_process.KIND_CANCELLED)
    assert time.monotonic() - started < 5


def test_a_download_that_writes_a_lot_to_stderr_does_not_freeze(tmp_path):
    """Progress bars go to stderr; an unread pipe used to fill and freeze it."""
    exe = _fake_python(tmp_path, "head -c 2000000 /dev/zero | tr '\\0' x >&2; exit 0")
    ok, kind, _ = lr._download_weights(exe, "r", "v", tmp_path / "hub",
                                       progress=None, timeout_s=8)
    assert ok is True, kind


def test_cancel_stops_a_package_install(tmp_path):
    exe = _fake_python(tmp_path, "sleep 60")
    timer = threading.Timer(0.5, laya_process.CANCEL.set)
    timer.start()
    ok, kind, _ = lr._run_subprocess([str(exe)], timeout_s=60)
    assert (ok, kind) == (False, laya_process.KIND_CANCELLED)


def test_the_install_job_cancels(tmp_path, monkeypatch):
    monkeypatch.setattr(lr, "_check_disk_space", lambda p: True)
    monkeypatch.setattr(lr, "_create_venv", lambda *a, **k: (True, "", ""))
    monkeypatch.setattr(lr, "_pip_install", lambda *a, **k: (True, "", ""))
    exe = _fake_python(tmp_path, "sleep 60")
    real = lr._download_weights
    monkeypatch.setattr(lr, "_download_weights",
                        lambda python, *a, **k: real(exe, *a, **k))
    job = lr.LayaInstallJob.instance()
    assert job.start()
    time.sleep(1.0)
    assert job.cancel() is True
    deadline = time.monotonic() + 10
    while job.running and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not job.running
    assert job.status().error == lr._CANCELLED_MESSAGE
    assert lr.detect().action == lr.ACTION_SETUP
    assert os.path.exists(lr.runtime_dir() / ".install-steps.json")
