# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

""""Use an existing install" must only ever run a Python interpreter, and only
one nobody else can change.

``adopt()`` used to run ANY existing path as "the interpreter" — /bin/sh, a
directory, a script in /Users/Shared that every account on the Mac can write —
and then saved it, so it ran again at every engine start. These tests pin the
rule (a regular executable file, inside a Python environment with pyvenv.cfg or
the interpreter SLM itself runs on, owned by this user or root, writable by no
one else, not in a shared or temporary folder) and that it is checked again
right before anything is executed.

Nothing here runs a real interpreter on an unsafe path: process creation is
recorded instead of performed wherever the rule should stop it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from superlocalmemory.core import laya_interpreter
from superlocalmemory.core import laya_runtime as lr


@pytest.fixture(autouse=True)
def _apple_silicon(monkeypatch):
    monkeypatch.setattr(lr, "_apple_silicon", lambda: True)


@pytest.fixture(autouse=True)
def _reset_jobs():
    lr.LayaInstallJob._instance = None
    yield
    lr.LayaInstallJob._instance = None


@pytest.fixture()
def spawned(monkeypatch):
    """Record every process laya_runtime tries to start; start none."""
    calls: list[list[str]] = []

    class _Recorder:
        def __init__(self, argv, **_kwargs):
            calls.append(list(argv))
            raise OSError("recorded, not run")

    monkeypatch.setattr(lr.subprocess, "Popen", _Recorder)
    return calls


@pytest.fixture()
def not_temp(monkeypatch):
    """tmp_path lives under the OS temp root; most tests need it to look normal."""
    monkeypatch.setattr(lr, "_is_in_temp_dir", lambda p: False)


def _venv(root: Path, *, target: Path | None = None, cfg: bool = True) -> Path:
    """A Python environment shape: <root>/bin/python (+ pyvenv.cfg)."""
    (root / "bin").mkdir(parents=True)
    if cfg:
        (root / "pyvenv.cfg").write_text("home = /usr/bin\n")
    python = root / "bin" / "python"
    if target is None:
        python.symlink_to(sys.executable)
    else:
        python.symlink_to(target)
    return python


def _script(path: Path, mode: int = 0o755) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\necho pwned\n")
    path.chmod(mode)
    return path


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------


class TestTheRule:
    def test_the_interpreter_slm_runs_on_is_accepted(self, safely_owned_interpreter):
        assert lr.check_interpreter(sys.executable) == ""

    def test_a_python_inside_an_environment_is_accepted(
            self, tmp_path, not_temp, safely_owned_interpreter):
        assert lr.check_interpreter(str(_venv(tmp_path / "venv"))) == ""

    def test_a_shell_is_refused(self):
        assert lr.check_interpreter("/bin/sh") != ""

    def test_a_folder_is_refused(self, tmp_path, not_temp):
        assert "interpreter" in lr.check_interpreter(str(tmp_path)).lower()

    def test_a_relative_path_is_refused(self):
        assert lr.check_interpreter("python3") != ""

    def test_a_script_outside_any_environment_is_refused(self, tmp_path, not_temp):
        assert lr.check_interpreter(str(_script(tmp_path / "tools" / "python"))) != ""

    def test_an_interpreter_others_can_change_is_refused(self, tmp_path, not_temp):
        target = _script(tmp_path / "real" / "python3", mode=0o775)
        python = _venv(tmp_path / "venv", target=target)
        assert "other" in lr.check_interpreter(str(python)).lower()

    def test_an_environment_folder_others_can_change_is_refused(self, tmp_path, not_temp):
        python = _venv(tmp_path / "venv")
        (tmp_path / "venv" / "bin").chmod(0o777)
        try:
            assert "other" in lr.check_interpreter(str(python)).lower()
        finally:
            (tmp_path / "venv" / "bin").chmod(0o755)

    def test_a_shared_folder_is_refused(self, tmp_path, monkeypatch, not_temp):
        shared = tmp_path / "Shared"
        monkeypatch.setattr(laya_interpreter, "SHARED_ROOTS", (shared,))
        python = _venv(shared / "venv")
        assert lr.check_interpreter(str(python)) != ""

    def test_a_temporary_folder_is_refused(self, tmp_path):
        python = _venv(tmp_path / "venv")
        assert "temporary" in lr.check_interpreter(str(python)).lower()


# ---------------------------------------------------------------------------
# adopt() never executes what the rule refuses
# ---------------------------------------------------------------------------


class TestAdoptRunsNothingUnsafe:
    @pytest.mark.parametrize("target", ["/bin/sh", "/usr/bin/osascript"])
    def test_system_programs_are_never_run(self, target, spawned):
        status = lr.adopt(target, "", "/usr")
        assert status.state == lr.STATE_FAILED
        assert spawned == []

    def test_a_folder_is_never_run(self, tmp_path, spawned, not_temp):
        (tmp_path / "model").mkdir()
        status = lr.adopt(str(tmp_path), "", str(tmp_path / "model"))
        assert status.state == lr.STATE_FAILED
        assert spawned == []

    def test_a_writable_interpreter_is_never_run(self, tmp_path, spawned, not_temp):
        target = _script(tmp_path / "real" / "python3", mode=0o777)
        python = _venv(tmp_path / "venv", target=target)
        (tmp_path / "model").mkdir()
        status = lr.adopt(str(python), "", str(tmp_path / "model"))
        assert status.state == lr.STATE_FAILED
        assert spawned == []
        assert not (lr.runtime_dir() / "adopted.json").exists()


# ---------------------------------------------------------------------------
# checked again right before anything runs
# ---------------------------------------------------------------------------


class TestCheckedAgainBeforeRunning:
    def test_verify_refuses_an_unsafe_interpreter_without_running_it(self, spawned):
        ok, reason = lr.verify("/bin/sh", "", "/usr")
        assert ok is False and reason
        assert spawned == []

    def test_an_adopted_install_changed_afterwards_is_no_longer_ready(self, tmp_path, not_temp):
        target = _script(tmp_path / "real" / "python3")
        python = _venv(tmp_path / "venv", target=target)
        model = tmp_path / "model"
        model.mkdir()
        record = {"python": str(python), "model_path": str(model), "verified": True}
        lr.runtime_dir().mkdir(parents=True, exist_ok=True)
        (lr.runtime_dir() / "adopted.json").write_text(json.dumps(record))
        assert lr.detect().state == lr.STATE_READY

        target.chmod(0o777)  # someone else can now swap what runs
        status = lr.detect()
        assert status.state == lr.STATE_FAILED
        assert "other" in status.error.lower()


# ---------------------------------------------------------------------------
# M-8: the canary never loads a second model, and never without a memory cap
# ---------------------------------------------------------------------------

_RECORDING_WORKER = r'''
import json, os, sys
log = os.environ["FAKE_LOG"]
for raw in sys.stdin:
    req = json.loads(raw)
    with open(log, "a") as fh:
        fh.write(json.dumps(req) + "\n")
    if req.get("cmd") == "load":
        print(json.dumps({"ok": True}), flush=True)
    elif req.get("cmd") == "judge":
        print(json.dumps({"ok": True, "probabilities": [0.9, 0.1]}), flush=True)
'''


@pytest.fixture()
def recording_worker(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text(_RECORDING_WORKER)
    log = tmp_path / "requests.jsonl"
    monkeypatch.setattr(lr, "WORKER_PATH", worker)
    monkeypatch.setenv("FAKE_LOG", str(log))
    return log


class TestTheCanaryIsBounded:
    def test_the_load_request_carries_a_memory_cap(
            self, recording_worker, tmp_path, safely_owned_interpreter):
        ok, _ = lr.verify(sys.executable, "", str(tmp_path))
        assert ok is True
        load = json.loads(recording_worker.read_text().splitlines()[0])
        assert load["cmd"] == "load"
        assert isinstance(load.get("memory_limit_mb"), int) and load["memory_limit_mb"] > 0

    def test_no_second_model_while_the_answer_check_holds_the_slot(
            self, recording_worker, tmp_path, monkeypatch, safely_owned_interpreter):
        from superlocalmemory.retrieval import sufficiency

        held = sufficiency._take_slot()
        assert held is not None
        try:
            spawned: list = []
            real_popen = subprocess.Popen
            monkeypatch.setattr(lr.subprocess, "Popen",
                                lambda *a, **k: spawned.append(a) or real_popen(*a, **k))
            ok, reason = lr.verify(sys.executable, "", str(tmp_path))
        finally:
            sufficiency._release_slot(held)
        assert ok is False
        assert "running" in reason.lower()
        assert spawned == []

    def test_the_slot_is_given_back_afterwards(
            self, recording_worker, tmp_path, safely_owned_interpreter):
        from superlocalmemory.retrieval import sufficiency

        assert lr.verify(sys.executable, "", str(tmp_path))[0] is True
        again = sufficiency._take_slot()
        assert again is not None
        sufficiency._release_slot(again)

    def test_a_busy_check_does_not_mark_a_working_install_broken(
            self, monkeypatch, tmp_path):
        run_dir = lr.runtime_dir()
        venv_python = _venv(run_dir / "venv")
        model = run_dir / "model"
        model.mkdir(parents=True)
        record = {"python": str(venv_python), "model_path": str(model), "verified": True,
                  "verified_at": "earlier"}
        (run_dir / ".slm-managed").write_text(json.dumps(record))
        monkeypatch.setattr(lr, "verify", lambda *a, **k: (False, lr.VERIFY_BUSY))

        ok, _ = lr._verify_and_record(run_dir, venv_python, run_dir / "hf", model)
        assert json.loads((run_dir / ".slm-managed").read_text())["verified"] is True
        assert ok is True
