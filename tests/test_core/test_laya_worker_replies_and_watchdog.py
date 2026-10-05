# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The Laya worker names each reply, says whether a failed load can ever
succeed, and leaves when the process that started it is gone.

* A reply carries the id of the request it answers. A recall that gave up on a
  slow answer must never read that late answer as the reply to its own
  question.
* A load that can never succeed (the library or the weights are missing) says
  so, so the judge stops trying instead of reloading every few seconds forever.
* The watchdog compares the parent's id with the one it started under. A
  liveness probe on the old id cannot tell a dead parent from a new process
  that happens to reuse the number, and an orphaned worker keeps ~1 GB resident.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

WORKER = (Path(__file__).resolve().parents[2]
          / "src" / "superlocalmemory" / "core" / "laya_worker.py")

_STUB = r'''
class _Agent:
    def predict(self, state, questions):
        memory = state["memory"]
        return {"answers": {"sufficient": {"type": "noul",
                "noul": 0.8 if "answer" in memory else 0.2}}}

def load(model, **kwargs):
    if model == "missing":
        raise FileNotFoundError("weights not cached")
    if model == "oom":
        raise RuntimeError("[metal] out of memory")
    if model == "offline":
        class LocalEntryNotFoundError(FileNotFoundError):
            pass
        raise LocalEntryNotFoundError("not in the cache and offline")
    return _Agent()
'''


def _spawn(tmp_path: Path, *, with_library: bool = True) -> subprocess.Popen:
    if with_library:
        (tmp_path / "laya_mlx.py").write_text(_STUB, encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, str(WORKER)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
        env={"PYTHONPATH": str(tmp_path) if with_library else "",
             "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )


def _ask(proc: subprocess.Popen, request: dict) -> dict:
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


@pytest.fixture()
def worker(tmp_path):
    proc = _spawn(tmp_path)
    yield proc
    if proc.poll() is None:
        proc.kill()
        proc.wait(5)


class TestEveryReplyNamesItsRequest:
    def test_ping_load_and_judge_echo_the_id(self, worker) -> None:
        assert _ask(worker, {"cmd": "ping", "id": 7})["id"] == 7
        assert _ask(worker, {"cmd": "load", "model": "aac6fef/laya-mlx", "id": 8})["id"] == 8
        reply = _ask(worker, {"cmd": "judge", "query": "q", "documents": ["the answer"],
                              "id": 9})
        assert reply == {"ok": True, "probabilities": [0.8], "id": 9}

    def test_errors_echo_the_id_too(self, worker) -> None:
        assert _ask(worker, {"cmd": "judge", "query": "q", "documents": ["x"],
                             "id": 3}) == {"ok": False, "error": "not loaded", "id": 3}
        assert _ask(worker, {"cmd": "nope", "id": 4})["id"] == 4

    def test_a_request_without_an_id_gets_the_old_reply(self, worker) -> None:
        """The setup check (``laya_runtime.verify``) sends no id."""
        assert _ask(worker, {"cmd": "ping"}) == {"ok": True, "backend": "laya-mlx",
                                                  "model": "", "loaded": False}


class TestAFailedLoadSaysWhetherItCanEverSucceed:
    @pytest.mark.parametrize("model", ["missing", "offline"])
    def test_missing_weights_are_permanent(self, worker, model) -> None:
        reply = _ask(worker, {"cmd": "load", "model": model, "id": 1})
        assert reply["ok"] is False and reply["error_kind"] == "permanent"

    def test_a_missing_model_folder_is_permanent(self, worker, tmp_path) -> None:
        reply = _ask(worker, {"cmd": "load", "model": str(tmp_path / "nowhere"), "id": 1})
        assert reply["ok"] is False and reply["error_kind"] == "permanent"

    def test_running_out_of_memory_is_transient(self, worker) -> None:
        reply = _ask(worker, {"cmd": "load", "model": "oom", "id": 1})
        assert reply["ok"] is False and reply["error_kind"] == "transient"

    def test_a_missing_library_is_permanent(self, tmp_path) -> None:
        proc = _spawn(tmp_path, with_library=False)
        try:
            reply = _ask(proc, {"cmd": "load", "model": "aac6fef/laya-mlx", "id": 1})
        finally:
            proc.kill()
            proc.wait(5)
        assert reply["ok"] is False and reply["error_kind"] == "permanent"


def _load_worker_module():
    spec = importlib.util.spec_from_file_location("laya_worker_under_test", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTheWatchdog:
    def test_it_leaves_when_the_parent_changes_even_if_the_old_id_is_alive(
        self, monkeypatch,
    ) -> None:
        """PID reuse: the old parent id answers a liveness probe (a new process
        owns it), but the worker has been re-parented. It must leave."""
        worker = _load_worker_module()
        parents = iter([4242] + [1] * 100)
        monkeypatch.setattr(worker.os, "getppid", lambda: next(parents))
        monkeypatch.setattr(worker.os, "kill", lambda pid, sig: None)  # "alive"
        exits: list[int] = []

        def fake_exit(code: int) -> None:
            exits.append(code)
            raise SystemExit(code)

        monkeypatch.setattr(worker.os, "_exit", fake_exit)
        monkeypatch.setattr(worker, "WATCH_INTERVAL_S", 0.01)
        with pytest.raises(SystemExit):
            worker._watch_parent(start_thread=False)
        assert exits == [0]

    def test_it_stays_while_the_parent_is_the_same(self, monkeypatch) -> None:
        worker = _load_worker_module()
        monkeypatch.setattr(worker.os, "getppid", lambda: 4242)
        checks = {"n": 0}

        def stop_after_a_few() -> bool:
            checks["n"] += 1
            return checks["n"] > 5

        monkeypatch.setattr(worker, "WATCH_INTERVAL_S", 0.001)
        monkeypatch.setattr(worker.os, "_exit", lambda code: pytest.fail("it left"))
        worker._watch_parent(start_thread=False, stop=stop_after_a_few)
        assert checks["n"] > 5
