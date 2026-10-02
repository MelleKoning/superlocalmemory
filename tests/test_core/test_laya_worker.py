# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The Laya worker's stdout is a protocol, and nothing else may write to it.

laya-mlx prints download progress at load and calibration warnings at
inference. A single stray line on stdout would be read as the next response
and desynchronise every request after it. The stub below misbehaves exactly
that way, so the worker's guard is tested against the real failure mode —
without loading a model.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

WORKER = (Path(__file__).resolve().parents[2]
          / "src" / "superlocalmemory" / "core" / "laya_worker.py")

_STUB = r'''
import json, os

def _log(kind, value):
    # Echo what the worker handed the library, so a test can see it.
    path = os.environ.get("LAYA_STUB_LOG")
    if path:
        with open(path, "a") as fh:
            fh.write(json.dumps({"kind": kind, "value": value}) + "\n")

class _Agent:
    def predict(self, state, questions):
        print("calibration warning written to stdout")
        memory = state["memory"]
        if memory != "warm-up":
            _log("question", questions["sufficient"]["instructions"])
        return {"answers": {"sufficient": {"type": "noul",
                "noul": 0.8 if "answer" in memory else 0.2,
                "seen_chars": len(memory)}}}

def load(model, **kwargs):
    print("Fetching 6 files: 100%")
    _log("load", model)
    if model == "missing":
        raise FileNotFoundError("weights not cached")
    return _Agent()
'''


@pytest.fixture()
def worker(tmp_path):
    (tmp_path / "laya_mlx.py").write_text(_STUB)
    proc = subprocess.Popen(
        [sys.executable, str(WORKER)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, env={"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin",
                                   "LAYA_STUB_LOG": str(tmp_path / "seen.jsonl"),
                                   "HOME": str(tmp_path)},
    )

    def ask(request: dict) -> dict:
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        return json.loads(line)  # raises if anything but protocol reached stdout

    yield proc, ask
    if proc.poll() is None:
        proc.kill()
        proc.wait(5)


def test_library_chatter_never_reaches_the_protocol(worker) -> None:
    proc, ask = worker
    assert ask({"cmd": "ping"}) == {"ok": True, "backend": "laya-mlx", "model": "",
                                     "loaded": False}
    assert ask({"cmd": "load", "model": "aac6fef/laya-mlx"})["ok"] is True
    resp = ask({"cmd": "judge", "query": "q", "documents": ["the answer", "noise"]})
    assert resp == {"ok": True, "probabilities": [0.8, 0.2]}
    assert ask({"cmd": "ping"})["loaded"] is True


def test_it_refuses_to_judge_before_it_is_loaded(worker) -> None:
    _, ask = worker
    assert ask({"cmd": "judge", "query": "q", "documents": ["x"]}) == {
        "ok": False, "error": "not loaded"}


def test_a_failed_load_is_reported_not_raised(worker) -> None:
    proc, ask = worker
    resp = ask({"cmd": "load", "model": "missing"})
    assert resp["ok"] is False
    assert "weights not cached" in resp["error"]
    assert proc.poll() is None, "a failed load must not kill the worker"


@pytest.mark.parametrize("request_", [
    {"cmd": "judge", "query": 5, "documents": ["x"]},
    {"cmd": "judge", "query": "q", "documents": "not a list"},
    {"cmd": "unknown"},
])
def test_bad_requests_get_an_answer_not_a_crash(worker, request_) -> None:
    proc, ask = worker
    ask({"cmd": "load", "model": "aac6fef/laya-mlx"})
    assert ask(request_)["ok"] is False
    assert proc.poll() is None


def test_invalid_json_is_answered_and_the_worker_keeps_going(worker) -> None:
    proc, ask = worker
    proc.stdin.write("this is not json\n")
    proc.stdin.flush()
    assert json.loads(proc.stdout.readline()) == {"ok": False, "error": "invalid JSON"}
    assert ask({"cmd": "ping"})["ok"] is True


def test_long_memories_are_cut_not_refused(worker, tmp_path) -> None:
    """The English checkpoint reads 512 tokens; a 6,000-character checkpoint
    memory must still be judged, on its first MAX_DOCUMENT_CHARS."""
    _, ask = worker
    ask({"cmd": "load", "model": "aac6fef/laya-mlx"})
    resp = ask({"cmd": "judge", "query": "q", "documents": ["answer " + "x" * 6000]})
    assert resp["ok"] is True and len(resp["probabilities"]) == 1


def test_quit_exits_cleanly(worker) -> None:
    proc, ask = worker
    proc.stdin.write('{"cmd": "quit"}\n')
    proc.stdin.flush()
    assert proc.wait(5) == 0


# ---------------------------------------------------------------------------
# 4.1.18: the recipe's question, and a local model folder
# ---------------------------------------------------------------------------


def _seen(tmp_path: Path, kind: str) -> list:
    log = tmp_path / "seen.jsonl"
    if not log.exists():
        return []
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    return [r["value"] for r in rows if r["kind"] == kind]


def test_the_question_in_the_request_is_the_one_asked(worker, tmp_path) -> None:
    _, ask = worker
    ask({"cmd": "load", "model": "aac6fef/laya-mlx"})
    resp = ask({"cmd": "judge", "query": "q", "documents": ["the answer"],
                "question": "Does this memory say where Alice lives?"})
    assert resp == {"ok": True, "probabilities": [0.8]}
    assert _seen(tmp_path, "question") == ["Does this memory say where Alice lives?"]


def test_without_a_question_it_asks_the_measured_one(worker, tmp_path) -> None:
    """An older daemon sends no question; the worker must still ask the
    wording the threshold was measured on — the same text as the recipe."""
    from superlocalmemory.retrieval.judge_recipe import RECIPE_V1

    _, ask = worker
    ask({"cmd": "load", "model": "aac6fef/laya-mlx"})
    assert ask({"cmd": "judge", "query": "q", "documents": ["x"]})["ok"] is True
    assert _seen(tmp_path, "question") == [RECIPE_V1.question]


@pytest.mark.parametrize("question", [5, "", "   ", "x" * 1001, None, ["a list"]],
                         ids=["int", "empty", "blank", "too-long", "null", "list"])
def test_a_malformed_question_is_refused_not_replaced(worker, tmp_path, question) -> None:
    """Answering under a different wording than the judge asked for would label
    a verdict with a recipe it did not come from."""
    proc, ask = worker
    ask({"cmd": "load", "model": "aac6fef/laya-mlx"})
    resp = ask({"cmd": "judge", "query": "q", "documents": ["x"], "question": question})
    assert resp == {"ok": False, "error": "invalid question"}
    assert _seen(tmp_path, "question") == []
    assert proc.poll() is None


def test_a_question_at_the_limit_is_accepted(worker) -> None:
    _, ask = worker
    ask({"cmd": "load", "model": "aac6fef/laya-mlx"})
    assert ask({"cmd": "judge", "query": "q", "documents": ["x"],
                "question": "y" * 1000})["ok"] is True


def test_a_local_model_folder_is_loaded_by_its_absolute_path(worker, tmp_path) -> None:
    folder = tmp_path / "models--aac6fef--laya-mlx" / "snapshots" / "20aed815fc6a"
    folder.mkdir(parents=True)
    _, ask = worker
    assert ask({"cmd": "load", "model": str(folder)})["ok"] is True
    # HOME is tmp_path in this worker's environment.
    assert ask({"cmd": "load",
                "model": "~/models--aac6fef--laya-mlx/snapshots/20aed815fc6a"})["ok"] is True
    assert _seen(tmp_path, "load") == [str(folder), str(folder)]


def test_a_missing_model_folder_is_refused_before_the_library_sees_it(worker, tmp_path) -> None:
    """Handed to the library, a missing folder would be read as a repo id and
    fail offline with an error about the network, not about the folder."""
    proc, ask = worker
    resp = ask({"cmd": "load", "model": str(tmp_path / "no-such-folder")})
    assert resp["ok"] is False
    assert "model folder not found" in resp["error"]
    assert str(tmp_path) not in resp["error"], "the error must not carry the path"
    assert _seen(tmp_path, "load") == []
    assert proc.poll() is None


def test_a_repo_id_is_handed_over_unchanged(worker, tmp_path) -> None:
    _, ask = worker
    assert ask({"cmd": "load", "model": "aac6fef/laya-mlx"})["ok"] is True
    assert _seen(tmp_path, "load") == ["aac6fef/laya-mlx"]


def test_the_worker_stays_self_contained() -> None:
    """It runs under Laya's own interpreter, where SLM is not installed."""
    import ast

    tree = ast.parse(WORKER.read_text())
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {node.module or "" for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom)}
    assert not any(name.startswith("superlocalmemory") for name in imported)


# ---------------------------------------------------------------------------
# memory: the freed-buffer cache is capped, not left at the memory limit
# ---------------------------------------------------------------------------

def _import_worker():
    import importlib.util

    spec = importlib.util.spec_from_file_location("laya_worker_under_test", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_mlx(monkeypatch, *, with_cache_limit=True):
    import types

    calls: dict[str, int] = {}
    core = types.ModuleType("mlx.core")
    core.set_memory_limit = lambda n: calls.__setitem__("memory", n) or 0
    if with_cache_limit:
        core.set_cache_limit = lambda n: calls.__setitem__("cache", n) or 0
    package = types.ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return calls


def test_the_freed_buffer_cache_is_capped_well_below_the_memory_limit(monkeypatch) -> None:
    """MLX keeps freed buffers up to the memory limit unless told otherwise:
    measured, that held a worker at 2.2 GB where 1.1 GB gave identical answers
    at the same speed."""
    worker = _import_worker()
    calls = _fake_mlx(monkeypatch)
    worker._limit_memory(2048)
    assert calls["memory"] == 2048 * 1024 * 1024
    assert calls["cache"] == worker.CACHE_LIMIT_MB * 1024 * 1024
    assert worker.CACHE_LIMIT_MB <= 256


def test_an_mlx_without_a_cache_limit_still_loads(monkeypatch) -> None:
    worker = _import_worker()
    calls = _fake_mlx(monkeypatch, with_cache_limit=False)
    worker._limit_memory(2048)
    assert calls == {"memory": 2048 * 1024 * 1024}
