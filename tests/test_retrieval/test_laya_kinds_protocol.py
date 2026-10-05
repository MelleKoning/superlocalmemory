# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The real Laya worker's ``kinds`` and ``ask`` commands, against a fake ``laya_mlx``.

No weights, no network: a stub library on PYTHONPATH answers like laya-mlx
does (same answer shapes) and, like the real one, prints chatter to stdout —
so the worker's stdout discipline is tested against the real failure mode.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests._portable import child_env_base

WORKER = (Path(__file__).resolve().parents[2]
          / "src" / "superlocalmemory" / "core" / "laya_worker.py")

_STUB = r'''
import json, os

def _log(value):
    path = os.environ.get("LAYA_STUB_LOG")
    if path:
        with open(path, "a") as fh:
            fh.write(json.dumps(value) + "\n")

class _Agent:
    def predict(self, state, questions):
        print("calibration warning written to stdout")
        _log({"state": state, "questions": questions})
        text = " ".join(str(v) for v in state.values())
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "choice":
                labels = list(q["criteria"])
                pick = next((l for l in labels if l in text), labels[0])
                p = {l: (0.6 if l == pick else 0.4 / (len(labels) - 1)) for l in labels}
                answers[qid] = {"type": "choice", "choice": pick, "probabilities": p,
                                "confidence": 0.31, "action": {"act_probability": 0.5}}
            else:
                v = 0.9 if "wrong" in text else 0.1
                answers[qid] = {"type": "noul", "noul": v, "confidence": max(v, 1 - v)}
        return {"model": "laya-rl-agent", "answers": answers, "usage": {}}

def load(model, **kwargs):
    print("Fetching 6 files: 100%")
    return _Agent()
'''

CRITERIA = {"semantic": "a lasting fact", "episodic": "something that happened",
            "procedure": "steps for doing something"}


@pytest.fixture()
def worker(tmp_path):
    (tmp_path / "laya_mlx.py").write_text(_STUB, encoding="utf-8")
    log = tmp_path / "seen.jsonl"
    proc = subprocess.Popen(
        [sys.executable, str(WORKER)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, env={**child_env_base(tmp_path),  # Windows needs a few
                                   "PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin",
                                   "LAYA_STUB_LOG": str(log)},
    )

    def ask(request: dict) -> dict:
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
        return json.loads(proc.stdout.readline())  # raises if chatter reached stdout

    def seen() -> list[dict]:
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

    yield ask, seen
    if proc.poll() is None:
        proc.kill()
        proc.wait(5)


def _kinds(documents, *, criteria=None, instructions="What kind is this?", verify=None):
    req = {"cmd": "kinds", "documents": documents, "instructions": instructions,
           "criteria": CRITERIA if criteria is None else criteria}
    if verify is not None:
        req["verify"] = verify
    return req


def _loaded(ask) -> None:
    assert ask({"cmd": "load", "model": "aac6fef/laya-mlx"})["ok"] is True


def test_one_answer_per_document(worker) -> None:
    ask, seen = worker
    _loaded(ask)
    resp = ask(_kinds(["the procedure is: run x", "this was wrong earlier", "a fact"],
                      verify={"indices": [1], "instructions": "Says an earlier claim was wrong."}))
    assert resp["ok"] is True
    answers = resp["answers"]
    assert len(answers) == 3
    assert [a["choice"] for a in answers] == ["procedure", "semantic", "semantic"]
    assert [a["verify"] for a in answers] == [None, 0.9, None]
    for a in answers:
        assert set(a) == {"choice", "probabilities", "confidence", "verify"}
        assert set(a["probabilities"]) == set(CRITERIA)
    # Each document is asked about on its own, under the field name the
    # question was measured with.
    asked = [s for s in seen() if "warm" not in json.dumps(s["state"])]
    assert [s["state"] for s in asked] == [{"memory": "the procedure is: run x"},
                                           {"memory": "this was wrong earlier"},
                                           {"memory": "a fact"}]
    assert "verify" in asked[1]["questions"] and "verify" not in asked[0]["questions"]


def test_documents_are_cut_not_refused(worker) -> None:
    ask, seen = worker
    _loaded(ask)
    assert ask(_kinds(["x" * 5000]))["ok"] is True
    assert len(seen()[-1]["state"]["memory"]) == 1800


@pytest.mark.parametrize("bad", [
    _kinds(["d"], criteria={f"l{i}": "x" for i in range(11)}),     # > 10 labels
    _kinds(["d"], criteria={"only": "one"}),                        # < 2 labels
    _kinds(["d"], criteria={"x" * 33: "a", "b": "b"}),              # label too long
    _kinds(["d"], criteria={"a": "x" * 121, "b": "b"}),             # description too long
    _kinds(["d"], instructions="x" * 301),                          # instructions too long
    _kinds(["d"], instructions=""),
    _kinds(["d"] * 17),                                             # > 16 documents
    _kinds([]),
    _kinds(["d", 3]),                                               # not text
    _kinds(["d"], verify={"indices": [1], "instructions": "v"}),    # index out of range
    _kinds(["d"], verify={"indices": [True], "instructions": "v"}),
    _kinds(["d"], verify={"indices": [0], "instructions": "x" * 301}),
    _kinds(["d"], verify=["not", "an", "object"]),
    {"cmd": "kinds", "documents": "d", "instructions": "i", "criteria": CRITERIA},
])
def test_rejects_more_than_10_labels_or_long_text(worker, bad) -> None:
    ask, seen = worker
    _loaded(ask)
    before = len(seen())
    assert ask(bad) == {"ok": False, "error": "invalid kinds request"}
    assert len(seen()) == before, "an invalid request never reaches the model"


def test_only_json_on_stdout(worker) -> None:
    ask, _ = worker
    assert ask(_kinds(["d"])) == {"ok": False, "error": "not loaded"}
    _loaded(ask)
    for _ in range(3):
        assert ask({**_kinds(["a fact", "b"]), "id": 7})["id"] == 7
    assert ask({"cmd": "ping"})["loaded"] is True


def test_generic_ask_carries_a_two_memory_question(worker) -> None:
    """The protocol is not tied to the kinds question: a pair question (the
    shape a later 'does NEW make OLD out of date' recipe needs) works too."""
    ask, seen = worker
    _loaded(ask)
    item = {"state": {"new": "the old limit was wrong", "old": "limit is 10"},
            "questions": {"replaces": {"type": "noul",
                                       "instructions": "NEW makes OLD no longer current."}}}
    resp = ask({"cmd": "ask", "items": [item, item]})
    assert resp["ok"] is True
    assert resp["answers"] == [{"replaces": {"type": "noul", "noul": 0.9,
                                             "confidence": 0.9}}] * 2
    assert seen()[-1]["state"] == item["state"]


@pytest.mark.parametrize("bad_item", [
    {"state": {}, "questions": {"q": {"type": "noul", "instructions": "x"}}},
    {"state": {"Bad Key": "x"}, "questions": {"q": {"type": "noul", "instructions": "x"}}},
    {"state": {"m": "x"}, "questions": {}},
    {"state": {"m": "x"}, "questions": {"q": {"type": "score", "instructions": "x"}}},
    {"state": {"m": "x"}, "questions": {"q": {"type": "choice", "instructions": "x"}}},
    {"state": {f"k{i}": "x" for i in range(5)},
     "questions": {"q": {"type": "noul", "instructions": "x"}}},
])
def test_generic_ask_validates_every_item(worker, bad_item) -> None:
    ask, _ = worker
    _loaded(ask)
    assert ask({"cmd": "ask", "items": [bad_item]}) == {
        "ok": False, "error": "invalid ask request"}
