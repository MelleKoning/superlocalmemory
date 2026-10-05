# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A choice withdrawn anywhere stops the check everywhere.

* The online check re-reads the stored choice before every request, so a
  process holding its own engine (an MCP server, a CLI) stops sending the
  moment consent is withdrawn — not only the daemon that heard the click. A
  stored choice it cannot read means no consent.
* An on-device check is never started with an interpreter that other accounts
  could change.
* Removing the on-device install can stop every engine holding its check.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import httpx
import pytest

from superlocalmemory.core import answer_check_state, judge_selection, laya_runtime
from superlocalmemory.core.judge_keys import JudgeKeyStore
from superlocalmemory.retrieval import answer_check_status as acs
from superlocalmemory.retrieval import sufficiency as suff
from superlocalmemory.retrieval.jev_judge import JEV_ENDPOINTS, JevSufficiencyJudge
from superlocalmemory.retrieval.jev_rerank import CHOICE_KEY
from superlocalmemory.retrieval.judge_recipe import JudgeDocument

FAKE_KEY = "sk-test-" + "c0N5e6T7" * 4
MODEL = JEV_ENDPOINTS["typesafe"][1]


class _Provider:
    def __init__(self) -> None:
        self.bodies: list[dict] = []
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request):
        body = json.loads(request.content)
        self.bodies.append(body)
        keys = list(body["state"]["memories"])
        answers = {k: {"type": "noul", "noul": 0.9} for k in keys}
        if CHOICE_KEY in body["questions"]:
            share = 1.0 / len(keys)
            answers[CHOICE_KEY] = {"type": "choice", "choice": keys[0],
                                   "probabilities": {k: share for k in keys}}
        return httpx.Response(200, json={"model": MODEL, "answers": answers, "usage": {}})


def _store(state_dir, **values) -> None:
    base = {"sufficiency_judge": "jev", "sufficiency_jev_provider": "typesafe",
            "sufficiency_jev_consent": True, "sufficiency_jev_rerank": True,
            "sufficiency_jev_rerank_consent": True}
    answer_check_state.update(state_dir, lambda current: {**current, **base, **values},
                              seed=lambda: {})


@pytest.fixture
def online(tmp_path):
    keys = JudgeKeyStore(slm_home=tmp_path / "keys")
    keys.set_key("typesafe", FAKE_KEY)
    provider = _Provider()
    state_dir = tmp_path / "data"
    state_dir.mkdir()
    judges = []

    def build(**kw):
        judge = JevSufficiencyJudge(provider="typesafe", key_store=keys,
                                    transport=provider.transport, state_dir=state_dir, **kw)
        judges.append(judge)
        return judge

    yield build, provider, state_dir
    for judge in judges:
        judge.shutdown()


_DOCS = [JudgeDocument(f"memory {i}") for i in range(4)]


class TestTheOnlineCheckReReadsTheChoice:
    def test_consent_withdrawn_elsewhere_stops_the_next_request(self, online) -> None:
        build, provider, state_dir = online
        _store(state_dir)
        judge = build()
        assert judge.assess("q?", _DOCS).status == acs.STATUS_JUDGED
        _store(state_dir, sufficiency_jev_consent=False)   # another process withdrew it
        outcome = judge.assess("q?", _DOCS)
        assert len(provider.bodies) == 1, "memories were sent after consent was withdrawn"
        assert outcome.verdict is None and outcome.status == acs.STATUS_OFF

    def test_switching_to_another_check_stops_it_too(self, online) -> None:
        build, provider, state_dir = online
        _store(state_dir, sufficiency_judge="off")
        assert build().assess("q?", _DOCS).status == acs.STATUS_OFF
        assert provider.bodies == []

    def test_a_damaged_store_means_no_consent(self, online) -> None:
        build, provider, state_dir = online
        answer_check_state.state_path(state_dir).write_text("{not json", encoding="utf-8")
        assert build().assess("q?", _DOCS).verdict is None
        assert provider.bodies == []

    def test_reorder_consent_withdrawn_falls_back_to_the_plain_check(self, online) -> None:
        build, provider, state_dir = online
        _store(state_dir, sufficiency_jev_rerank_consent=False)
        outcome = build(rerank_k=20).rerank_and_judge("q?", _DOCS)
        (body,) = provider.bodies
        assert CHOICE_KEY not in body["questions"], "it reordered without the consent"
        assert outcome.order is None and outcome.verdict is not None

    def test_with_no_store_yet_the_choice_it_was_built_with_stands(self, online) -> None:
        build, provider, _state_dir = online
        assert build().assess("q?", _DOCS).status == acs.STATUS_JUDGED


class TestTheOnDeviceCheckRunsOnlyASafeInterpreter:
    def test_a_refused_interpreter_is_never_started(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(laya_runtime, "check_interpreter",
                            lambda path, strict_location=True: "writable by others")
        started: list = []
        monkeypatch.setattr(suff.subprocess, "Popen",
                            lambda *a, **k: started.append(a) or pytest.fail("started"))
        judge = suff.LayaSufficiencyJudge(python=sys.executable, start=False)
        try:
            judge.start_warmup()
            for _ in range(100):
                if not judge.loading:
                    break
                import time
                time.sleep(0.02)
            assert started == []
            assert "writable by others" in judge._permanent_error
        finally:
            judge.shutdown()

    def test_finding_laya_refuses_an_unsafe_interpreter(self, monkeypatch) -> None:
        from superlocalmemory.core.config import RetrievalConfig

        monkeypatch.setattr(suff, "laya_supported", lambda: True)
        monkeypatch.setattr(judge_selection, "interpreter_has_module", lambda py, m: True)
        monkeypatch.setattr(laya_runtime, "detect",
                            lambda cfg=None: laya_runtime.LayaRuntimeStatus(state="not_installed"))
        checked: list = []

        def refuse(path, strict_location=True):
            checked.append((path, strict_location))
            return "writable by others"

        monkeypatch.setattr(laya_runtime, "check_interpreter", refuse)
        for cfg in (RetrievalConfig(sufficiency_judge="laya",
                                    sufficiency_python="/opt/laya/bin/python"),
                    RetrievalConfig(sufficiency_judge="laya")):
            assert judge_selection.find_laya(cfg) is None
        assert checked == [("/opt/laya/bin/python", False), (sys.executable, False)]


def test_every_holder_of_the_on_device_check_can_be_stopped(monkeypatch) -> None:
    monkeypatch.setattr(judge_selection, "_live", None)

    class _Laya:
        backend = "laya"
        closed = False

        def shutdown(self) -> None:
            self.closed = True

    class _Engine:
        def __init__(self, judge) -> None:
            self._sufficiency_judge = judge

    judge = judge_selection._acquire(("laya", "k"), _Laya)
    first, second = _Engine(judge), _Engine(judge_selection._acquire(("laya", "k"), _Laya))
    judge_selection.register_engine(first)
    judge_selection.register_engine(second)
    assert judge_selection.stop_on_device_check() is True
    assert judge.closed is True
    assert first._sufficiency_judge is None and second._sufficiency_judge is None
    assert judge_selection.stop_on_device_check() is False


def test_stopping_the_on_device_check_leaves_an_online_one_alone(monkeypatch) -> None:
    monkeypatch.setattr(judge_selection, "_live", None)
    online = SimpleNamespace(backend="jev", closed=False, shutdown=lambda: None)
    judge_selection._acquire(None, lambda: online)
    assert judge_selection.stop_on_device_check() is False
    assert judge_selection._live.judge is online
