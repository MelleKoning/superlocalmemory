# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""One Save applies the whole Answer check form, live, and one option runs.

Varun, on 4.1.19: the key saved but Jev did not turn on until SLM was
restarted; and "if someone has connected to Jev, then only Jev should work.
Laya should not work." These tests drive the real routes into the real judge
selection and engine wiring — only the two judges' constructors are fakes
(the on-device one would load a model; the online one gets a local transport
that records every request), so they prove:

* Save writes the choice, provider, key, notice and reordering, and the very
  next recall is checked by the newly chosen option — no restart;
* while Jev is chosen the on-device check is never built, warmed or asked;
* while the on-device check is chosen nothing reaches the Jev endpoint —
  reordering included — and the reordering choice comes back with Jev;
* a form that can't be applied yet is refused whole, saying what to finish.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.core import engine_wiring, judge_selection, laya_runtime
from superlocalmemory.core.judge_keys import JudgeKeyStore
from superlocalmemory.retrieval import sufficiency
from superlocalmemory.retrieval.jev_judge import JEV_ENDPOINTS, JevSufficiencyJudge
from superlocalmemory.retrieval.jev_rerank import CHOICE_KEY
from superlocalmemory.retrieval.judge_recipe import JudgeDocument
from superlocalmemory.server.routes import answer_check, answer_check_actions

FAKE_KEY = "sk-test-" + "n0tRe4L5" * 4
MODEL = JEV_ENDPOINTS["typesafe"][1]
DOCS = [JudgeDocument(f"memory {i}") for i in range(4)]


class _JevEndpoint:
    """A local stand-in for the provider; records every request it receives."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request):
        body = json.loads(request.content)
        self.bodies.append(body)
        keys = list(body["state"]["memories"])
        answers = {k: {"type": "noul", "noul": 0.9} for k in keys}
        if CHOICE_KEY in body["questions"]:
            answers[CHOICE_KEY] = {"type": "choice", "choice": keys[0],
                                   "probabilities": {k: 1.0 / len(keys) for k in keys}}
        return httpx.Response(200, json={"model": MODEL, "answers": answers, "usage": {}})


class _FakeLaya:
    backend = "laya"

    def __init__(self, calls: list) -> None:
        self.calls = calls
        calls.append("built")

    def start_warmup(self) -> None:
        self.calls.append("warmup")

    def assess(self, query, documents):
        self.calls.append("judged")

    def shutdown(self) -> None:
        self.calls.append("stopped")


class _Engine:
    """The published engine, as far as the answer check sees it."""

    def __init__(self) -> None:
        self._retrieval_engine = type("RE", (), {"_sufficiency_judge": None})()


@pytest.fixture
def world(monkeypatch, tmp_path):
    monkeypatch.setattr(answer_check, "_require_manage", lambda request: None)
    monkeypatch.setattr(answer_check, "_require_credential", lambda request: None)
    monkeypatch.setattr(sufficiency, "laya_supported", lambda: True)
    # The real builder (conftest replaces it with "no judge" for every test).
    monkeypatch.setattr(engine_wiring, "init_sufficiency_judge",
                        judge_selection.build_sufficiency_judge)
    laya_calls: list = []
    endpoint = _JevEndpoint()
    monkeypatch.setattr(judge_selection, "_construct_laya", lambda plan: _FakeLaya(laya_calls))

    def jev(provider, store, timeout_s, rerank_k):
        return JevSufficiencyJudge(provider=provider, key_store=store, timeout_s=timeout_s,
                                   rerank_k=rerank_k, transport=endpoint.transport,
                                   state_dir=tmp_path)

    monkeypatch.setattr(judge_selection, "_construct_jev", jev)
    ready = laya_runtime.LayaRuntimeStatus(
        state=laya_runtime.STATE_READY, python="/opt/laya/bin/python",
        model_path="/opt/laya/model")
    monkeypatch.setattr(laya_runtime, "detect", lambda cfg=None: ready)
    monkeypatch.setattr(judge_selection, "find_laya", lambda cfg: judge_selection._LayaPlan(
        "/opt/laya/bin/python", "/opt/laya/model", "", 5.0))
    (tmp_path / "config.json").write_text(json.dumps({"retrieval": {
        "sufficiency_judge": "auto", "sufficiency_jev_provider": "typesafe",
        "sufficiency_jev_consent": False}}), encoding="utf-8")

    app = FastAPI()
    app.state.engine = _Engine()
    app.include_router(answer_check.router)
    app.include_router(answer_check_actions.router)
    yield TestClient(app), app.state.engine._retrieval_engine, laya_calls, endpoint
    judge_selection.release_sufficiency_judge(app.state.engine._retrieval_engine, final=True)


def _save(client, **form):
    body = {"provider": "typesafe", "consent": False, "rerank": False, **form}
    return client.post("/api/v3/answer-check/save", json=body)


def _recall(engine):
    """What a recall does with the answer check: ask the judge on the engine."""
    judge = engine._sufficiency_judge
    if judge is None:
        return None
    if getattr(judge, "rerank_k", 0):
        return judge.rerank_and_judge("When is the launch?", DOCS)
    return judge.assess("When is the launch?", DOCS)


def test_saving_jev_turns_it_on_now_and_the_next_recall_uses_it(world):
    client, engine, laya_calls, endpoint = world
    response = _save(client, mode="jev", key=FAKE_KEY, consent=True)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["active"] == "jev" and body["jev"]["has_key"] is True
    assert body["message"].startswith("Saved.")
    assert FAKE_KEY not in response.text
    _recall(engine)
    assert len(endpoint.bodies) == 1           # no restart: the next recall went to Jev
    assert JudgeKeyStore().load("typesafe") == FAKE_KEY
    assert laya_calls == []                    # Jev chosen: Laya never built, warmed or asked


def test_while_jev_is_chosen_laya_is_never_invoked(world):
    client, engine, laya_calls, _ = world
    assert _save(client, mode="jev", key=FAKE_KEY, consent=True, rerank=True).status_code == 200
    for _ in range(3):
        _recall(engine)
    assert _save(client, mode="jev", consent=True, rerank=True).status_code == 200  # re-save
    _recall(engine)
    assert laya_calls == []


def test_while_laya_is_chosen_nothing_reaches_jev_and_reordering_comes_back(world):
    client, engine, laya_calls, endpoint = world
    assert _save(client, mode="jev", key=FAKE_KEY, consent=True, rerank=True).status_code == 200
    _recall(engine)
    (first,) = endpoint.bodies
    assert CHOICE_KEY in first["questions"]     # reordering ran with Jev
    old_jev = engine._sufficiency_judge

    response = _save(client, mode="laya", consent=True, rerank=True)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["active"] == "laya"
    assert body["jev"]["rerank"] == {**body["jev"]["rerank"], "enabled": True, "active": False}
    assert "Reordering with Jev is off while it isn't chosen" in body["message"]
    for _ in range(3):
        _recall(engine)
    old_jev.assess("q?", DOCS)                  # even a judge someone still held
    assert len(endpoint.bodies) == 1, "memory text reached Jev while Laya was chosen"
    assert "judged" in laya_calls

    assert _save(client, mode="jev", consent=True, rerank=True).status_code == 200
    _recall(engine)
    assert len(endpoint.bodies) == 2
    assert CHOICE_KEY in endpoint.bodies[-1]["questions"]   # the remembered choice is back


def test_a_form_that_cannot_apply_yet_is_refused_whole(world, monkeypatch):
    client, engine, _, _ = world
    response = _save(client, mode="jev", key=FAKE_KEY, consent=False)
    assert response.status_code == 400
    assert "Tick the notice" in response.json()["error"]
    assert JudgeKeyStore().has_key("typesafe") is False      # nothing half-saved
    response = _save(client, mode="jev")
    assert "Paste your Jev key" in response.json()["error"]
    not_ready = laya_runtime.LayaRuntimeStatus(
        state=laya_runtime.STATE_FAILED, action=laya_runtime.ACTION_CHECK)
    monkeypatch.setattr(laya_runtime, "detect", lambda cfg=None: not_ready)
    response = _save(client, mode="laya")
    assert response.status_code == 400
    assert "isn't ready yet" in response.json()["error"]
    assert engine._sufficiency_judge is None


def test_withdrawing_consent_on_save_stops_jev_and_forgets_reordering(world):
    client, engine, _, endpoint = world
    assert _save(client, mode="jev", key=FAKE_KEY, consent=True, rerank=True).status_code == 200
    response = _save(client, mode="off", consent=False, rerank=True)
    assert response.status_code == 200
    assert response.json()["jev"]["rerank"]["enabled"] is False
    assert engine._sufficiency_judge is None
    _recall(engine)
    assert endpoint.bodies == []


def test_a_save_with_nothing_changed_still_repairs_a_check_that_never_switched(world):
    """4.1.19: the key was saved but the running check stayed off."""
    client, engine, _, endpoint = world
    JudgeKeyStore().set_key("typesafe", FAKE_KEY)
    from superlocalmemory.server.routes import answer_check_support as support

    support.save(lambda v: {**v, "sufficiency_judge": "jev", "sufficiency_jev_consent": True})
    assert engine._sufficiency_judge is None       # saved, but nothing switched it on
    assert _save(client, consent=True).status_code == 200   # mode omitted: keep the choice
    assert judge_selection.backend_of(engine._sufficiency_judge) == "jev"
    _recall(engine)
    assert len(endpoint.bodies) == 1


def test_saving_only_a_key_while_jev_is_chosen_applies_it_live(world):
    client, engine, _, endpoint = world
    from superlocalmemory.server.routes import answer_check_support as support

    support.save(lambda v: {**v, "sufficiency_judge": "jev", "sufficiency_jev_consent": True})
    response = client.post("/api/v3/answer-check/jev/key",
                           json={"provider": "typesafe", "key": FAKE_KEY})
    assert response.status_code == 200
    assert judge_selection.backend_of(engine._sufficiency_judge) == "jev"
    _recall(engine)
    assert len(endpoint.bodies) == 1


def test_auto_kept_when_the_choice_is_not_touched(world):
    """Saving a key must not turn the never-chosen start state into "off"."""
    client, _, _, _ = world
    response = _save(client, key=FAKE_KEY)
    assert response.status_code == 200, response.text
    assert response.json()["mode"] == "auto"
