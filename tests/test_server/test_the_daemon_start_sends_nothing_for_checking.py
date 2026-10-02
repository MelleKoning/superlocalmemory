# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Starting the daemon sends no memory anywhere and bills nothing.

The daemon fires three recalls at start to warm its caches. Before this, with
the online answer check on, each one made a billed request carrying the
person's top memories — and with reordering on, up to five of them — although
nobody had asked a question. They are the system's own recalls now: never
judged, never sent. The on-device check's model is still warmed at start, but
by loading it, not by asking it about memories.
"""

from __future__ import annotations

import contextlib
import inspect
import json
from types import SimpleNamespace

import httpx
import pytest

from superlocalmemory.core import judge_selection, recall_pipeline
from superlocalmemory.core.judge_keys import JudgeKeyStore
from superlocalmemory.retrieval.jev_judge import JEV_ENDPOINTS, JevSufficiencyJudge
from superlocalmemory.server import recall_warmup
from superlocalmemory.storage.models import AtomicFact, RecallResponse, RetrievalResult

FAKE_KEY = "sk-test-" + "w4R5t6Y7" * 4
MODEL = JEV_ENDPOINTS["typesafe"][1]


@pytest.fixture(autouse=True)
def _no_live_judge(monkeypatch):
    monkeypatch.setattr(judge_selection, "_live", None)


class _Provider:
    def __init__(self) -> None:
        self.bodies: list[dict] = []
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        answers = {k: {"type": "noul", "noul": 0.9} for k in body["state"]["memories"]}
        return httpx.Response(200, json={"model": MODEL, "answers": answers, "usage": {}})


class _Engine:
    """MemoryEngine's recall signature over the real recall pipeline."""

    def __init__(self, judge, config) -> None:
        response = RecallResponse(results=[
            RetrievalResult(fact=AtomicFact(fact_id=f"f{i}", content=f"private memory {i}"),
                            score=0.9, confidence=1.0)
            for i in range(5)
        ])
        self._retrieval_engine = SimpleNamespace(_sufficiency_judge=judge,
                                                 recall=lambda *a, **k: response)
        self._config = config
        self.calls: list[dict] = []

    def recall(self, query, profile_id=None, mode=None, limit=20, agent_id="unknown",
               session_id=None, fast=None, **kwargs):
        self.calls.append({"query": query, "fast": fast, **kwargs})
        return recall_pipeline.run_recall(
            query, "default", limit=limit, fast=fast, config=self._config,
            retrieval_engine=self._retrieval_engine, trust_scorer=None, embedder=None,
            db=SimpleNamespace(db_path=None), llm=None, hooks=None, **kwargs)


class _Runtime:
    def __init__(self, preempted: bool = False) -> None:
        self.preempted = preempted

    @contextlib.contextmanager
    def operation_nowait(self):
        yield None if self.preempted else SimpleNamespace(profile_id="default")


@pytest.fixture
def online_check(tmp_path):
    store = JudgeKeyStore(slm_home=tmp_path)
    store.set_key("typesafe", FAKE_KEY)
    provider = _Provider()
    judge = JevSufficiencyJudge(provider="typesafe", key_store=store,
                                transport=provider.transport, rerank_k=20)
    yield judge, provider
    judge.shutdown()


def test_the_start_up_recalls_send_nothing_to_the_provider(
    online_check, mode_a_config, monkeypatch,
) -> None:
    judge, provider = online_check
    monkeypatch.setattr(recall_pipeline, "apply_ranking", lambda resp, *a, **k: resp)
    engine = _Engine(judge, mode_a_config)
    recall_warmup.run_warmup_recalls(engine, _Runtime(),
                                     warm_spreading_activation=lambda e, r: None)
    assert len(engine.calls) == 3, "all three warm-up recalls still run"
    assert provider.bodies == [], (
        f"daemon start sent {len(provider.bodies)} billed request(s) carrying memories")


def test_a_preempted_start_still_sends_nothing(online_check, mode_a_config, monkeypatch) -> None:
    judge, provider = online_check
    monkeypatch.setattr(recall_pipeline, "apply_ranking", lambda resp, *a, **k: resp)
    engine = _Engine(judge, mode_a_config)
    recall_warmup.run_warmup_recalls(engine, _Runtime(preempted=True),
                                     warm_spreading_activation=lambda e, r: None)
    assert engine.calls == [] and provider.bodies == []


def test_the_on_device_model_is_warmed_by_loading_it_not_by_asking() -> None:
    started: list[bool] = []
    judge = SimpleNamespace(start_warmup=lambda: started.append(True))
    engine = SimpleNamespace(_retrieval_engine=SimpleNamespace(_sufficiency_judge=judge))
    assert recall_warmup.warm_answer_check(engine) is True
    assert started == [True]
    assert recall_warmup.warm_answer_check(SimpleNamespace()) is False


def test_the_daemon_start_uses_these_recalls() -> None:
    """Wiring guard: the daemon's warm-up thread calls this module."""
    from superlocalmemory.server import unified_daemon

    source = inspect.getsource(unified_daemon)
    start = source.index("def _warmup_recall():")
    body = source[start:source.index("def _backfill_vector_store", start)]
    assert "run_warmup_recalls(" in body
    assert "engine.recall(" not in body, "a warm-up recall bypasses the skip"
