# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A loading embedding model must not hold a whole recall (4.1.20 WP10).

On a fresh daemon the model took 15-33 s to load. The query embedding was a
blocking call made before any channel ran, so keyword search — which needs no
vector — waited too, the daemon's 25 s budget fired, and a memory saved
seconds earlier came back as "No confident match".
"""

from __future__ import annotations

import threading
import time

import pytest

from superlocalmemory.retrieval import channel_status as chstat
from superlocalmemory.retrieval import engine as engine_mod
from superlocalmemory.retrieval.query_embedding import QueryEmbedder


class _ColdEmbedder:
    """Blocks like a model that is still loading, until released."""

    def __init__(self, *, report_warm: bool | None = False, hold: float = 5.0) -> None:
        self.release = threading.Event()
        self.calls = 0
        self._hold = hold
        self._report_warm = report_warm

    @property
    def is_warm(self):
        if self._report_warm is None:
            raise AttributeError("is_warm")
        return self._report_warm if not self.release.is_set() else True

    def embed(self, text):
        self.calls += 1
        self.release.wait(self._hold)
        return [0.1] * 8


class _FastEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text):
        self.calls += 1
        return [0.2] * 8


# -- QueryEmbedder -----------------------------------------------------------

def test_a_cold_model_is_reported_as_warming_not_waited_for() -> None:
    cold = _ColdEmbedder(report_warm=False)
    qe = QueryEmbedder(lambda: cold)
    t0 = time.monotonic()
    vector, status = qe.embed("q", 0.2)
    assert time.monotonic() - t0 < 1.5
    assert vector is None
    assert status == chstat.WARMING
    cold.release.set()
    qe.close()


def test_a_slow_but_loaded_model_is_reported_as_timeout() -> None:
    slow = _ColdEmbedder(report_warm=True)
    qe = QueryEmbedder(lambda: slow)
    _, status = qe.embed("q", 0.2)
    assert status == chstat.TIMEOUT
    slow.release.set()
    qe.close()


def test_the_abandoned_embed_finishes_and_the_next_recall_gets_it() -> None:
    cold = _ColdEmbedder(report_warm=False)
    qe = QueryEmbedder(lambda: cold)
    assert qe.embed("q", 0.1) == (None, chstat.WARMING)
    cold.release.set()
    deadline = time.monotonic() + 3
    while "q" not in qe.cache and time.monotonic() < deadline:
        time.sleep(0.02)
    assert qe.embed("q", 0.1) == ([0.1] * 8, None)
    assert cold.calls == 1  # served from the cache, not embedded twice
    qe.close()


def test_concurrent_recalls_of_one_question_share_one_embed() -> None:
    cold = _ColdEmbedder(report_warm=False, hold=0.5)
    qe = QueryEmbedder(lambda: cold)
    out: list = []
    threads = [threading.Thread(target=lambda: out.append(qe.embed("same", 2.0)))
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert cold.calls == 1
    assert all(vec == [0.1] * 8 and st is None for vec, st in out)
    qe.close()


def test_a_warm_model_answers_with_no_status() -> None:
    fast = _FastEmbedder()
    qe = QueryEmbedder(lambda: fast)
    assert qe.embed("q", 5.0) == ([0.2] * 8, None)
    qe.close()


def test_an_embedder_error_still_raises() -> None:
    class _Broken:
        def embed(self, text):
            raise RuntimeError("provider down")

    qe = QueryEmbedder(lambda: _Broken())
    with pytest.raises(RuntimeError):
        qe.embed("q", 2.0)
    qe.close()


def test_background_work_embeds_inline_exactly_as_before() -> None:
    from superlocalmemory.core.recall_gate import background_work

    seen: list[str] = []

    class _Where:
        def embed(self, text):
            seen.append(threading.current_thread().name)
            return [0.3] * 8

    qe = QueryEmbedder(lambda: _Where())
    with background_work():
        assert qe.embed("q", 0.0) == ([0.3] * 8, None)
    assert seen == [threading.current_thread().name]
    qe.close()


def test_the_current_embedder_is_used_after_a_swap() -> None:
    holder = {"e": _FastEmbedder()}
    qe = QueryEmbedder(lambda: holder["e"])
    holder["e"] = None
    assert qe.embed("q", 1.0) == (None, None)
    qe.close()


# -- RetrievalEngine._run_channels ------------------------------------------

class _Channel:
    def __init__(self, hits) -> None:
        self._hits = hits

    def search(self, *args, **kwargs):
        return list(self._hits)


def _engine(embedder):
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.models import Mode

    config = SLMConfig.for_mode(Mode.A)
    config.retrieval.use_cross_encoder = False
    return engine_mod.RetrievalEngine(
        db=None,
        channels={
            "semantic": _Channel([("s1", 0.9)]),
            "hopfield": _Channel([("h1", 0.8)]),
            "spreading_activation": _Channel([("a1", 0.7)]),
            "bm25": _Channel([("exact", 3.0)]),
        },
        config=config.retrieval,
        embedder=embedder,
    )


def _run(eng):
    from superlocalmemory.retrieval.strategy import QueryStrategy

    status: dict[str, str] = {}
    dropped: set[str] = set()
    out = eng._run_channels(
        "When is the migration window?", "default",
        QueryStrategy(query_type="factual", weights={}),
        dropped_channels=dropped, channel_status=status,
    )
    return out, status, dropped


def test_keyword_search_answers_while_the_model_loads(monkeypatch) -> None:
    monkeypatch.setattr(engine_mod, "CHANNEL_HANG_GUARD_SECONDS", 0.3)
    cold = _ColdEmbedder(report_warm=False, hold=5.0)
    eng = _engine(cold)
    t0 = time.monotonic()
    out, status, dropped = _run(eng)
    elapsed = time.monotonic() - t0
    cold.release.set()
    eng.close()

    assert elapsed < 2.0, f"recall waited {elapsed:.1f}s on a loading model"
    assert out.get("bm25") == [("exact", 3.0)]
    for name in ("semantic", "hopfield", "spreading_activation"):
        assert status[name] == chstat.WARMING
        assert chstat.is_fault(status[name])
    # Incomplete, not "found nothing": the caller is told to ask again.
    assert dropped == {"semantic", "hopfield", "spreading_activation"}
    assert status["bm25"] == chstat.OK


def test_a_warm_model_still_runs_every_channel(monkeypatch) -> None:
    eng = _engine(_FastEmbedder())
    out, status, dropped = _run(eng)
    eng.close()
    assert dropped == set()
    assert set(out) >= {"semantic", "hopfield", "spreading_activation", "bm25"}
    assert all(status[n] == chstat.OK for n in
               ("semantic", "hopfield", "spreading_activation", "bm25"))


def test_a_model_that_answers_inside_the_guard_is_waited_for(monkeypatch) -> None:
    """The guard bounds a hang; it is not a speed cutoff. A reload that takes
    a moment (an idle-unloaded worker) must still feed the vector channels."""
    monkeypatch.setattr(engine_mod, "CHANNEL_HANG_GUARD_SECONDS", 3.0)
    slowish = _ColdEmbedder(report_warm=False, hold=0.4)
    eng = _engine(slowish)
    out, status, dropped = _run(eng)
    eng.close()
    assert dropped == set()
    assert status["semantic"] == chstat.OK
    assert "semantic" in out
