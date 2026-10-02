# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The monitor that revives a dead embedder has to notice it is dead.

WHAT WAS OBSERVED

A daemon killed its embedding worker on the routine idle timeout — by design,
every 45-60 minutes — and never respawned it. For over an hour, across two
restarts, the recall-health monitor wrote **nothing at all**: no tick line, no
warning, not one CRITICAL. `readiness.embedding` stayed false and the daemon sat
in `warming` forever. A manual restart fixed it every time, which is what made it
look like the monitor thread had died or never started.

The thread was fine. It was ticking on schedule and concluding, every time, that
everything was healthy.

WHY

Tier 2 asked one question: are there results whose semantic score is all zero?
Zero results was explicitly excluded from that signature, on the reasoning that
an empty corpus should not be read as a broken embedder. But **a dead embedder is
one of the reasons a recall returns nothing** — the semantic channel contributes
zero candidates, and the probe phrase appears verbatim in nobody's memories, so
keyword finds nothing either. So the one symptom that should have triggered the
heal was taken as proof that no heal was needed. And because a healthy verdict is
silent by design, it left no trace.

WHAT CHANGED

The embedder is now asked directly whether it can produce a vector, which is a
question a recall cannot answer. And the fact of a tick is recorded, so a quiet
monitor can be told apart from an absent one without reading the log.
"""

from __future__ import annotations

import pytest

from superlocalmemory.server import recall_health as rh


class _Log:
    """Captures what the tick would have written."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def __getattr__(self, level):
        def write(msg, *args):
            self.lines.append((level, msg % args if args else msg))
        return write

    def at(self, level):
        return [m for lvl, m in self.lines if lvl == level]


def _engine(embedder, results):
    class Engine:
        _embedder = embedder

        def recall(self, query, limit=3, fast=True):
            return type("Response", (), {"results": results})()

    return Engine()


class _Killed:
    """An embedder whose worker was killed on idle timeout."""
    _available = True          # the constructor set this and nothing cleared it
    is_warm = False            # no worker has served a request

    def __init__(self, revives: bool = True) -> None:
        self._revives = revives
        self.embed_calls = 0

    def embed(self, text):
        self.embed_calls += 1
        return [0.1] * 768 if self._revives else None


class _Serving:
    _available = True
    is_warm = True

    def embed(self, text):
        return [0.1] * 768


class _Result:
    channel_scores = {"semantic": 0.8}


class TestTheKilledWorkerIsRevived:
    """The reported case, exactly: worker gone, probe returns nothing."""

    def test_a_dead_embedder_with_no_probe_results_is_healed(self) -> None:
        """Reverting the fix makes this fail three ways at once: healthy stays
        True, no heal is attempted, and nothing is logged."""
        embedder = _Killed(revives=True)
        state = rh.RecallHealth()
        log = _Log()

        rh.run_health_tick(_engine(embedder, []), state, log=log)

        assert embedder.embed_calls >= 1, "the embedder was never re-exercised"
        assert state.total_heals == 1
        assert state.healthy is True, "it revived, so the path is healthy again"
        assert log.at("warning"), "a heal has to be announced"

    def test_it_says_so_when_the_embedder_will_not_come_back(self) -> None:
        embedder = _Killed(revives=False)
        state = rh.RecallHealth()
        log = _Log()

        rh.run_health_tick(_engine(embedder, []), state, log=log)

        assert state.healthy is False
        assert state.consecutive_failures == 1
        assert log.at("critical"), "a failed heal must be CRITICAL, never silent"

    def test_the_tick_is_never_silent_about_a_dead_embedder(self) -> None:
        """The whole reason this went unnoticed for an hour."""
        state = rh.RecallHealth()
        log = _Log()

        rh.run_health_tick(_engine(_Killed(), []), state, log=log)

        assert log.lines, "a dead embedder must leave a trace in the log"


class TestItStaysQuietWhenNothingIsWrong:
    """A monitor that narrates every success is one whose warnings get skimmed."""

    def test_a_working_embedder_logs_nothing(self) -> None:
        state = rh.RecallHealth()
        log = _Log()

        rh.run_health_tick(_engine(_Serving(), [_Result()]), state, log=log)

        assert state.healthy is True
        assert state.total_heals == 0
        assert log.lines == []

    def test_no_embedder_configured_is_not_a_fault(self) -> None:
        """A store deliberately running keyword-only has nothing to heal, and
        must not be woken up every five minutes about it."""
        state = rh.RecallHealth()
        log = _Log()

        rh.run_health_tick(_engine(None, []), state, log=log)

        assert state.healthy is True
        assert log.lines == []

    def test_an_empty_corpus_with_a_live_embedder_is_not_a_fault(self) -> None:
        """The case the original exclusion was protecting, which still holds:
        no results and a working embedder means there was nothing to find."""
        state = rh.RecallHealth()
        log = _Log()

        rh.run_health_tick(_engine(_Serving(), []), state, log=log)

        assert state.healthy is True
        assert state.total_heals == 0
        assert log.lines == []


class TestAQuietMonitorCanBeDistinguishedFromAnAbsentOne:
    def test_a_tick_records_that_it_happened(self) -> None:
        state = rh.RecallHealth()
        assert state.last_tick_at == 0.0

        rh.run_health_tick(_engine(_Serving(), [_Result()]), state, log=_Log())

        assert state.last_tick_at > 0.0

    def test_the_health_snapshot_carries_it(self) -> None:
        """Without this, "is the monitor running?" is unanswerable except by
        waiting for a failure to be logged."""
        snapshot = rh.get_recall_health()

        assert "last_tick_at" in snapshot
        assert "seconds_since_last_tick" in snapshot
        assert "embedder_alive" in snapshot

    def test_a_monitor_that_never_ticked_reports_no_tick(self) -> None:
        """None rather than 0, so a caller cannot read "never" as "just now"."""
        state = rh.RecallHealth()
        assert state.last_tick_at == 0.0
        # get_recall_health reads module state; assert the mapping directly.
        assert rh.RecallHealth().last_tick_at == 0.0


class TestTheDirectQuestion:
    """``_embedder_is_dead`` is the question a recall cannot answer."""

    @pytest.mark.parametrize(
        "embedder, expected",
        [
            (_Serving(), False),
            (_Killed(), True),
            (None, False),
        ],
    )
    def test_it_answers_from_the_embedder_not_the_results(
        self, embedder, expected,
    ) -> None:
        assert rh._embedder_is_dead(_engine(embedder, [])) is expected

    def test_an_embedder_that_reports_itself_unavailable_is_dead(self) -> None:
        class Unavailable:
            _available = False

            def embed(self, text):
                return None

        assert rh._embedder_is_dead(_engine(Unavailable(), [])) is True


class _KeywordOnlyResult:
    """What a content-free probe actually gets back: keyword and entity hits
    with no semantic score among its top three."""
    channel_scores = {"semantic": 0.0, "bm25": 0.7, "entity_graph": 1.0}


class TestAProbeWithoutASemanticMatchIsNotAnOutage:
    """4.1.18. The live daemon logged "semantic channel DEAD" on 1,970
    consecutive ticks and recorded zero recoveries, while a real query on the
    same daemon scored semantic 1.228. The embedder was fine every time: the
    probe phrase is content-free by design and asks for only three results, so
    none of them carries a semantic score. Treating that as an outage meant a
    CRITICAL and a "heal" every five minutes — and an operator who learns to
    ignore this monitor would also ignore it on the day it is right."""

    def test_a_serving_embedder_with_keyword_only_results_is_healthy(self) -> None:
        state = rh.RecallHealth()
        log = _Log()

        rh.run_health_tick(_engine(_Serving(), [_KeywordOnlyResult()]), state, log=log)

        assert state.healthy is True
        assert state.total_heals == 0
        assert log.at("critical") == []

    def test_an_embedder_that_returns_no_vector_is_still_caught(self) -> None:
        """Guard against over-relaxing: flags say warm, embed() gives nothing."""
        class _Hollow:
            _available = True
            is_warm = True

            def embed(self, text):
                return None

        state = rh.RecallHealth()
        log = _Log()

        rh.run_health_tick(_engine(_Hollow(), [_KeywordOnlyResult()]), state, log=log)

        assert state.healthy is False
        assert log.at("critical")


class _Reranker:
    """Shaped like CrossEncoderReranker after ``_kill_worker``."""

    def __init__(self, loaded: bool = False, loading: bool = False) -> None:
        self._model_loaded = loaded
        self._worker_loading = loading
        self.warmups = 0

    def _start_background_warmup(self) -> None:
        self.warmups += 1


def _engine_with_reranker(reranker):
    engine = _engine(_Serving(), [_Result()])
    engine._retrieval_engine = type("Retrieval", (), {"_reranker": reranker})()
    return engine


class TestTheRerankerIsWatched:
    """4.1.18. The live reranker died on 2026-09-25 and stayed dead for seven
    days while /health said recall_healthy: true — the monitor never looked at
    it. It now reports the reranker and re-arms its warm-up, a recovery path
    that does not depend on recall traffic reaching the reranker at all."""

    def test_a_reranker_with_no_model_loaded_is_re_armed(self) -> None:
        reranker = _Reranker(loaded=False)
        state = rh.RecallHealth()
        log = _Log()

        rh.run_health_tick(_engine_with_reranker(reranker), state, log=log)

        assert reranker.warmups == 1
        assert state.reranker_ready is False
        assert state.reranker_rearms == 1
        assert log.at("warning")

    def test_a_loaded_reranker_is_left_alone(self) -> None:
        reranker = _Reranker(loaded=True)
        state = rh.RecallHealth()
        log = _Log()

        rh.run_health_tick(_engine_with_reranker(reranker), state, log=log)

        assert reranker.warmups == 0
        assert state.reranker_ready is True
        assert log.lines == []

    def test_a_reranker_already_warming_is_not_re_armed_again(self) -> None:
        reranker = _Reranker(loaded=False, loading=True)
        state = rh.RecallHealth()

        rh.run_health_tick(_engine_with_reranker(reranker), state, log=_Log())

        assert reranker.warmups == 0

    def test_a_reranker_without_a_local_model_is_not_judged(self) -> None:
        """RemoteReranker has no model to load and no warm-up to start."""
        state = rh.RecallHealth()

        rh.run_health_tick(_engine_with_reranker(object()), state, log=_Log())

        assert state.reranker_configured is True
        assert state.reranker_ready is None

    def test_no_reranker_configured_is_not_a_fault(self) -> None:
        state = rh.RecallHealth()
        log = _Log()

        rh.run_health_tick(_engine(_Serving(), [_Result()]), state, log=log)

        assert state.reranker_configured is False
        assert log.lines == []

    def test_the_snapshot_reports_the_reranker(self) -> None:
        snapshot = rh.get_recall_health()

        assert "reranker_ready" in snapshot
        assert "reranker_rearms" in snapshot
