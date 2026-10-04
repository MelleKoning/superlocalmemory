# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.retrieval.reranker — Cross-Encoder Reranker.

V3.3.3: Tests for the subprocess-isolated architecture. The main process
never imports torch/sentence_transformers. All model work runs in a child
process via JSON over stdin/stdout.

Covers:
  - Initialization (model name, backend, warmup trigger)
  - Worker lifecycle (spawn, kill, respawn, idle timer)
  - rerank() with worker available -> scored and sorted
  - rerank() with worker unavailable -> fallback to existing scores
  - rerank() with empty candidates
  - score_pair() via worker subprocess
  - is_available property (worker ping)
  - Worker recycling after N requests
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.retrieval.reranker import CrossEncoderReranker
from superlocalmemory.storage.models import AtomicFact

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fact(fact_id: str, content: str = "") -> AtomicFact:
    return AtomicFact(
        fact_id=fact_id, memory_id="m0",
        content=content or f"Content for {fact_id}",
    )


def _make_candidates(n: int = 3) -> list[tuple[AtomicFact, float]]:
    return [
        (_make_fact(f"f{i}", f"Document {i}"), 0.5 - i * 0.1)
        for i in range(n)
    ]


def _make_reranker(**kwargs) -> CrossEncoderReranker:
    """Create a reranker with background warmup disabled (no real subprocess)."""
    with patch.object(CrossEncoderReranker, "_start_background_warmup"):
        return CrossEncoderReranker(**kwargs)


# ---------------------------------------------------------------------------
# Initialization & warmup
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_default_model_is_l12(self) -> None:
        """Default model is MiniLM-L-12-v2 (better quality, ONNX backend)."""
        reranker = _make_reranker()
        assert reranker._model_name == "cross-encoder/ms-marco-MiniLM-L-12-v2"
        assert reranker._backend == "onnx"

    def test_model_not_loaded_at_init(self) -> None:
        """Worker hasn't confirmed model ready yet."""
        reranker = _make_reranker(model_name="fake-model")
        assert reranker._model_loaded is False
        assert reranker._worker_proc is None

    def test_background_warmup_called_on_init(self) -> None:
        """Constructor triggers background warmup."""
        with patch.object(
            CrossEncoderReranker, "_start_background_warmup",
        ) as mock_warmup:
            CrossEncoderReranker("fake-model")
            mock_warmup.assert_called_once()

    def test_custom_model_and_backend(self) -> None:
        reranker = _make_reranker(model_name="test-model", backend="")
        assert reranker._model_name == "test-model"
        assert reranker._backend == ""


# ---------------------------------------------------------------------------
# Worker lifecycle
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_spawn(tmp_path, monkeypatch):
    """Spawn bookkeeping kept off the machine: no singleton PID check and a
    tmp PID file. (``_ensure_worker`` still makes its 1 s crash check.)

    Replaces an ``importlib.reload`` of the reranker module that these two
    tests used (and were skipped over since v3.4.22): reloading rebinds the
    module's globals under every other test holding the old class, which is
    itself suite-order pollution. Patching the instance's ``_spawn_process``
    needs no reload.
    """
    from superlocalmemory.retrieval import reranker as mod

    monkeypatch.setattr(mod, "_is_reranker_worker_alive", lambda: False)
    monkeypatch.setattr(mod, "_RERANKER_PID_FILE", tmp_path / ".reranker.pid")
    return tmp_path / ".reranker.pid"


class TestWorkerManagement:
    def test_shutdown_cancels_and_joins_background_warmup(self) -> None:
        """Explicit shutdown leaves no warmup thread attached to a reranker."""
        warmup_started = threading.Event()

        def _ensure_worker(reranker: CrossEncoderReranker) -> None:
            reranker._worker_proc = object()

        def _wait_for_shutdown(reranker: CrossEncoderReranker, *_args, **_kwargs):
            warmup_started.set()
            reranker._shutdown_event.wait(timeout=5)
            return None

        with patch.object(CrossEncoderReranker, "_ensure_worker", _ensure_worker), patch.object(
            CrossEncoderReranker, "_send_request", _wait_for_shutdown,
        ):
            reranker = CrossEncoderReranker(model_name="fake-model")
            assert warmup_started.wait(timeout=1)

            reranker.shutdown(timeout=1)

        assert reranker._shutdown_event.is_set()
        assert not reranker._warmup_thread.is_alive()

    def test_shutdown_prevents_future_background_warmup(self) -> None:
        """A closed reranker must stay closed when fallback scoring is used."""
        reranker = _make_reranker(model_name="fake-model")
        reranker.shutdown()

        with patch.object(reranker, "_start_background_warmup") as warmup:
            reranker.rerank("query", _make_candidates())

        warmup.assert_not_called()

    def test_ensure_worker_spawns_subprocess(self, isolated_spawn) -> None:
        reranker = _make_reranker(model_name="fake-model")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 99999
        with patch.object(reranker, "_spawn_process", return_value=mock_proc):
            reranker._ensure_worker()
        assert reranker._worker_proc is mock_proc
        assert isolated_spawn.read_text() == "99999"

    def test_ensure_worker_noop_if_alive(self) -> None:
        reranker = _make_reranker(model_name="fake-model")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Still alive
        reranker._worker_proc = mock_proc
        with patch("subprocess.Popen") as mock_popen:
            reranker._ensure_worker()
            mock_popen.assert_not_called()

    def test_ensure_worker_respawns_if_dead(self, isolated_spawn) -> None:
        reranker = _make_reranker(model_name="fake-model")
        dead_proc = MagicMock()
        dead_proc.poll.return_value = 1  # Exited
        reranker._worker_proc = dead_proc
        new_proc = MagicMock()
        new_proc.poll.return_value = None
        new_proc.pid = 99998
        with patch.object(reranker, "_spawn_process", return_value=new_proc) as spawn:
            reranker._ensure_worker()
        spawn.assert_called_once()
        assert reranker._worker_proc is new_proc
        assert isolated_spawn.read_text() == "99998"

    def test_ensure_worker_handles_spawn_failure(self) -> None:
        reranker = _make_reranker(model_name="fake-model")
        with patch("subprocess.Popen", side_effect=OSError("spawn failed")):
            reranker._ensure_worker()
        assert reranker._worker_proc is None

    def test_kill_worker_sends_quit(self) -> None:
        reranker = _make_reranker(model_name="fake-model")
        mock_proc = MagicMock()
        reranker._worker_proc = mock_proc
        reranker._kill_worker()
        mock_proc.stdin.write.assert_called_with('{"cmd":"quit"}\n')
        mock_proc.wait.assert_called_once()
        assert reranker._worker_proc is None

    def test_kill_worker_force_kills_on_timeout(self) -> None:
        reranker = _make_reranker(model_name="fake-model")
        mock_proc = MagicMock()
        mock_proc.stdin.write.side_effect = BrokenPipeError("pipe closed")
        reranker._worker_proc = mock_proc
        reranker._kill_worker()
        mock_proc.kill.assert_called_once()
        assert reranker._worker_proc is None

    def test_unload_kills_worker(self) -> None:
        reranker = _make_reranker(model_name="fake-model")
        mock_proc = MagicMock()
        reranker._worker_proc = mock_proc
        reranker.unload()
        assert reranker._worker_proc is None


# ---------------------------------------------------------------------------
# _send_request
# ---------------------------------------------------------------------------

class TestSendRequest:
    def test_send_request_success(self) -> None:
        reranker = _make_reranker(model_name="fake-model")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        reranker._worker_proc = mock_proc

        with patch.object(
            reranker, "_readline_with_timeout",
            return_value='{"ok": true, "scores": [0.9]}\n',
        ):
            resp = reranker._send_request({"cmd": "ping"})

        assert resp == {"ok": True, "scores": [0.9]}

    def test_send_request_returns_none_on_timeout(self) -> None:
        reranker = _make_reranker(model_name="fake-model")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        reranker._worker_proc = mock_proc

        with patch.object(
            reranker, "_readline_with_timeout", return_value="",
        ):
            resp = reranker._send_request({"cmd": "ping"})

        assert resp is None

    def test_send_request_returns_none_when_no_worker(self) -> None:
        reranker = _make_reranker(model_name="fake-model")
        reranker._worker_proc = None
        with patch.object(reranker, "_ensure_worker"):
            resp = reranker._send_request({"cmd": "ping"})
        assert resp is None

    def test_send_request_handles_broken_pipe(self) -> None:
        reranker = _make_reranker(model_name="fake-model")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdin.write.side_effect = BrokenPipeError("pipe")
        reranker._worker_proc = mock_proc

        resp = reranker._send_request({"cmd": "ping"})
        assert resp is None
        assert reranker._model_loaded is False


# ---------------------------------------------------------------------------
# rerank() — worker available
# ---------------------------------------------------------------------------

class TestRerankWithModel:
    def test_rerank_sorts_by_cross_encoder_score(self) -> None:
        reranker = _make_reranker(model_name="fake-model")
        reranker._model_loaded = True
        candidates = _make_candidates(3)

        with patch.object(reranker, "_send_request", return_value={
            "ok": True,
            "scores": [0.1, 0.5, 0.9],
        }):
            results = reranker.rerank("query", candidates, top_k=10)

        # Scores [0.1, 0.5, 0.9] -> f2 (0.9) should be first
        assert results[0][0].fact_id == "f2"
        assert results[0][1] == pytest.approx(0.9)
        assert results[1][0].fact_id == "f1"
        assert results[2][0].fact_id == "f0"

    def test_rerank_respects_top_k(self) -> None:
        reranker = _make_reranker(model_name="fake-model")
        reranker._model_loaded = True
        candidates = _make_candidates(3)

        with patch.object(reranker, "_send_request", return_value={
            "ok": True,
            "scores": [0.1, 0.5, 0.9],
        }):
            results = reranker.rerank("query", candidates, top_k=2)

        assert len(results) == 2

    def test_rerank_passes_correct_documents(self) -> None:
        reranker = _make_reranker(model_name="fake-model")
        reranker._model_loaded = True
        candidates = [
            (_make_fact("f1", "doc one"), 0.5),
            (_make_fact("f2", "doc two"), 0.3),
        ]

        with patch.object(reranker, "_send_request", return_value={
            "ok": True,
            "scores": [0.8, 0.4],
        }) as mock_send:
            reranker.rerank("my query", candidates)

        req = mock_send.call_args[0][0]
        assert req["cmd"] == "rerank"
        assert req["query"] == "my query"
        assert req["documents"] == ["doc one", "doc two"]


# ---------------------------------------------------------------------------
# rerank() — fallback (worker not ready or failed)
# ---------------------------------------------------------------------------

class TestRerankFallback:
    def test_fallback_when_model_not_loaded(self) -> None:
        """When worker hasn't loaded model yet, return by existing score."""
        reranker = _make_reranker(model_name="fake-model")
        reranker._model_loaded = False
        candidates = [
            (_make_fact("f1"), 0.3),
            (_make_fact("f2"), 0.9),
            (_make_fact("f3"), 0.6),
        ]
        results = reranker.rerank("query", candidates)
        assert results[0][0].fact_id == "f2"
        assert results[1][0].fact_id == "f3"

    def test_fallback_when_worker_returns_none(self) -> None:
        """When worker crashes or times out, return by existing score."""
        reranker = _make_reranker(model_name="fake-model")
        reranker._model_loaded = True
        candidates = _make_candidates(3)

        with patch.object(reranker, "_send_request", return_value=None):
            results = reranker.rerank("query", candidates)

        # Fallback: sorted by existing score, f0 (0.5) > f1 (0.4) > f2 (0.3)
        assert results[0][0].fact_id == "f0"

    def test_fallback_when_worker_returns_error(self) -> None:
        reranker = _make_reranker(model_name="fake-model")
        reranker._model_loaded = True
        candidates = _make_candidates(3)

        with patch.object(reranker, "_send_request", return_value={"ok": False}):
            results = reranker.rerank("query", candidates)

        assert results[0][0].fact_id == "f0"

    def test_fallback_respects_top_k(self) -> None:
        reranker = _make_reranker(model_name="fake-model")
        reranker._model_loaded = False
        candidates = _make_candidates(5)
        results = reranker.rerank("query", candidates, top_k=2)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# rerank() — empty candidates
# ---------------------------------------------------------------------------

class TestRerankEmpty:
    def test_empty_candidates_returns_empty(self) -> None:
        reranker = _make_reranker(model_name="fake-model")
        reranker._model_loaded = True
        assert reranker.rerank("query", []) == []


# ---------------------------------------------------------------------------
# score_pair()
# ---------------------------------------------------------------------------

class TestScorePair:
    def test_score_pair_with_worker(self) -> None:
        reranker = _make_reranker(model_name="fake-model")

        with patch.object(reranker, "_send_request", return_value={
            "ok": True,
            "score": 0.75,
        }):
            score = reranker.score_pair("query", "document text")

        assert score == pytest.approx(0.75)

    def test_score_pair_worker_failure(self) -> None:
        reranker = _make_reranker(model_name="fake-model")

        with patch.object(reranker, "_send_request", return_value=None):
            score = reranker.score_pair("query", "doc")

        assert score == 0.0

    def test_score_pair_sends_correct_request(self) -> None:
        reranker = _make_reranker(model_name="test-model", backend="onnx")

        with patch.object(reranker, "_send_request", return_value={
            "ok": True,
            "score": 0.5,
        }) as mock_send:
            reranker.score_pair("my query", "my doc")

        req = mock_send.call_args[0][0]
        assert req["cmd"] == "score"
        assert req["query"] == "my query"
        assert req["document"] == "my doc"
        assert req["model_name"] == "test-model"
        assert req["backend"] == "onnx"


# ---------------------------------------------------------------------------
# is_available property
# ---------------------------------------------------------------------------

class TestIsAvailable:
    def test_available_when_worker_responds(self) -> None:
        reranker = _make_reranker(model_name="fake-model")

        with patch.object(
            reranker, "_send_request", return_value={"ok": True},
        ):
            assert reranker.is_available is True

    def test_not_available_when_worker_fails(self) -> None:
        reranker = _make_reranker(model_name="fake-model")

        with patch.object(reranker, "_send_request", return_value=None):
            assert reranker.is_available is False

    def test_not_available_when_worker_returns_error(self) -> None:
        reranker = _make_reranker(model_name="fake-model")

        with patch.object(
            reranker, "_send_request", return_value={"ok": False},
        ):
            assert reranker.is_available is False


# ---------------------------------------------------------------------------
# Worker recycling (4.1.18)
# ---------------------------------------------------------------------------

class TestWorkerRecycle:
    """The module docstring has always claimed recycling was covered; no test
    existed, and the path was broken.

    After 500 requests the worker is recycled. The old code then sent the live
    request straight to the freshly spawned worker under the 15 s request
    timeout. A fresh worker has to import torch and load the model first —
    22 to 63 s in the daemon logs — so it always timed out. Both recycles in
    the live daemon's history died exactly that way, and reranking stayed off
    until the daemon was restarted.
    """

    def _recycle_ready(self):
        from superlocalmemory.retrieval import reranker as mod

        reranker = _make_reranker(model_name="fake-model")
        old = MagicMock()
        old.poll.return_value = None
        reranker._worker_proc = old
        reranker._model_loaded = True
        reranker._request_count = mod._WORKER_RECYCLE_AFTER
        return reranker

    def test_recycle_never_sends_a_live_request_to_a_cold_worker(self) -> None:
        """M-12: the request at the threshold is answered by the warm worker;
        the replacement is warmed separately and only ever receives a load."""
        reranker = self._recycle_ready()
        old = reranker._worker_proc

        with patch.object(reranker, "_ensure_worker") as spawn, \
                patch.object(reranker, "_readline_with_timeout",
                             return_value='{"ok": true, "scores": [1.0]}') as read, \
                patch.object(reranker, "_begin_replacement") as replace:
            resp = reranker._send_request(
                {"cmd": "rerank", "query": "q", "documents": ["d"]},
                timeout=15.0, block=False,
            )

        assert resp == {"ok": True, "scores": [1.0]}
        spawn.assert_not_called()
        old.stdin.write.assert_called_once()
        read.assert_called_once()
        replace.assert_called_once()

    def test_recycle_keeps_the_warm_worker_until_its_replacement_is_ready(self) -> None:
        """4.1.18 dropped the worker here and set the model unloaded, which
        left every recall unranked for a whole cold load."""
        reranker = self._recycle_ready()
        old = reranker._worker_proc
        with patch.object(reranker, "_begin_replacement"), \
                patch.object(reranker, "_readline_with_timeout",
                             return_value='{"ok": true}'):
            reranker._send_request({"cmd": "ping"})
        assert reranker._worker_proc is old
        assert reranker._model_loaded is True
