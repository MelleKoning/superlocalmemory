# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com

"""The recall marker must reach the wire, or the whole learning loop is dead.

WHY THIS EXISTS
---------------
``run_recall`` has always computed ``result.marker`` — an HMAC of the fact id —
on the hot path. Until 4.0.8 **no serialiser ever read it**, so every recall
computed the value and threw it away.

That single omission severed the closed loop end to end:

    recall emits marker  ->  (dropped here)  ->  agent's tool response has no
    marker  ->  post_tool_outcome finds nothing  ->  register_signal never
    called  ->  outcome settles at the formula base 0.5

and the consequences were measurable on a real store: 162 outcomes all at the
default label, all 294 source-quality observations at exactly 0.5, therefore
``alpha == beta`` for all 18 sources ("no quality signal has settled"), and 165
bandit arms with 4 plays between them.

The failure was invisible from every side. Recall worked. The hook was
installed and ran. The reward formula was correct. Nothing logged an error.
Only comparing what recall produced against what crossed the wire showed it.

OPTION B — markers are gated on ``session_id``
----------------------------------------------
A marker costs ~33 characters of the agent's context per result, and it can
only buy a signal when a ``pending_outcomes`` row exists to settle — which
happens only for session-bearing recalls. So markers ride along exactly when
they can be used, and ad-hoc recalls pay nothing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from superlocalmemory.server.recall_serializer import serialize_recall_response


def _response(n: int = 3, marker: bool = True):
    """A minimal RecallResponse duck-type, with or without engine markers."""
    results = []
    for i in range(n):
        fact = SimpleNamespace(
            fact_id=f"fact{i:04d}", memory_id=f"mem{i}", content=f"content {i}",
            fact_type="semantic", lifecycle="warm", access_count=0,
            created_at="2026-08-17T10:00:00+00:00",
        )
        r = SimpleNamespace(
            fact=fact, score=0.9, relevance_score=0.9, confidence=0.8,
            memory_confidence=0.8, ranking_score=1.0, rank_position=i,
            trust_score=0.99, channel_scores={}, evidence_chain=[],
        )
        if marker:
            r.marker = f"slm:fact:fact{i:04d}:abcd1234"
        results.append(r)
    return SimpleNamespace(results=results, no_confident_match=False)


class TestMarkerReachesTheWire:
    def test_marker_is_emitted_when_requested(self):
        out, _ = serialize_recall_response(_response(), include_marker=True)
        assert out, "no results serialized"
        assert all("marker" in r for r in out), (
            "marker dropped by the serializer — the learning loop is severed"
        )
        assert out[0]["marker"] == "slm:fact:fact0000:abcd1234"

    def test_marker_is_absent_by_default(self):
        """Option B: ad-hoc recalls must not pay context for an unusable signal."""
        out, _ = serialize_recall_response(_response())
        assert out
        assert not any("marker" in r for r in out)

    def test_absent_engine_marker_yields_no_key(self):
        """A missing marker must not become an empty string — the hook would
        then have to distinguish "not emitted" from "failed to compute"."""
        out, _ = serialize_recall_response(_response(marker=False), include_marker=True)
        assert out
        assert not any("marker" in r for r in out)

    def test_marker_survives_the_content_budget(self):
        """Budgeting rewrites result dicts; a stubbed result must keep its
        marker or long recalls would silently stop producing signals."""
        out, _ = serialize_recall_response(
            _response(n=6), include_marker=True, per_fact_max=5, total_max=5,
        )
        assert any(r.get("stub") for r in out), "budget did not stub anything"
        assert all("marker" in r for r in out), "stubbed results lost their marker"


class TestMarkerFormatMatchesTheValidator:
    """The hook validates before trusting. A format drift on either side
    silently reverts the loop to the state this test exists to prevent."""

    def test_emitted_marker_matches_the_hook_regex(self):
        from superlocalmemory.core.recall_pipeline import _emit_marker
        from superlocalmemory.hooks import post_tool_outcome_hook as hook

        marker = _emit_marker("abcdef0123456789")
        assert marker.startswith("slm:fact:")

        pattern = getattr(hook, "_MARKER_RE", None)
        if pattern is None:  # name drift, not a contract change
            pytest.skip("hook marker regex not exposed as _MARKER_RE")
        assert pattern.search(marker), (
            f"hook regex does not match the marker recall emits: {marker!r}"
        )


class TestCallSitesAreGated:
    """Every agent-facing recall path must pass include_marker; the dashboard
    trace endpoint deliberately must not."""

    @pytest.mark.parametrize(
        "module_path",
        [
            "superlocalmemory/server/unified_daemon.py",
            "superlocalmemory/core/recall_worker.py",
        ],
    )
    def test_agent_paths_gate_on_session_id(self, module_path):
        from pathlib import Path

        src = Path("src") / module_path
        text = src.read_text(encoding="utf-8")
        assert "include_marker=bool(session_id)" in text, (
            f"{module_path} serializes recalls without threading the marker gate"
        )

    def test_dashboard_trace_does_not_emit_markers(self):
        """/api/v3/recall/trace is a diagnostic view, not an agent that will
        later cite a fact. Markers there are pure noise in the trace output."""
        from pathlib import Path

        text = Path("src/superlocalmemory/server/routes/v3_api.py").read_text(encoding="utf-8")
        assert "include_marker" not in text
