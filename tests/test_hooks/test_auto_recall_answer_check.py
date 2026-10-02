# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""AutoRecall.get_session_context auto-injects memories without the agent
asking (SessionStart / ``slm://context`` MCP resource / ``slm session-context
--full``). When the judge says none of the retrieved memories answers the
query (``abstention_reason == "judged_insufficient"``), the memories are kept
and one line saying so is added above them — the same line session_init and
the CLI print (H-3, 4.1.18: replacing the list hid real context on every
false abstention). With no judge configured, behaviour must be byte-identical
to before (empty ``abstention_reason`` is the overwhelming majority case).
"""

from __future__ import annotations

from types import SimpleNamespace

from superlocalmemory.hooks.auto_recall import AutoRecall


def _response(fact_ids, score=0.9, content="Some memory", **meta):
    results = [
        SimpleNamespace(
            fact=SimpleNamespace(fact_id=fid, content=content, created_at=""),
            score=score,
        )
        for fid in fact_ids
    ]
    defaults = dict(abstained=False, abstention_reason=None)
    defaults.update(meta)
    return SimpleNamespace(results=results, **defaults)


class TestAutoRecallJudgedInsufficient:
    def test_judged_insufficient_keeps_the_list_and_adds_one_line(self) -> None:
        def fake_recall(query, limit=10, **_):
            return _response(
                ["f1", "f2"], content="JWT with 1h expiry",
                abstained=True, abstention_reason="judged_insufficient",
                calibration_status="measured_small_sample_not_calibrated",
                answer_confidence=0.04,
            )

        auto = AutoRecall(recall_fn=fake_recall)
        out = auto.get_session_context(query="what did we ship")

        assert "JWT with 1h expiry" in out
        assert out.startswith("Answer check: none of these memories answers the question")

    def test_judged_and_answered_still_injects_the_list(self) -> None:
        """abstained=False (judge ran, found an answer) -> unchanged list."""
        def fake_recall(query, limit=10, **_):
            return _response(
                ["f1"], content="JWT with 1h expiry",
                abstained=False, abstention_reason=None,
            )

        auto = AutoRecall(recall_fn=fake_recall)
        out = auto.get_session_context(query="what did we ship")

        assert "JWT with 1h expiry" in out
        assert "none of these memories" not in out

    def test_no_judge_configured_unchanged(self) -> None:
        """abstention_reason absent entirely (old engines) -> unchanged."""
        def fake_recall(query, limit=10, **_):
            return _response(["f1"], content="JWT with 1h expiry")

        auto = AutoRecall(recall_fn=fake_recall)
        out = auto.get_session_context(query="what did we ship")

        assert "JWT with 1h expiry" in out

    def test_the_verdict_is_reported_even_when_no_memory_clears_the_local_threshold(self) -> None:
        """The judge's verdict is reported independently of AutoRecall's own
        relevance_threshold filtering."""
        def fake_recall(query, limit=10, **_):
            return _response(
                ["f1"], score=0.1, abstained=True, abstention_reason="judged_insufficient",
                calibration_status="measured_small_sample_not_calibrated",
                answer_confidence=0.04,
            )

        auto = AutoRecall(recall_fn=fake_recall)
        out = auto.get_session_context(query="what did we ship")

        assert out.startswith("Answer check: none of these memories answers the question")
