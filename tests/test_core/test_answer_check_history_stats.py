# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The Answer Check tab's numbers: honest denominators, withheld small samples."""

from __future__ import annotations

import pytest

from superlocalmemory.core import answer_check_history_stats as stats


def _row(status="judged", abstained=False, detail="", result_count=5, backend="laya",
         total_ms=800.0, judge_ms=200.0, occurred_ms=0, origin=""):
    return {"status": status, "abstained": abstained, "detail": detail,
            "result_count": result_count, "backend": backend, "total_ms": total_ms,
            "judge_ms": judge_ms, "occurred_ms": occurred_ms, "origin": origin}


def test_nearest_rank_known_values() -> None:
    values = [float(v) for v in range(1, 101)]
    assert stats.nearest_rank(values, 0.50) == 50.0
    assert stats.nearest_rank(values, 0.95) == 95.0
    assert stats.nearest_rank([], 0.5) is None
    assert stats.nearest_rank([7.0], 0.95) == 7.0


def test_wilson_matches_reference() -> None:
    lo, hi = stats.wilson_interval(10, 20)
    assert lo == pytest.approx(0.2993, abs=1e-4)
    assert hi == pytest.approx(0.7007, abs=1e-4)
    lo0, _ = stats.wilson_interval(0, 20)
    assert lo0 == 0.0
    assert stats.wilson_interval(0, 0) is None


def test_rate_hidden_below_min_sample() -> None:
    nineteen = [_row(abstained=i < 5) for i in range(19)]
    out = stats.summarize(nineteen, ceiling_ms=3000.0)["abstention"]
    assert out["n"] == 19 and out["rate"] is None and out["ci95"] is None
    assert out["enough"] is False and out["min_sample"] == 20
    twenty = [_row(abstained=i < 5) for i in range(20)]
    out = stats.summarize(twenty, ceiling_ms=3000.0)["abstention"]
    assert out["enough"] is True and out["rate"] == 0.25 and len(out["ci95"]) == 2


def test_nothing_found_not_in_denominator() -> None:
    rows = [_row(abstained=True) for _ in range(5)] + [_row() for _ in range(15)]
    rows += [_row(status="skipped", detail="no_results", result_count=0, abstained=True)
             for _ in range(30)]
    rows += [_row(status="off", abstained=False) for _ in range(30)]
    out = stats.summarize(rows, ceiling_ms=3000.0)
    assert out["abstention"]["n"] == 20 and out["abstention"]["k"] == 5
    assert out["abstention"]["rate"] == 0.25
    assert out["counts"]["nothing_found"] == 30
    assert out["counts"]["not_checked"]["off"] == 30
    assert out["counts"]["total"] == 80


@pytest.mark.parametrize("status,detail,abstained,count,expected", [
    ("off", "", False, 3, "not_checked_off"),
    ("judged", "", False, 3, "answered"),
    ("judged", "", True, 3, "abstained"),
    ("skipped", "no_results", True, 0, "nothing_found"),
    ("skipped", "budget", False, 3, "not_checked_time"),
    ("skipped", "other_profile_memory", False, 3, "not_checked_shared"),
    ("busy", "", False, 3, "not_checked_busy"),
    ("warming", "", False, 3, "not_checked_loading"),
    ("unavailable", "", False, 3, "not_checked_unavailable"),
    ("skipped", "", True, 0, "nothing_found"),
    ("nonsense", "", False, 3, "not_checked_unavailable"),
])
def test_outcome_key_table(status, detail, abstained, count, expected) -> None:
    assert stats.outcome_key(status, detail, abstained, count) == expected


def test_latency_against_ceiling_and_judge_only_for_judged() -> None:
    rows = [_row(total_ms=float(t), judge_ms=100.0, occurred_ms=t) for t in range(100, 2100, 100)]
    rows.append(_row(status="off", total_ms=3500.0, judge_ms=0.0, occurred_ms=99999))
    out = stats.summarize(rows, ceiling_ms=3000.0, recent_n=5)["latency"]
    assert out["ceiling_ms"] == 3000.0
    assert out["over_ceiling"] == 1
    assert out["total"]["n"] == 21 and out["total"]["max"] == 3500.0
    assert out["judge"]["n"] == 20  # the "off" row has no judge time to report
    assert out["recent_total_ms"] == [1700.0, 1800.0, 1900.0, 2000.0, 3500.0]


def test_by_judge_counts_judged_only() -> None:
    rows = [_row(backend="laya"), _row(backend="jev"), _row(status="off", backend="")]
    assert stats.summarize(rows, ceiling_ms=3000.0)["by_judge"] == {"laya": 1, "jev": 1}
