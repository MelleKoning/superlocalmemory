# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""The answer-quality counts are right on hand-checked synthetic cases."""

from __future__ import annotations

import json

import pytest

from superlocalmemory.evaluation.answer_quality import (
    CORRECT_ABSTAIN,
    CORRECT_ACCEPT,
    FALSE_ABSTAIN,
    FALSE_ACCEPT,
    GoldFileError,
    JudgedRecall,
    RankedResult,
    choose_threshold,
    compare_ranks,
    first_right_rank,
    judge_report,
    load_gold,
    nearest_rank_ms,
    outcome,
    retrieval_report,
)


def _gold(tmp_path, rows):
    path = tmp_path / "gold.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


ROWS = [
    {"qid": "A", "question": "q a", "category": "release", "gold_memory_ids": ["m1"]},
    {"qid": "B", "question": "q b", "category": "howto", "gold_fact_ids": ["f9"]},
    {"qid": "C", "question": "q c", "category": "howto", "gold_memory_ids": ["m3"]},
    {"qid": "U", "question": "q u", "answerable": False},
]


class TestGold:
    def test_loads_and_matches_by_memory_or_fact(self, tmp_path) -> None:
        a, b, _c, u = load_gold(_gold(tmp_path, ROWS))
        assert first_right_rank(a, [RankedResult("m0", "f0"), RankedResult("m1", "f1")]) == 2
        assert first_right_rank(b, [RankedResult("mx", "f9")]) == 1
        assert first_right_rank(a, [RankedResult("m0", "f0")]) is None
        assert not u.answerable

    @pytest.mark.parametrize("row", [
        {"qid": "X", "question": "q"},                                  # no gold
        {"qid": "X", "question": "q", "answerable": False, "gold_memory_ids": ["m"]},
        {"qid": "", "question": "q", "gold_memory_ids": ["m"]},
        {"qid": "X", "question": "q", "gold_memory_ids": "m"},
        {"qid": "X", "question": "q", "answerable": "yes", "gold_memory_ids": ["m"]},
    ])
    def test_refuses_rows_that_cannot_be_scored(self, tmp_path, row) -> None:
        with pytest.raises(GoldFileError):
            load_gold(_gold(tmp_path, [row]))

    def test_refuses_duplicates_and_empty(self, tmp_path) -> None:
        with pytest.raises(GoldFileError):
            load_gold(_gold(tmp_path, [ROWS[0], ROWS[0]]))
        empty = tmp_path / "e.jsonl"
        empty.write_text("\n", encoding="utf-8")
        with pytest.raises(GoldFileError):
            load_gold(empty)


class TestRetrievalReport:
    def test_hits_mrr_and_categories(self, tmp_path) -> None:
        questions = load_gold(_gold(tmp_path, ROWS))
        report = retrieval_report({"A": 1, "B": 3, "C": None, "U": None}, questions,
                                  [100.0, 200.0, 300.0, 4000.0])
        overall = report["overall"]
        assert (overall["n"], overall["hit@1"], overall["hit@3"], overall["hit@5"]) == (3, 1, 2, 2)
        assert overall["mrr"] == pytest.approx((1 + 1 / 3) / 3, abs=1e-4)
        assert report["by_category"]["howto"]["hit@3"] == 1
        assert report["misses_at_1"] == ["B", "C"]
        assert report["latency_ms"]["p50"] == 200.0
        assert report["latency_ms"]["p95"] == 4000.0

    def test_a_missing_result_is_an_error_not_a_miss(self, tmp_path) -> None:
        questions = load_gold(_gold(tmp_path, ROWS))
        with pytest.raises(ValueError):
            retrieval_report({"A": 1}, questions)

    def test_compare_lists_every_change_at_1(self) -> None:
        diff = compare_ranks({"A": 1, "B": 2, "C": None}, {"A": 2, "B": 1, "C": None})
        assert diff["newly_wrong_at_1"] == ["A"]
        assert diff["newly_right_at_1"] == ["B"]
        assert diff["worse_rank"] == ["A"]

    def test_nearest_rank(self) -> None:
        assert nearest_rank_ms([5.0], 95) == 5.0
        assert nearest_rank_ms(list(map(float, range(1, 101))), 95) == 95.0
        with pytest.raises(ValueError):
            nearest_rank_ms([], 50)


class TestJudgeReport:
    def test_the_four_outcomes(self) -> None:
        assert outcome(True, False) == CORRECT_ACCEPT
        assert outcome(True, True) == FALSE_ABSTAIN
        assert outcome(False, False) == FALSE_ACCEPT
        assert outcome(False, True) == CORRECT_ABSTAIN

    def _recalls(self):
        # answered at 0.9, 0.7, 0.4; not answered at 0.6, 0.2
        return [JudgedRecall("a", "laya", True, False, 0.9),
                JudgedRecall("b", "laya", True, True, 0.7),
                JudgedRecall("c", "laya", True, True, 0.4),
                JudgedRecall("d", "laya", False, False, 0.6),
                JudgedRecall("e", "laya", False, True, 0.2),
                JudgedRecall("f", "jev", True, False, None)]

    def test_observed_counts_per_provider(self) -> None:
        report = judge_report(self._recalls(), thresholds=(0.5,))
        laya = report["laya"]["observed"]
        assert (laya[CORRECT_ACCEPT], laya[FALSE_ABSTAIN],
                laya[FALSE_ACCEPT], laya[CORRECT_ABSTAIN]) == (1, 2, 1, 1)
        assert report["jev"]["without_confidence"] == 1

    def test_the_sweep_recounts_at_each_threshold(self) -> None:
        (row,) = judge_report(self._recalls(), thresholds=(0.5,))["laya"]["sweep"]
        assert (row[CORRECT_ACCEPT], row[FALSE_ABSTAIN],
                row[FALSE_ACCEPT], row[CORRECT_ABSTAIN]) == (2, 1, 1, 1)

    def test_choose_keeps_false_accepts_under_the_limit(self) -> None:
        sweep = judge_report(self._recalls(), thresholds=(0.3, 0.5, 0.65))["laya"]["sweep"]
        best = choose_threshold(sweep, max_false_accept_rate=0.0)
        assert best["threshold"] == 0.65 and best[FALSE_ACCEPT] == 0
        assert choose_threshold(sweep, max_false_accept_rate=0.5)["threshold"] == 0.3

    def test_choose_needs_both_kinds(self) -> None:
        only_answered = [JudgedRecall("a", "x", True, False, 0.9)]
        sweep = judge_report(only_answered, thresholds=(0.5,))["x"]["sweep"]
        assert choose_threshold(sweep, max_false_accept_rate=1.0) is None
