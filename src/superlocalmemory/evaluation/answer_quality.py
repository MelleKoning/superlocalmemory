# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Answer-quality measurement against a hand-labelled question set.

A gold file is JSON Lines, one question per line::

    {"qid": "Q01", "question": "...", "category": "release",
     "answerable": true, "gold_memory_ids": ["..."], "gold_fact_ids": []}

A result is right when its memory id is in ``gold_memory_ids`` or its fact id
is in ``gold_fact_ids``. A row with ``answerable`` false has no gold: the
right outcome for it is that the answer check abstains.

Two measurements, both pure functions of what recall returned:

* retrieval — where the first right memory sits (hit@1/3/5, MRR), overall and
  per category, plus the recall latency spread;
* judges — what the answer check did with each answer, per provider:
  correct-accept, false-accept, false-abstain, correct-abstain, and the same
  four counts at every threshold, so a threshold is chosen from data rather
  than guessed.

Nothing here reads a store, opens a socket or prints memory text. The caller
owns the I/O; this module only counts.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

HIT_DEPTHS = (1, 3, 5)


class GoldFileError(ValueError):
    """The gold file cannot support a truthful measurement."""


@dataclass(frozen=True)
class GoldQuestion:
    qid: str
    question: str
    category: str
    answerable: bool
    gold_memory_ids: frozenset[str] = field(default_factory=frozenset)
    gold_fact_ids: frozenset[str] = field(default_factory=frozenset)

    def is_right(self, memory_id: str, fact_id: str) -> bool:
        return memory_id in self.gold_memory_ids or fact_id in self.gold_fact_ids


@dataclass(frozen=True)
class RankedResult:
    """One returned memory, as much as scoring needs: its two ids."""

    memory_id: str
    fact_id: str


def _ids(value: object, *, field_name: str, qid: str) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise GoldFileError(f"{qid}: {field_name} must be a list of non-empty strings")
    return frozenset(value)


def parse_gold_row(row: object, *, line: int) -> GoldQuestion:
    if not isinstance(row, dict):
        raise GoldFileError(f"line {line}: each line must be a JSON object")
    qid = str(row.get("qid", "")).strip()
    question = str(row.get("question", "")).strip()
    if not qid or not question:
        raise GoldFileError(f"line {line}: qid and question are required")
    answerable = row.get("answerable", True)
    if not isinstance(answerable, bool):
        raise GoldFileError(f"{qid}: answerable must be true or false")
    memories = _ids(row.get("gold_memory_ids"), field_name="gold_memory_ids", qid=qid)
    facts = _ids(row.get("gold_fact_ids"), field_name="gold_fact_ids", qid=qid)
    if answerable and not (memories or facts):
        raise GoldFileError(f"{qid}: an answerable question needs at least one gold id")
    if not answerable and (memories or facts):
        raise GoldFileError(f"{qid}: an unanswerable question cannot have gold ids")
    return GoldQuestion(
        qid=qid, question=question,
        category=str(row.get("category") or "uncategorised"),
        answerable=answerable, gold_memory_ids=memories, gold_fact_ids=facts,
    )


def load_gold(path: Path) -> tuple[GoldQuestion, ...]:
    """Every question in ``path``; refuses duplicates and malformed rows."""
    questions: list[GoldQuestion] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for number, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise GoldFileError(f"line {number}: not JSON ({exc.msg})") from exc
            question = parse_gold_row(row, line=number)
            if question.qid in seen:
                raise GoldFileError(f"{question.qid}: duplicate qid")
            seen.add(question.qid)
            questions.append(question)
    if not questions:
        raise GoldFileError("the gold file has no questions")
    return tuple(questions)


def first_right_rank(question: GoldQuestion, results: Sequence[RankedResult]) -> int | None:
    """1-based rank of the first right result, or None when none is right."""
    for rank, result in enumerate(results, start=1):
        if question.is_right(result.memory_id, result.fact_id):
            return rank
    return None


def nearest_rank_ms(samples: Sequence[float], percentile: int) -> float:
    """The nearest-rank percentile of ``samples`` (no interpolation)."""
    if not samples:
        raise ValueError("no latency samples")
    ordered = sorted(samples)
    index = max(0, math.ceil(len(ordered) * percentile / 100) - 1)
    return ordered[index]


def _hit_block(ranks: Sequence[int | None]) -> dict:
    n = len(ranks)
    block: dict[str, float | int] = {"n": n}
    for depth in HIT_DEPTHS:
        hits = sum(1 for r in ranks if r is not None and r <= depth)
        block[f"hit@{depth}"] = hits
        block[f"hit@{depth}_rate"] = round(hits / n, 4) if n else 0.0
    block["mrr"] = round(sum(1.0 / r for r in ranks if r) / n, 4) if n else 0.0
    return block


def retrieval_report(
    ranks: Mapping[str, int | None],
    questions: Sequence[GoldQuestion],
    latencies_ms: Sequence[float] = (),
) -> dict:
    """hit@k and MRR over the answerable questions, overall and per category."""
    answerable = [q for q in questions if q.answerable]
    missing = [q.qid for q in answerable if q.qid not in ranks]
    if missing:
        raise ValueError(f"no result recorded for: {', '.join(missing)}")
    by_category: dict[str, list[int | None]] = {}
    for q in answerable:
        by_category.setdefault(q.category, []).append(ranks[q.qid])
    report = {
        "overall": _hit_block([ranks[q.qid] for q in answerable]),
        "by_category": {c: _hit_block(r) for c, r in sorted(by_category.items())},
        "misses_at_1": sorted(q.qid for q in answerable
                              if ranks[q.qid] is None or ranks[q.qid] > 1),
    }
    if latencies_ms:
        report["latency_ms"] = {
            "n": len(latencies_ms),
            "p50": round(nearest_rank_ms(latencies_ms, 50), 1),
            "p95": round(nearest_rank_ms(latencies_ms, 95), 1),
            "max": round(max(latencies_ms), 1),
        }
    return report


def compare_ranks(before: Mapping[str, int | None],
                  after: Mapping[str, int | None]) -> dict:
    """Which questions changed at #1 between two runs (by qid)."""
    def top(rank: int | None) -> bool:
        return rank == 1

    shared = sorted(set(before) & set(after))
    return {
        "newly_right_at_1": [q for q in shared if top(after[q]) and not top(before[q])],
        "newly_wrong_at_1": [q for q in shared if top(before[q]) and not top(after[q])],
        "worse_rank": [q for q in shared if before[q] is not None
                       and (after[q] is None or after[q] > before[q])],
        "better_rank": [q for q in shared if after[q] is not None
                        and (before[q] is None or after[q] < before[q])],
    }


# -- judges --------------------------------------------------------------------

#: How one checked recall turned out. "Answered" means the right memory was
#: within the top ``judged_depth`` results the check reads.
CORRECT_ACCEPT = "correct_accept"
FALSE_ACCEPT = "false_accept"
FALSE_ABSTAIN = "false_abstain"
CORRECT_ABSTAIN = "correct_abstain"
OUTCOMES = (CORRECT_ACCEPT, FALSE_ACCEPT, FALSE_ABSTAIN, CORRECT_ABSTAIN)


@dataclass(frozen=True)
class JudgedRecall:
    """One recall the answer check actually judged."""

    qid: str
    provider: str
    answered: bool          # the right memory was among what the check read
    abstained: bool
    confidence: float | None


def outcome(answered: bool, abstained: bool) -> str:
    if answered:
        return FALSE_ABSTAIN if abstained else CORRECT_ACCEPT
    return CORRECT_ABSTAIN if abstained else FALSE_ACCEPT


def _counts(items: Iterable[str]) -> dict[str, int]:
    out = {name: 0 for name in OUTCOMES}
    for item in items:
        out[item] += 1
    return out


def _rates(counts: Mapping[str, int]) -> dict[str, float]:
    answered = counts[CORRECT_ACCEPT] + counts[FALSE_ABSTAIN]
    unanswered = counts[FALSE_ACCEPT] + counts[CORRECT_ABSTAIN]
    return {
        # Of the recalls that held the answer, how many the check let through.
        "accept_rate_when_answered": round(counts[CORRECT_ACCEPT] / answered, 4)
        if answered else 0.0,
        # Of the recalls that did NOT hold the answer, how many it let through.
        "false_accept_rate": round(counts[FALSE_ACCEPT] / unanswered, 4)
        if unanswered else 0.0,
        "false_abstain_rate": round(counts[FALSE_ABSTAIN] / answered, 4)
        if answered else 0.0,
    }


def threshold_sweep(recalls: Sequence[JudgedRecall],
                    thresholds: Sequence[float]) -> list[dict]:
    """The four outcomes as they would be at each threshold (abstain below it).

    Only recalls that carry a confidence take part; the count of those that do
    not is reported by ``judge_report`` so nothing is silently dropped.
    """
    scored = [r for r in recalls if r.confidence is not None]
    rows = []
    for t in thresholds:
        counts = _counts(outcome(r.answered, r.confidence < t) for r in scored)
        rows.append({"threshold": t, **counts, **_rates(counts)})
    return rows


DEFAULT_SWEEP = tuple(round(x * 0.05, 2) for x in range(0, 21))


def judge_report(recalls: Sequence[JudgedRecall],
                 thresholds: Sequence[float] = DEFAULT_SWEEP) -> dict:
    """Per provider: observed outcomes, their rates, and the threshold sweep."""
    by_provider: dict[str, list[JudgedRecall]] = {}
    for r in recalls:
        by_provider.setdefault(r.provider, []).append(r)
    report = {}
    for provider, items in sorted(by_provider.items()):
        counts = _counts(outcome(r.answered, r.abstained) for r in items)
        report[provider] = {
            "n": len(items),
            "without_confidence": sum(1 for r in items if r.confidence is None),
            "observed": {**counts, **_rates(counts)},
            "sweep": threshold_sweep(items, thresholds),
        }
    return report


def choose_threshold(sweep: Sequence[Mapping], *, max_false_accept_rate: float) -> dict | None:
    """The threshold that lets the most answered recalls through while keeping
    the false-accept rate at or under the limit. Ties go to the higher
    threshold (the more careful one). None when no threshold meets the limit
    or the sweep has nothing on one side (both kinds are needed to choose).
    """
    usable = [row for row in sweep
              if row[CORRECT_ACCEPT] + row[FALSE_ABSTAIN] > 0
              and row[FALSE_ACCEPT] + row[CORRECT_ABSTAIN] > 0
              and row["false_accept_rate"] <= max_false_accept_rate]
    if not usable:
        return None
    return max(usable, key=lambda row: (row[CORRECT_ACCEPT], row["threshold"]))


__all__ = [
    "CORRECT_ABSTAIN",
    "CORRECT_ACCEPT",
    "DEFAULT_SWEEP",
    "FALSE_ABSTAIN",
    "FALSE_ACCEPT",
    "GoldFileError",
    "GoldQuestion",
    "JudgedRecall",
    "RankedResult",
    "choose_threshold",
    "compare_ranks",
    "first_right_rank",
    "judge_report",
    "load_gold",
    "nearest_rank_ms",
    "outcome",
    "parse_gold_row",
    "retrieval_report",
    "threshold_sweep",
]
