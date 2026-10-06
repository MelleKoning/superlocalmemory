# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Measured precision and recall of the source check, per finding.

Only ``number_became_date`` takes a fact out of answers, so it alone must be
exact on the labelled set: precision 1.0 and recall 1.0. The other findings
only mark a result unverified; their floors are the measured values, so a
change that makes them worse fails here. Run with ``-s`` to print the table.
"""

from __future__ import annotations

from superlocalmemory.core.source_fidelity_guard import WITHHOLD_REASONS
from superlocalmemory.encoding.source_fidelity import check_fact_against_source
from tests.test_encoding._source_fidelity_cases import DAMAGED, FAITHFUL, REASONS

#: Floors, set from the measurement in this file (4.1.22). Precision, recall.
FLOORS: dict[str, tuple[float, float]] = {
    "number_became_date": (1.0, 1.0),   # 15 tp, 0 fp, 0 fn
    "unsupported_date": (1.0, 1.0),     # 10 tp, 0 fp, 0 fn
    "unsupported_number": (1.0, 0.9),   # 9 tp, 0 fp, 1 fn ("3.0 s" -> "30 s")
    "negation_lost": (0.8, 1.0),        # 12 tp, 3 fp ("not available" -> "unavailable")
    "order_reversed": (1.0, 0.5),       # 5 tp, 0 fp, 5 fn (a verb next to "before")
    "status_lost": (1.0, 0.57),         # 4 tp, 0 fp, 3 fn
}
#: Faithful facts the check flags (all negation_lost), of len(FAITHFUL).
MAX_FAITHFUL_FLAGGED = 3


def _labelled() -> list[tuple[str, str, str | None]]:
    return [(s, f, None) for s, f in FAITHFUL] + list(DAMAGED)


def confusion() -> dict[str, dict[str, int]]:
    table = {r: {"tp": 0, "fp": 0, "fn": 0} for r in REASONS}
    table["any_flag_on_faithful"] = {"tp": 0, "fp": 0, "fn": 0}
    for source, fact, gold in _labelled():
        predicted = set(check_fact_against_source(fact, source).reasons)
        if gold is None and predicted:
            table["any_flag_on_faithful"]["fp"] += 1
        for reason in REASONS:
            if reason in predicted and reason == gold:
                table[reason]["tp"] += 1
            elif reason in predicted:
                table[reason]["fp"] += 1
            elif reason == gold:
                table[reason]["fn"] += 1
    return table


def _ratio(num: int, den: int) -> float:
    return num / den if den else 1.0


def test_the_labelled_set_is_large_and_balanced() -> None:
    assert len(FAITHFUL) + len(DAMAGED) >= 120
    assert len(FAITHFUL) >= 55
    for reason in REASONS:
        assert sum(1 for *_, gold in DAMAGED if gold == reason) >= 7, reason


def test_withholding_is_exact_on_the_labelled_set() -> None:
    table = confusion()
    for reason in WITHHOLD_REASONS:
        row = table[reason]
        assert row["fp"] == 0, f"{reason} would withhold a fact it should not: {row}"
        assert row["fn"] == 0, f"{reason} missed a number turned into a date: {row}"


def test_each_finding_meets_its_floor() -> None:
    table = confusion()
    lines = [f"{'finding':22} tp  fp  fn  precision  recall"]
    for reason in REASONS:
        row = table[reason]
        precision = _ratio(row["tp"], row["tp"] + row["fp"])
        recall = _ratio(row["tp"], row["tp"] + row["fn"])
        lines.append(f"{reason:22} {row['tp']:2d}  {row['fp']:2d}  {row['fn']:2d}"
                     f"  {precision:9.2f}  {recall:6.2f}")
        floor_p, floor_r = FLOORS[reason]
        assert precision >= floor_p, (reason, row)
        assert recall >= floor_r, (reason, row)
    faithful_flags = table["any_flag_on_faithful"]["fp"]
    lines.append(f"faithful facts flagged: {faithful_flags} of {len(FAITHFUL)}")
    print("\n" + "\n".join(lines))
    assert faithful_flags <= MAX_FAITHFUL_FLAGGED
