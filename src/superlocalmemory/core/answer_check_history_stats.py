# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Numbers for the Answer Check tab, from recorded checks. Pure — no I/O.

The abstention rate is the one figure a reader will quote, so it is defined
narrowly and shown honestly:

* Denominator = recalls the check actually judged. "Nothing found" and every
  kind of "not checked" are separate counts and never enter it — otherwise a
  judge that was switched off would read as a judge that abstained.
* Below ``MIN_SAMPLE`` judged recalls the rate is withheld (``None``): a rate
  over a handful of checks is noise that looks like a fact.
* When it is shown, it carries a Wilson score 95% interval (Wilson 1927),
  which stays inside [0, 1] and is honest at small n, unlike p ± 1.96·se.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

#: Fewest judged checks before an abstention rate is shown at all.
MIN_SAMPLE = 20
#: The windows the tab offers, in milliseconds.
WINDOWS = {"24h": 86_400_000, "7d": 604_800_000, "30d": 2_592_000_000}
_Z95 = 1.959963985

OUTCOME_KEYS = (
    "answered", "abstained", "nothing_found", "not_checked_off", "not_checked_time",
    "not_checked_shared", "not_checked_busy", "not_checked_loading",
    "not_checked_unavailable",
)
_NOT_CHECKED_BUCKET = {
    "not_checked_off": "off", "not_checked_time": "time",
    "not_checked_shared": "shared", "not_checked_busy": "busy",
    "not_checked_loading": "loading", "not_checked_unavailable": "unavailable",
}
_SKIP_DETAIL = {
    "no_results": "nothing_found",
    "budget": "not_checked_time",
    "other_profile_memory": "not_checked_shared",
}
_STATUS_OUTCOME = {
    "off": "not_checked_off",
    "busy": "not_checked_busy",
    "warming": "not_checked_loading",
    "unavailable": "not_checked_unavailable",
}


def outcome_key(status: str, detail: str, abstained: bool, result_count: int) -> str:
    """The one word the tab uses for what happened on a recall."""
    if status == "judged":
        return "abstained" if abstained else "answered"
    if status == "skipped":
        if detail in _SKIP_DETAIL:
            return _SKIP_DETAIL[detail]
        return "nothing_found" if int(result_count or 0) == 0 else "not_checked_unavailable"
    return _STATUS_OUTCOME.get(status, "not_checked_unavailable")


def nearest_rank(sorted_vals: Sequence[float], q: float) -> float | None:
    """The nearest-rank percentile: the ceil(q·n)-th smallest value."""
    n = len(sorted_vals)
    if n == 0:
        return None
    rank = max(1, min(n, math.ceil(q * n)))
    return float(sorted_vals[rank - 1])


def wilson_interval(k: int, n: int, z: float = _Z95) -> tuple[float, float] | None:
    """Wilson score interval for k successes in n trials; None when n == 0."""
    if n <= 0:
        return None
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _abstention(k: int, n: int) -> dict[str, Any]:
    enough = n >= MIN_SAMPLE
    ci = wilson_interval(k, n) if enough else None
    return {
        "k": k, "n": n,
        "rate": round(k / n, 4) if enough else None,
        "ci95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
        "min_sample": MIN_SAMPLE, "enough": enough,
    }


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _latency(rows: list[Mapping], ceiling_ms: float, recent_n: int) -> dict[str, Any]:
    totals = [(r.get("occurred_ms") or 0, _finite(r.get("total_ms"))) for r in rows]
    totals = [(at, t) for at, t in totals if t is not None]
    total_sorted = sorted(t for _, t in totals)
    judge_sorted = sorted(
        j for j in (_finite(r.get("judge_ms")) for r in rows if r.get("status") == "judged")
        if j is not None)
    recent = [t for _, t in sorted(totals, key=lambda pair: pair[0])][-recent_n:]
    return {
        "ceiling_ms": ceiling_ms,
        "over_ceiling": sum(1 for t in total_sorted if t > ceiling_ms),
        "total": {"n": len(total_sorted), "p50": nearest_rank(total_sorted, 0.50),
                  "p95": nearest_rank(total_sorted, 0.95),
                  "max": total_sorted[-1] if total_sorted else None},
        "judge": {"n": len(judge_sorted), "p50": nearest_rank(judge_sorted, 0.50),
                  "p95": nearest_rank(judge_sorted, 0.95)},
        "recent_total_ms": recent,
    }


def summarize(rows: Iterable[Mapping], *, ceiling_ms: float,
              recent_n: int = 60) -> dict[str, Any]:
    """Counts, abstention rate and latency over ``rows`` (already filtered)."""
    rows = list(rows)
    counts = {key: 0 for key in OUTCOME_KEYS}
    by_judge = {"laya": 0, "jev": 0}
    for row in rows:
        key = outcome_key(str(row.get("status") or ""), str(row.get("detail") or ""),
                          bool(row.get("abstained")), int(row.get("result_count") or 0))
        counts[key] += 1
        if row.get("status") == "judged" and row.get("backend") in by_judge:
            by_judge[row["backend"]] += 1
    judged = counts["answered"] + counts["abstained"]
    return {
        "counts": {
            "total": len(rows), "answered": counts["answered"],
            "abstained": counts["abstained"], "nothing_found": counts["nothing_found"],
            "not_checked": {bucket: counts[key] for key, bucket in _NOT_CHECKED_BUCKET.items()},
        },
        "by_judge": by_judge,
        "abstention": _abstention(counts["abstained"], judged),
        "latency": _latency(rows, ceiling_ms, recent_n),
    }


__all__ = ["MIN_SAMPLE", "OUTCOME_KEYS", "WINDOWS", "nearest_rank", "outcome_key",
           "summarize", "wilson_interval"]
