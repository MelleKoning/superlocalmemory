# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Truthful public retrieval score contract.

Retrieval produces an ordering, not an answer probability.  This module keeps
query relevance, stored assertion confidence, trust, and internal ranking
utility distinct while preserving the V3.6 aliases for one compatibility
release.
"""

from __future__ import annotations

import math

from superlocalmemory.retrieval.sufficiency import SufficiencyVerdict
from superlocalmemory.storage.models import RecallResponse


def _bounded(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    return min(1.0, max(0.0, number))


def finalize_score_contract(
    response: RecallResponse, verdict: SufficiencyVerdict | None = None,
) -> RecallResponse:
    """Finalize aliases, rank positions, and response abstention metadata.

    ``score`` IS NOT THE ORDERING KEY, and a caller that sorts by it will get a
    different answer than the one this returns. That is deliberate, and it is
    load-bearing:

    * ``score`` answers "how well does this match the query" — a property of the
      result and the query alone.
    * ``rank_position``, assigned here from the order of the list, answers "what
      is recommended, and in what order". Ranking also weighs whether a memory
      has helped before and whether the session was just looking at it, neither
      of which is a property of the query.

    So the two can disagree, and the top result can legitimately carry a lower
    ``score`` than the one beneath it. Consumers should present the list in the
    order given, or sort by ``rank_position``; ``ranking_score`` carries the
    internal utility for anyone who needs to re-derive it.

    This ordering is NOT recomputed here. Re-sorting by ``score`` at this point
    would silently undo every ranking pass that ran before it.
    """
    for position, result in enumerate(response.results or (), start=1):
        relevance = _bounded(
            result.relevance_score
            if result.relevance_score is not None
            else result.score
        )
        memory_confidence = _bounded(
            result.memory_confidence
            if result.memory_confidence is not None
            else getattr(result.fact, "confidence", 0.0)
        )
        result.relevance_score = relevance
        result.score = relevance
        result.memory_confidence = memory_confidence
        result.confidence = memory_confidence
        result.rank_position = position

    response.score_contract_version = "2"
    if verdict is None or not response.results:
        response.calibration_status = "uncalibrated"
        response.calibration_id = None
        response.answer_confidence = None
        response.abstained = not bool(response.results)
    else:
        # 4.1.18: a separate decision on whether the results answer the
        # question. The ranking score cannot make it — measured on a real store
        # it was the same for right answers, wrong ones and questions with no
        # answer in memory. Results are never removed: abstention is a signal
        # for the caller, and a caller that ignores it sees what it saw before.
        response.calibration_status = verdict.calibration_status
        response.calibration_id = verdict.calibration_id
        response.answer_confidence = round(verdict.answer_confidence, 4)
        response.abstained = verdict.insufficient
    if response.abstained:
        if response.results:
            response.abstention_reason = "judged_insufficient"
        else:
            response.abstention_reason = (
                "evidence_floor" if response.no_confident_match else "no_candidates"
            )
    else:
        response.abstention_reason = None
    return response


__all__ = ["finalize_score_contract"]
