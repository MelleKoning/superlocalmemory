# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The answer-check stage of one recall: asked at most once, inside the recall's
time budget, and always reported.

The check is a signal about the results, never part of retrieving them, so
every way this stage can end — judged, skipped, busy, warming, unavailable,
off — leaves the results exactly as retrieval and ranking produced them, apart
from the one opt-in exception: the hosted check's reorder, which the person
switched on and which this stage applies before anything downstream reads the
order.

Who may ask (``retrieval.answer_check_status`` has the full rules):

* Recalls that are not questions run inside ``skip_answer_check()``
  (``core.answer_check_scope``): the daemon warm-up, context loading, the
  per-prompt hook. Nothing is judged, sent or billed. Best-effort background
  work (health probe, materialiser) is skipped too, as before.
* ``REQUEST_NO_REORDER`` — the bounded-loop gate: it needs the verdict, not
  the reorder.
* ``REQUEST_FULL`` — every other recall a person or an agent asked for.

The judge is read off the engine once: a switch may replace it at any moment,
and one read means this recall uses one judge from start to end.
"""

from __future__ import annotations

import logging
from typing import Any

from superlocalmemory.core.answer_check_scope import answer_check_skipped
from superlocalmemory.retrieval.answer_check_status import (
    ANSWER_CHECK_STATUSES,
    REQUEST_FULL,
    STATUS_JUDGED,
    STATUS_OFF,
    STATUS_SKIPPED,
    STATUS_UNAVAILABLE,
    JudgeOutcome,
    judge_deadline,
)

logger = logging.getLogger(__name__)

_JUDGE_ATTR = "_sufficiency_judge"


def run_answer_check(retrieval_engine: Any, query: str, response: Any, *,
                     request: str = REQUEST_FULL,
                     recall_started: float | None = None,
                     profile_id: str | None = None) -> JudgeOutcome:
    """Ask the engine's answer check about ``response``, once, and say what happened.

    ``recall_started`` is ``time.monotonic()`` at the start of the recall. When
    given, the check must answer inside what is left of the recall ceiling, and
    is not asked at all when less than the floor is left. Skipping it costs the
    verdict, never a result. Without it (direct callers), each backend's own
    timeout applies, as before.

    ``profile_id`` is the profile the recall runs for. The online check is not
    asked when what it would read includes another profile's memory (a shared
    or global result): consent to send is this install's, but in a team that
    memory may be someone else's.
    """
    from superlocalmemory.core.recall_gate import is_background_work

    judge = getattr(retrieval_engine, _JUDGE_ATTR, None)
    if judge is None:
        return JudgeOutcome(None, STATUS_OFF)
    if answer_check_skipped() or is_background_work() or not response.results:
        return JudgeOutcome(None, STATUS_SKIPPED)
    if _would_send_another_profiles_memory(judge, response, profile_id, request):
        logger.debug("Answer check skipped: the online check would read another "
                     "profile's memory")
        return JudgeOutcome(None, STATUS_SKIPPED)
    deadline = None
    if recall_started is not None:
        deadline = judge_deadline(recall_started)
        if deadline is None:
            logger.debug("Answer check skipped: too little of the recall budget left")
            return JudgeOutcome(None, STATUS_SKIPPED)
    if request == REQUEST_FULL and reorders(judge):
        return rerank_and_judge(judge, query, response, deadline)
    return _plain_check(judge, query, response, deadline)


def _would_send_another_profiles_memory(judge: Any, response: Any,
                                       profile_id: str | None, request: str) -> bool:
    """Whether the online check would read a memory another profile owns.

    Only the hosted backend sends anything; only the memories it would read
    count — the top three, or the top ``rerank_k`` when it reorders.
    """
    if profile_id is None or getattr(judge, "backend", None) != "jev":
        return False
    if request == REQUEST_FULL and reorders(judge):
        span = getattr(judge, "rerank_k", 0)
    else:
        span = getattr(judge, "top_k", 3)
    if not isinstance(span, int) or isinstance(span, bool) or span < 1:
        span = len(response.results)
    return any(getattr(getattr(r, "fact", None), "profile_id", profile_id) != profile_id
               for r in response.results[:span])


def reorders(judge: Any) -> bool:
    """Only the hosted check, and only when it was built to reorder.

    ``is True`` and the backend check are deliberate: a test double, or a
    local judge that happens to grow the same attribute, never reorders.
    """
    return (getattr(judge, "backend", None) == "jev"
            and getattr(judge, "rerank_enabled", False) is True
            and callable(getattr(judge, "rerank_and_judge", None)))


def _genuine(verdict: Any) -> Any:
    from superlocalmemory.retrieval.sufficiency import SufficiencyVerdict

    return verdict if isinstance(verdict, SufficiencyVerdict) else None


def _settled(verdict: Any, status: Any) -> JudgeOutcome:
    """Only a genuine verdict counts, and only a known status is reported.

    A judge that misbehaves can never feed the contract something it did not
    decide, nor put an unknown word on the wire.
    """
    verdict = _genuine(verdict)
    if verdict is not None:
        return JudgeOutcome(verdict, STATUS_JUDGED)
    if status not in ANSWER_CHECK_STATUSES or status == STATUS_JUDGED:
        status = STATUS_UNAVAILABLE
    return JudgeOutcome(None, status)


def _plain_check(judge: Any, query: str, response: Any,
                 deadline: float | None) -> JudgeOutcome:
    from superlocalmemory.retrieval.judge_recipe import document_from_fact
    from superlocalmemory.retrieval.sufficiency import DEFAULT_TOP_K

    top_k = getattr(judge, "top_k", DEFAULT_TOP_K)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        top_k = DEFAULT_TOP_K
    try:
        documents = [document_from_fact(r.fact) for r in response.results[:top_k]]
        assess = getattr(judge, "assess", None)
        if callable(assess):
            outcome = assess(query, documents, deadline=deadline)
            if not isinstance(outcome, JudgeOutcome):
                return JudgeOutcome(None, STATUS_UNAVAILABLE)
            return _settled(outcome.verdict, outcome.status)
        return _settled(judge.judge(query, documents), STATUS_UNAVAILABLE)
    except Exception as exc:  # noqa: BLE001 — a judge never breaks a recall
        # The type only: a message could quote the memory text it was judging.
        logger.warning("Sufficiency judge failed; reporting the recall unjudged (%s)",
                       type(exc).__name__)
        return JudgeOutcome(None, STATUS_UNAVAILABLE)


def rerank_and_judge(judge: Any, query: str, response: Any,
                     deadline: float | None) -> JudgeOutcome:
    """Let the hosted check reorder the top results, then judge the new top three.

    One request (``JevSufficiencyJudge.rerank_and_judge``). The new order is
    applied here, before the score contract assigns ``rank_position`` and
    before markers and working memory see the list, so everything downstream
    describes the order the caller receives. Every failure leaves the results
    exactly as they were — the same objects, in the same order.

    Precedence: this runs after every other ordering pass, including the
    exact-lexical guard. Someone who turned it on asked Jev, which reads each
    memory, to choose the order of the top ``rerank_k``; a memory containing
    the question's words is one of the memories it reads.
    """
    from superlocalmemory.retrieval.jev_rerank import STATUS_LISTWISE, RerankVerdict
    from superlocalmemory.retrieval.judge_recipe import document_from_fact

    rerank_k = getattr(judge, "rerank_k", 0)
    if isinstance(rerank_k, bool) or not isinstance(rerank_k, int) or rerank_k < 1:
        return JudgeOutcome(None, STATUS_UNAVAILABLE)
    try:
        documents = [document_from_fact(r.fact) for r in response.results[:rerank_k]]
        if deadline is None:
            outcome = judge.rerank_and_judge(query, documents)
        else:
            outcome = judge.rerank_and_judge(query, documents, deadline=deadline)
    except Exception as exc:  # noqa: BLE001 — a judge never breaks a recall
        logger.warning("Answer check could not reorder; reporting the recall as it was (%s)",
                       type(exc).__name__)
        return JudgeOutcome(None, STATUS_UNAVAILABLE)
    if not isinstance(outcome, RerankVerdict):
        return JudgeOutcome(None, STATUS_UNAVAILABLE)
    if outcome.order is not None:
        if not apply_order(response, outcome.order):
            # The verdict describes a top three nobody will see.
            logger.warning("Answer check returned an unusable order; ignoring it")
            return JudgeOutcome(None, STATUS_UNAVAILABLE)
        response.local_reranker_status = response.reranker_status
        response.reranker_applied = True
        response.reranker_status = STATUS_LISTWISE
    return _settled(outcome.verdict, getattr(outcome, "status", "") or STATUS_UNAVAILABLE)


def apply_order(response: Any, order: tuple[int, ...]) -> bool:
    """Reorder the first ``len(order)`` results; the rest keep their places."""
    from superlocalmemory.retrieval.jev_rerank import is_permutation

    results = list(response.results)
    if not is_permutation(order, len(results)):
        return False
    block = results[: len(order)]
    response.results = [block[i] for i in order] + results[len(order):]
    return True


__all__ = ["apply_order", "reorders", "rerank_and_judge", "run_answer_check"]
