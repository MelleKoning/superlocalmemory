# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Answer checks finished after their recall returned — never at a recall's cost.

A recall that used its time budget is reported "skipped"; its question is
queued here (``answer_check_memo.finish_later``) so the next run of it is
judged instead of skipped again. The on-device worker is shared with live
recalls, and a live recall always comes first:

* Nothing is asked while any recall is in flight (``recall_gate``) or asking
  the check (``answer_check_memo.live_check``). The queue waits for a quiet
  moment, for at most ``WAIT_FOR_IDLE_S``, and is given up after that.
* The memories are judged ONE AT A TIME — the worker scores each memory on its
  own, so the verdict is the same as one request for all of them — and the
  quiet is checked again before each. A recall that arrives meanwhile waits for
  at most one memory's judgement, not a whole check (before 4.1.20 it waited
  for all of it, and a second question then came back "busy").
* Each memory's verdict is kept until the set is complete, so an interrupted
  check continues where it stopped at the next quiet moment.
* Only a worker that is already warm and free right now is used
  (``assess_if_idle``); nothing here ever loads a model or waits on the
  worker's lock.

One thread per judge at most, alive only while its queue is not empty.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from superlocalmemory.core import recall_gate

logger = logging.getLogger(__name__)

#: How long a queued question waits for a quiet moment before it is dropped.
WAIT_FOR_IDLE_S = 60.0
#: How often the quiet is looked at while waiting for it.
POLL_S = 0.02

THREAD_NAME = "answer-check-finish-later"


def start_worker(judge: Any, memo: Any) -> threading.Thread:
    """Start the one thread that empties ``memo.pending``. Caller holds ``memo.lock``."""
    thread = threading.Thread(target=_drain, args=(judge, memo), daemon=True,
                              name=THREAD_NAME)
    thread.start()
    return thread


def _drain(judge: Any, memo: Any) -> None:
    while True:
        closed = getattr(judge, "closed", False) is True
        with memo.lock:
            if closed:
                memo.pending.clear()  # a stopped judge answers nothing more
            if not memo.pending:
                memo.worker = None
                return
            key, (query, documents, by) = memo.pending.popitem(last=False)
        try:
            _finish(judge, memo, key, query, documents, by)
        except Exception as exc:  # noqa: BLE001 — best effort, never surfaces
            logger.debug("answer check finished later: failed (%s)", type(exc).__name__)


def _quiet(judge: Any, memo: Any, by: float) -> bool:
    """Wait until no recall is in flight or asking the check. False at ``by``."""
    while True:
        if getattr(judge, "closed", False) is True or not getattr(judge, "ready", False):
            return False
        with memo.lock:
            live = memo.live
        if live == 0 and recall_gate.in_flight() == 0:
            return True
        if time.monotonic() >= by:
            return False
        time.sleep(POLL_S)


def _finish(judge: Any, memo: Any, key: str, query: str,
            documents: tuple[Any, ...], by: float) -> None:
    from superlocalmemory.core import answer_check_memo
    from superlocalmemory.retrieval.answer_check_status import STATUS_BUSY

    with memo.lock:
        if answer_check_memo._fresh(memo.entries, key, time.monotonic()) is not None:
            return
        found = answer_check_memo._fresh(memo.partials, key, time.monotonic())
    resumable = found is not None and len(found) == len(documents)
    parts: list[Any] = list(found) if resumable else [None] * len(documents)
    for index, document in enumerate(documents):
        while parts[index] is None:
            if not _quiet(judge, memo, by):
                _keep(memo, key, parts)
                return
            outcome = judge.assess_if_idle(query, [document])
            if getattr(outcome, "status", None) == STATUS_BUSY:
                time.sleep(POLL_S)  # a recall (or memory typing) has the worker
                continue
            verdict = _single(outcome)
            if verdict is None:
                _keep(memo, key, parts)
                return
            parts[index] = verdict
        if answer_check_memo.lookup(judge, query, documents) is not None:
            return  # a live recall judged this question meanwhile
    verdict = _combine(parts)
    if verdict is not None:
        answer_check_memo.store(judge, query, documents, verdict)


def _single(outcome: Any) -> Any:
    """A genuine one-memory verdict, or None."""
    from superlocalmemory.core.answer_check_stage import genuine_verdict

    verdict = genuine_verdict(getattr(outcome, "verdict", None))
    if verdict is None or len(verdict.probabilities) != 1:
        return None
    return verdict


def _combine(parts: list[Any]) -> Any:
    """One verdict over every memory, exactly as one request would have given it."""
    from superlocalmemory.retrieval.sufficiency import SufficiencyVerdict

    if not parts or any(p is None for p in parts):
        return None
    first = parts[0]
    same = {(p.threshold, p.calibration_id, p.calibration_status, p.backend) for p in parts}
    if len(same) != 1:
        return None  # the judge changed between memories: nothing coherent to keep
    return SufficiencyVerdict(tuple(p.probabilities[0] for p in parts), first.threshold,
                              first.calibration_id, first.calibration_status, first.backend)


def _keep(memo: Any, key: str, parts: list[Any]) -> None:
    from superlocalmemory.core import answer_check_memo

    if any(p is not None for p in parts):
        with memo.lock:
            answer_check_memo._put(memo.partials, key, tuple(parts), time.monotonic())


__all__ = ["POLL_S", "THREAD_NAME", "WAIT_FOR_IDLE_S", "start_worker"]
