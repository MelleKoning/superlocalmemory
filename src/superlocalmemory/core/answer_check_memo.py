# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The same question over the same memories gets the same verdict (4.1.20).

Asking the on-device check twice about identical input is asking a
deterministic model the same thing twice. Before this, whether the second
recall got an answer depended on how long retrieval happened to take on a busy
machine: one run said "judged", the next "skipped", for one question and one
set of results. That is a repeatability defect, not a timing detail.

So the on-device check's verdict is remembered, per judge, keyed by everything
it read: the question, the exact text of the memories it judged (in order), and
the judge's threshold, calibration and top-k. A recall whose inputs match
reuses it — inside or outside the time budget — and says so
(``DETAIL_REUSED``). Change one memory's text, its position, the question or
the judge, and the key changes.

And when a recall has to skip the check because retrieval used its budget, the
check is finished after the recall returns, on a worker that is free right now
(``assess_if_idle``), so the next run of that question is judged.

Scope, on purpose:

* On-device (Laya) only. The hosted check re-reads consent and its key before
  every request and bills per request; a remembered verdict would answer after
  consent was withdrawn, and a finished-later check would bill for a verdict
  nobody asked to wait for.
* One judge, one memo. The memo hangs off the judge object, so a switch of
  provider or settings starts empty and a verdict from one provider can never
  answer for another.
* In memory only. Nothing is written; memory text is never kept — only a hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

#: Distinct (question, memories) pairs remembered per judge. A verdict is a few
#: floats; 512 covers every question one person asks in a working session.
MAX_ENTRIES = 512
#: A verdict is reused for this long. The judge is deterministic, so this is
#: hygiene (bounded staleness if anything outside the key changes), not accuracy.
TTL_S = 900.0

_MEMO_BACKENDS = frozenset({"laya"})


class _Memo:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.entries: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self.filling = False


_memos: weakref.WeakKeyDictionary[Any, _Memo] = weakref.WeakKeyDictionary()
_memos_lock = threading.Lock()


def remembers(judge: Any) -> bool:
    """Whether this judge's verdicts are remembered (the on-device check only)."""
    return getattr(judge, "backend", None) in _MEMO_BACKENDS


def _memo_for(judge: Any) -> _Memo | None:
    with _memos_lock:
        try:
            memo = _memos.get(judge)
            if memo is None:
                memo = _Memo()
                _memos[judge] = memo
            return memo
        except TypeError:  # a judge that cannot be weakly referenced: no memo
            return None


def _text(document: Any) -> str:
    content = getattr(document, "content", document)
    return content if isinstance(content, str) else repr(content)


def key_for(judge: Any, query: str, documents: Sequence[Any]) -> str:
    """Everything the verdict depends on, hashed. Memory text is not kept."""
    material = json.dumps([
        query,
        [_text(d) for d in documents],
        repr(getattr(judge, "threshold", None)),
        repr(getattr(judge, "calibration_id", None)),
        repr(getattr(judge, "top_k", None)),
    ], ensure_ascii=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def lookup(judge: Any, query: str, documents: Sequence[Any], *,
           now: float | None = None) -> Any:
    """The remembered verdict for exactly this input, or None."""
    memo = _memo_for(judge) if remembers(judge) else None
    if memo is None:
        return None
    key = key_for(judge, query, documents)
    current = time.monotonic() if now is None else now
    with memo.lock:
        hit = memo.entries.get(key)
        if hit is None:
            return None
        stored_at, verdict = hit
        if current - stored_at > TTL_S:
            del memo.entries[key]
            return None
        memo.entries.move_to_end(key)
        return verdict


def store(judge: Any, query: str, documents: Sequence[Any], verdict: Any, *,
          now: float | None = None) -> None:
    """Remember a genuine verdict. Callers pass only verdicts the judge gave."""
    memo = _memo_for(judge) if remembers(judge) else None
    if memo is None or verdict is None:
        return
    key = key_for(judge, query, documents)
    current = time.monotonic() if now is None else now
    with memo.lock:
        memo.entries[key] = (current, verdict)
        memo.entries.move_to_end(key)
        while len(memo.entries) > MAX_ENTRIES:
            memo.entries.popitem(last=False)


def finish_later(judge: Any, query: str, documents: Sequence[Any]) -> threading.Thread | None:
    """Judge after the recall returned, so the next identical recall is judged.

    One at a time per judge; only on a warm worker that is free right now
    (``assess_if_idle``). Returns the thread (tests join it), or None when
    nothing was started.
    """
    assess = getattr(judge, "assess_if_idle", None)
    memo = _memo_for(judge) if remembers(judge) and callable(assess) else None
    if memo is None or not getattr(judge, "ready", False) or getattr(judge, "closed", False):
        return None
    with memo.lock:
        if memo.filling:
            return None
        memo.filling = True
    snapshot = list(documents)

    def _run() -> None:
        from superlocalmemory.core.answer_check_stage import genuine_verdict
        try:
            outcome = assess(query, snapshot)
            verdict = genuine_verdict(getattr(outcome, "verdict", None))
            if verdict is not None:
                store(judge, query, snapshot, verdict)
        except Exception as exc:  # noqa: BLE001 — best effort, never surfaces
            logger.debug("answer check finished later: failed (%s)", type(exc).__name__)
        finally:
            with memo.lock:
                memo.filling = False

    thread = threading.Thread(target=_run, daemon=True, name="answer-check-finish-later")
    thread.start()
    return thread


def clear(judge: Any) -> None:
    """Forget every verdict this judge gave (tests; a judge being replaced)."""
    with _memos_lock:
        try:
            _memos.pop(judge, None)
        except TypeError:
            pass


__all__ = ["MAX_ENTRIES", "TTL_S", "clear", "finish_later", "key_for", "lookup",
           "remembers", "store"]
