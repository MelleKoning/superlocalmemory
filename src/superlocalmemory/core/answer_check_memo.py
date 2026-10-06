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
check is finished after the recall returns (``finish_later``, carried out by
``answer_check_deferred``), so the next run of that question is judged. A live
recall always comes first: that work runs only while no recall is in flight or
waiting for the check, one memory at a time, and stops as soon as one arrives.

Scope, on purpose:

* On-device (Laya) only. The hosted check re-reads consent and its key before
  every request and bills per request; a remembered verdict would answer after
  consent was withdrawn, and a finished-later check would bill for a verdict
  nobody asked to wait for.
* One judge, one memo. The memo hangs off the judge object, so a switch of
  provider or settings starts empty and a verdict from one provider can never
  answer for another. A judge that has been shut down answers nothing from its
  memo, and stopping it forgets the memo outright (``clear``).
* In memory only. Nothing is written; memory text is never kept — only a hash.

4.1.22 — what a verdict is bound to (``Binding``). Besides the question, the
memories' text and order, and the judge's threshold, calibration and top-k, the
key holds the profile the recall ran for, the ids of the memories judged and
their kinds, and the judge's backend, model, recipe and configuration epoch. A
verdict also remembers the store's change-log position when it was given
(``storage/fact_search_changes``): if any memory it judged has since been
erased, withheld, re-scoped, re-embedded or re-kinded, the verdict is dropped
instead of reused. A reused verdict says so (``DETAIL_REUSED``,
``answerability_reason = judged_from_memo``); a fresh one never does.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: Distinct (question, memories) pairs remembered per judge. A verdict is a few
#: floats; 512 covers every question one person asks in a working session.
MAX_ENTRIES = 512
#: A verdict is reused for this long. The judge is deterministic, so this is
#: hygiene (bounded staleness if anything outside the key changes), not accuracy.
TTL_S = 900.0
#: Questions waiting to be finished later, per judge. The newest are kept.
MAX_PENDING = 16

_MEMO_BACKENDS = frozenset({"laya"})


@dataclass(frozen=True)
class Binding:
    """Who the recall was for and exactly which memories were judged."""

    profile_id: str = ""
    fact_ids: tuple[str, ...] = ()
    kinds: tuple[str, ...] = ()
    #: The change-log position when the memories were read (validation only).
    seq: int = 0


#: ``changed(seq, fact_ids)``: whether any of those memories changed since seq.
Changed = Callable[[int, tuple[str, ...]], bool]


class _Memo:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.entries: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        #: Per-memory verdicts of a check finished later that a recall
        #: interrupted, so the next idle moment carries on where it stopped.
        self.partials: OrderedDict[str, tuple[float, tuple[Any, ...]]] = OrderedDict()
        #: Questions waiting to be finished later: key -> (query, documents, by, binding).
        self.pending: OrderedDict[str, tuple[str, tuple[Any, ...], float]] = OrderedDict()
        self.worker: threading.Thread | None = None
        #: Live recalls asking this judge right now (or waiting for its worker).
        self.live = 0


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


def _usable(judge: Any) -> _Memo | None:
    """The judge's memo, or None when it has none or has been shut down."""
    if not remembers(judge) or getattr(judge, "closed", False) is True:
        return None
    return _memo_for(judge)


def _text(document: Any) -> str:
    content = getattr(document, "content", document)
    return content if isinstance(content, str) else repr(content)


def key_for(judge: Any, query: str, documents: Sequence[Any],
            binding: Binding | None = None) -> str:
    """Everything the verdict depends on, hashed. Memory text is not kept."""
    b = binding or Binding()
    material = json.dumps([
        query,
        [_text(d) for d in documents],
        repr(getattr(judge, "threshold", None)),
        repr(getattr(judge, "calibration_id", None)),
        repr(getattr(judge, "top_k", None)),
        [repr(getattr(judge, a, None)) for a in ("backend", "model", "recipe",
                                                  "config_epoch")],
        b.profile_id, list(b.fact_ids), list(b.kinds),
    ], ensure_ascii=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _fresh(table: OrderedDict, key: str, current: float) -> Any:
    """``table[key]``'s value if it has not expired. Caller holds the lock."""
    hit = table.get(key)
    if hit is None:
        return None
    if current - hit[0] > TTL_S:
        del table[key]
        return None
    table.move_to_end(key)
    return hit[1]


def _put(table: OrderedDict, key: str, value: Any, current: float) -> None:
    """Caller holds the lock."""
    table[key] = (current, value)
    table.move_to_end(key)
    while len(table) > MAX_ENTRIES:
        table.popitem(last=False)


def lookup(judge: Any, query: str, documents: Sequence[Any], *,
           now: float | None = None, binding: Binding | None = None,
           changed: Changed | None = None) -> Any:
    """The remembered verdict for exactly this input, or None.

    ``changed`` (when given) is asked whether a judged memory changed since the
    verdict was given; if so, or if it cannot say, the verdict is dropped.
    """
    memo = _usable(judge)
    if memo is None:
        return None
    key = key_for(judge, query, documents, binding)
    current = time.monotonic() if now is None else now
    with memo.lock:
        hit = _fresh(memo.entries, key, current)
    if hit is None:
        return None
    verdict, seq, ids = hit
    if changed is not None and ids:
        try:
            stale = bool(changed(seq, ids))
        except Exception:  # noqa: BLE001 -- cannot prove it current: do not reuse
            stale = True
        if stale:
            with memo.lock:
                memo.entries.pop(key, None)
            return None
    return verdict


def store(judge: Any, query: str, documents: Sequence[Any], verdict: Any, *,
          now: float | None = None, binding: Binding | None = None) -> None:
    """Remember a genuine verdict. Callers pass only verdicts the judge gave."""
    memo = _usable(judge)
    if memo is None or verdict is None:
        return
    b = binding or Binding()
    key = key_for(judge, query, documents, b)
    current = time.monotonic() if now is None else now
    with memo.lock:
        _put(memo.entries, key, (verdict, b.seq, b.fact_ids), current)
        memo.partials.pop(key, None)


@contextmanager
def live_check(judge: Any) -> Iterator[None]:
    """Mark a live recall asking ``judge``: work finished later yields to it."""
    memo = _usable(judge)
    if memo is None:
        yield
        return
    with memo.lock:
        memo.live += 1
    try:
        yield
    finally:
        with memo.lock:
            memo.live = max(0, memo.live - 1)


def live_checks(judge: Any) -> int:
    """How many live recalls are asking ``judge`` right now."""
    memo = _usable(judge)
    if memo is None:
        return 0
    with memo.lock:
        return memo.live


def finish_later(judge: Any, query: str, documents: Sequence[Any], *,
                 binding: Binding | None = None) -> threading.Thread | None:
    """Judge after the recall returned, so the next identical recall is judged.

    Queued per judge and carried out by one thread (``answer_check_deferred``),
    only on a warm worker, only while no recall is in flight or asking the
    check. Returns that thread (tests join it), or None when nothing was queued.
    """
    from superlocalmemory.core import answer_check_deferred

    if not callable(getattr(judge, "assess_if_idle", None)):
        return None
    memo = _usable(judge)
    if memo is None or not getattr(judge, "ready", False):
        return None
    snapshot = tuple(documents)
    if not snapshot:
        return None
    key = key_for(judge, query, snapshot, binding)
    by = time.monotonic() + answer_check_deferred.WAIT_FOR_IDLE_S
    with memo.lock:
        if _fresh(memo.entries, key, time.monotonic()) is not None:
            return None
        memo.pending[key] = (query, snapshot, by, binding)
        memo.pending.move_to_end(key)
        while len(memo.pending) > MAX_PENDING:
            memo.pending.popitem(last=False)
        if memo.worker is not None:
            return memo.worker
        memo.worker = answer_check_deferred.start_worker(judge, memo)
        return memo.worker


def clear(judge: Any) -> None:
    """Forget every verdict this judge gave (a judge being stopped or replaced)."""
    with _memos_lock:
        try:
            memo = _memos.pop(judge, None)
        except TypeError:
            memo = None
    if memo is not None:
        with memo.lock:
            memo.entries.clear()
            memo.partials.clear()
            memo.pending.clear()


__all__ = ["Binding", "MAX_ENTRIES", "MAX_PENDING", "TTL_S", "clear", "finish_later", "key_for",
           "live_check", "live_checks", "lookup", "remembers", "store"]
