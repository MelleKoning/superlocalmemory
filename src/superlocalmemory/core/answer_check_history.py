# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Answer Check history, the in-memory side: what the recall path touches.

One bounded ring buffer plays two roles (Ring Buffer + Write-Behind Cache):

* it is the in-memory store — the tab's live feed reads it with zero I/O;
* it is the write-behind queue — the daemon's writer thread
  (``answer_check_history_store``) saves every entry newer than ``_saved_seq``
  to learning.db, and is the only thing that ever writes (Single-Writer).

What the recall path pays: building one small frozen record from attributes
already on the response, then one append under one uncontended lock. This
module never imports ``sqlite3``, ``os``, ``io``, ``pathlib`` or a network
module, and never opens a file — tests pin that by reading its AST.

Nothing that could identify what was asked or found is ever recorded: no
question text, no memory text, no memory ids, no session or agent names. The
record is outcomes, enums, counts and timings only, each validated against a
closed set or a numeric range before it is kept.

A full ring drops its oldest entry (``deque(maxlen)``). If that entry had not
been saved yet, ``dropped_before_save`` counts it and the tab says so. Recall
is never slowed down to keep a history entry.

An erasure removes a profile's entries that are older than it. "Older" is
decided by what this process knew, not by the wall clock alone: an entry
recorded after this process learned of an erasure is never removed by that
erasure, even if the clock stepped back in between (``note`` in
``forget_profile``; the saved side applies the same rule in SQL). Every entry
an erasure removes before it was saved is counted in ``erased_unsaved``.

Recording is off unless a writer runs in this process (``enable``): the CLI's
offline fallback and worker subprocesses keep ``record_recall_verdict`` a no-op,
so learning.db never gets a second writer.
"""

from __future__ import annotations

import math
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from superlocalmemory.core.answer_check_scope import answer_check_skipped
from superlocalmemory.core.recall_gate import is_background_work
from superlocalmemory.retrieval.answer_check_status import (
    ANSWER_CHECK_DETAILS,
    ANSWER_CHECK_STATUSES,
    STATUS_UNAVAILABLE,
)

RING_CAPACITY = 2048
#: Wake the writer early once this many entries are waiting to be saved.
WAKE_AT_UNSAVED = 128

ORIGIN_DASHBOARD = "dashboard"
#: A saved view run, by the surface it was run from (4.1.21, #113). A view run
#: is a real question, so unlike a dashboard test it counts in the summaries.
ORIGIN_VIEW_DASHBOARD = "view-dashboard"
ORIGIN_VIEW_CLI = "view-cli"
ORIGIN_VIEW_MCP = "view-mcp"
VIEW_ORIGINS = frozenset({ORIGIN_VIEW_DASHBOARD, ORIGIN_VIEW_CLI, ORIGIN_VIEW_MCP})
_ORIGINS = frozenset({"", ORIGIN_DASHBOARD}) | VIEW_ORIGINS
_ORIGIN: ContextVar[str] = ContextVar("slm_answer_check_origin", default="")

_BACKENDS = frozenset({"", "laya", "jev"})
_REASONS = frozenset({None, "judged_insufficient", "evidence_floor", "no_candidates"})
_EVENT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_QUERY_TYPE_RE = re.compile(r"^[a-z_-]{0,32}$")
_CALIBRATION_RE = re.compile(r"^[A-Za-z0-9._:@/+-]{0,96}$")
_MAX_MS = 600_000.0
_MAX_RESULTS = 10_000


@dataclass(frozen=True, slots=True)
class VerdictEvent:
    """One recall's answer-check outcome. No text field, by construction."""

    event_id: str
    profile_id: str
    occurred_ms: int
    status: str
    detail: str
    backend: str
    origin: str
    abstained: bool
    abstention_reason: str | None
    answer_confidence: float | None
    threshold: float | None
    reordered: bool
    result_count: int
    query_type: str
    retrieval_ms: float | None
    judge_ms: float | None
    total_ms: float | None
    calibration_id: str
    #: Within ``retrieval_ms``: waiting for the query's embedding, and ranking.
    #: None when that stage did not run or was not measured.
    embed_ms: float | None = None
    rerank_ms: float | None = None


_lock = threading.Lock()
_ring: deque[tuple[int, VerdictEvent]] = deque(maxlen=RING_CAPACITY)
_seq = 0           # last sequence number handed out
_saved_seq = 0     # every seq <= this is saved (or deliberately discarded)
_inflight_seq = 0  # the writer has taken a snapshot through here
_enabled = False   # True only while a writer runs in THIS process
_boot_id = uuid.uuid4().hex
_wake = threading.Event()
_COUNTER_NAMES = ("recorded", "not_a_question", "dropped_before_save", "saved",
                  "save_failures", "erased_unsaved")
_counters = dict.fromkeys(_COUNTER_NAMES, 0)
#: Erasures this process has applied, per profile: (erased_at_ms, last seq
#: handed out when it was applied). Entries recorded later are not its to erase.
_erasures_seen: dict[str, tuple[tuple[int, int], ...]] = {}
_ERASURES_KEPT_PER_PROFILE = 4
_ERASURE_PROFILES_MAX = 4096


# -- who asked ------------------------------------------------------------------

@contextmanager
def origin(name: str) -> Iterator[None]:
    """Tag recalls made inside this block: a dashboard test, or a view run.

    Must be entered in the thread that runs the recall: context variables do
    not follow work handed to a plain thread pool.
    """
    if not name or name not in _ORIGINS:
        raise ValueError(f"unknown answer-check origin {name!r}")
    token = _ORIGIN.set(name)
    try:
        yield
    finally:
        _ORIGIN.reset(token)


def call_as_dashboard(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run ``fn`` with recalls tagged as dashboard tests (for executor threads)."""
    with origin(ORIGIN_DASHBOARD):
        return fn(*args, **kwargs)


# -- lifecycle (called by the store's start_writer / stop_writer) -----------------

def enable(on: bool) -> None:
    global _enabled
    _enabled = bool(on)


def is_enabled() -> bool:
    return _enabled


def boot_id() -> str:
    return _boot_id


def wake_event() -> threading.Event:
    return _wake


# -- building a record --------------------------------------------------------------

def _number(value: Any, low: float, high: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return float(min(high, max(low, value)))


def _member(value: Any, allowed: frozenset, default: Any) -> Any:
    try:
        return value if value in allowed else default
    except TypeError:  # unhashable
        return default


def _matching(value: Any, pattern: re.Pattern) -> str:
    return value if isinstance(value, str) and pattern.match(value) else ""


def _stage(response: Any, name: str) -> float | None:
    """One recall stage's milliseconds from ``RecallResponse.stage_ms``."""
    stages = getattr(response, "stage_ms", None)
    if not isinstance(stages, dict):
        return None
    return _number(stages.get(name), 0.0, _MAX_MS)


def event_from_response(response: Any, profile_id: str, *, now_ms: int,
                        origin_name: str) -> VerdictEvent | None:
    """The record for one recall, every field validated. Pure; never raises."""
    try:
        if not isinstance(profile_id, str) or not profile_id:
            return None
        trace = getattr(response, "answer_check_trace", None)
        results = getattr(response, "results", None) or ()
        event_id = getattr(response, "query_id", "")
        return VerdictEvent(
            event_id=event_id if _matching(event_id, _EVENT_ID_RE) else uuid.uuid4().hex,
            profile_id=profile_id,
            occurred_ms=int(now_ms),
            status=_member(getattr(response, "answer_check_status", None),
                           ANSWER_CHECK_STATUSES, STATUS_UNAVAILABLE),
            detail=_member(getattr(trace, "detail", ""), ANSWER_CHECK_DETAILS, ""),
            backend=_member(getattr(trace, "backend", ""), _BACKENDS, ""),
            origin=_member(origin_name, _ORIGINS, ""),
            abstained=getattr(response, "abstained", False) is True,
            abstention_reason=_member(getattr(response, "abstention_reason", None),
                                      _REASONS, None),
            answer_confidence=_number(getattr(response, "answer_confidence", None), 0.0, 1.0),
            threshold=_number(getattr(trace, "threshold", None), 0.0, 1.0),
            reordered=getattr(trace, "reordered", False) is True,
            result_count=min(_MAX_RESULTS, len(results)),
            query_type=_matching(getattr(response, "query_type", ""), _QUERY_TYPE_RE),
            retrieval_ms=_number(getattr(trace, "retrieval_ms", None), 0.0, _MAX_MS),
            judge_ms=_number(getattr(trace, "judge_ms", None), 0.0, _MAX_MS),
            total_ms=_number(getattr(trace, "total_ms", None), 0.0, _MAX_MS),
            calibration_id=_matching(getattr(response, "calibration_id", "") or "",
                                     _CALIBRATION_RE),
            embed_ms=_stage(response, "query_embedding"),
            rerank_ms=_stage(response, "rerank"),
        )
    except Exception:  # noqa: BLE001 — a malformed response is simply not recorded
        return None


# -- the recall path ---------------------------------------------------------------

def _bump(name: str, by: int = 1) -> None:
    with _lock:
        _counters[name] += by


def record_recall_verdict(response: Any, *, profile_id: str) -> None:
    """Record one recall's outcome. O(1), no I/O, never blocks on the writer."""
    global _seq
    if not _enabled:
        return
    if answer_check_skipped() or is_background_work():
        _bump("not_a_question")  # hooks, context loads, warm-up: not questions
        return
    event = event_from_response(response, profile_id, now_ms=int(time.time() * 1000),
                                origin_name=_ORIGIN.get())
    if event is None:
        return
    with _lock:
        _seq += 1
        if len(_ring) == RING_CAPACITY and _ring[0][0] > max(_saved_seq, _inflight_seq):
            _counters["dropped_before_save"] += 1
        _ring.append((_seq, event))
        _counters["recorded"] += 1
        wake = (_seq - _saved_seq) >= WAKE_AT_UNSAVED
    if wake:
        _wake.set()


# -- the writer side (writer thread only) ----------------------------------------

def _unsaved_suffix() -> list[tuple[int, VerdictEvent]]:
    """Unsaved entries, oldest first. Caller holds ``_lock``.

    The ring is ordered by seq, so the unsaved entries are a suffix of it:
    walk back from the newest end only as far as needed (O(unsaved), not
    O(RING_CAPACITY)) — the recall path waits on this lock.
    """
    out = []
    for entry in reversed(_ring):
        if entry[0] <= _saved_seq:
            break
        out.append(entry)
    out.reverse()
    return out


def snapshot_unsaved(max_items: int) -> list[tuple[int, VerdictEvent]]:
    """The oldest unsaved entries, at most ``max_items``; marks them in flight."""
    global _inflight_seq
    with _lock:
        batch = _unsaved_suffix()[:max_items]
        _inflight_seq = batch[-1][0] if batch else _saved_seq
    return batch


def mark_saved(through_seq: int, saved: int, *, erased: int = 0) -> None:
    """Entries through ``through_seq`` are done: ``saved`` were written, and
    ``erased`` were refused because their profile was erased after them."""
    global _saved_seq, _inflight_seq
    with _lock:
        _saved_seq = max(_saved_seq, int(through_seq))
        _inflight_seq = _saved_seq
        _counters["saved"] += int(saved)
        _counters["erased_unsaved"] += max(0, int(erased))


def mark_failed(batch: list[tuple[int, VerdictEvent]]) -> None:
    """A save failed: entries evicted while in flight are now truly lost."""
    global _inflight_seq
    with _lock:
        oldest = _ring[0][0] if _ring else _seq + 1
        _counters["dropped_before_save"] += sum(1 for seq, _ in batch if seq < oldest)
        _counters["save_failures"] += 1
        _inflight_seq = _saved_seq


# -- readers (no I/O) ----------------------------------------------------------------

def recent(profile_id: str, *, after_seq: int,
           limit: int) -> tuple[list[tuple[int, VerdictEvent]], int]:
    """Newest-first entries of one profile newer than ``after_seq``; and the last seq."""
    with _lock:
        snapshot = list(_ring)
        last = _seq
    items = [entry for entry in reversed(snapshot)
             if entry[0] > after_seq and entry[1].profile_id == profile_id]
    return items[:max(0, limit)], last


def unsaved_for(profile_id: str) -> list[VerdictEvent]:
    with _lock:
        return [ev for _, ev in _unsaved_suffix() if ev.profile_id == profile_id]


def _known_locked(profile_id: str, seq: int) -> int:
    """The latest erasure of ``profile_id`` applied before ``seq`` was handed
    out (0 if none). Caller holds ``_lock``."""
    return max((ms for ms, at in _erasures_seen.get(profile_id, ()) if at < seq), default=0)


def erasure_known_at(profile_id: str, seq: int) -> int:
    """The latest erasure this process had applied when entry ``seq`` was recorded.

    An erasure at or before this instant does not apply to that entry: the
    entry was recorded after it, whatever its wall-clock time says.
    """
    with _lock:
        return _known_locked(profile_id, seq)


def _note_locked(profile_id: str, erased_at_ms: int) -> None:
    notes = _erasures_seen.pop(profile_id, ())
    _erasures_seen[profile_id] = (notes + ((int(erased_at_ms), _seq),))[
        -_ERASURES_KEPT_PER_PROFILE:]
    while len(_erasures_seen) > _ERASURE_PROFILES_MAX:
        _erasures_seen.pop(next(iter(_erasures_seen)))


def forget_profile(profile_id: str, *, occurred_before_ms: int | None = None,
                   erased_at_ms: int | None = None) -> int:
    """Drop a profile's entries from the ring (all, or those at/before a cutoff).

    With a cutoff, an entry recorded after this process had already applied an
    erasure at or after that cutoff is spared (a clock that stepped back must
    not make a newer recall look erased). ``erased_at_ms`` notes the erasure,
    in the same step, so entries recorded from now on are never its to erase.
    Returns how many were removed; unsaved ones are counted as ``erased_unsaved``.
    """
    def doomed(entry: tuple[int, VerdictEvent]) -> bool:
        seq, ev = entry
        if ev.profile_id != profile_id:
            return False
        if occurred_before_ms is None:
            return True
        return (ev.occurred_ms <= occurred_before_ms
                and _known_locked(profile_id, seq) < occurred_before_ms)

    with _lock:
        verdicts = [(entry, doomed(entry)) for entry in _ring]
        kept = [entry for entry, gone in verdicts if not gone]
        removed = len(verdicts) - len(kept)
        unsaved = sum(1 for (seq, _ev), gone in verdicts if gone and seq > _saved_seq)
        if removed:
            _ring.clear()
            _ring.extend(kept)
        _counters["erased_unsaved"] += unsaved
        if erased_at_ms is not None:
            _note_locked(profile_id, erased_at_ms)
    return removed


def counters() -> dict[str, int]:
    with _lock:
        out = dict(_counters)
        out["unsaved"] = len(_unsaved_suffix())
    return out


def _reset_for_testing() -> None:
    """Forget everything, as a fresh process would (tests only)."""
    global _seq, _saved_seq, _inflight_seq, _enabled, _boot_id
    with _lock:
        _ring.clear()
        _seq = _saved_seq = _inflight_seq = 0
        _enabled = False
        _boot_id = uuid.uuid4().hex
        for name in _COUNTER_NAMES:
            _counters[name] = 0
        _erasures_seen.clear()
    _wake.clear()


__all__ = [
    "ORIGIN_DASHBOARD", "ORIGIN_VIEW_CLI", "ORIGIN_VIEW_DASHBOARD", "ORIGIN_VIEW_MCP",
    "RING_CAPACITY", "VIEW_ORIGINS", "VerdictEvent", "WAKE_AT_UNSAVED", "boot_id",
    "call_as_dashboard", "counters", "enable", "erasure_known_at", "event_from_response",
    "forget_profile",
    "is_enabled", "mark_failed", "mark_saved", "origin", "recent", "record_recall_verdict",
    "snapshot_unsaved", "unsaved_for", "wake_event",
]
