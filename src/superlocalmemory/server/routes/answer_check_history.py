# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Answer Check history — what the dashboard's Answer Check tab reads.

Four routes, all scoped to the active profile:

* ``GET /history/live``    — the newest checks, from memory (no disk read);
* ``GET /history/summary`` — counts, the abstention rate, latency vs 3.0 s;
* ``GET /history``         — saved checks, newest first, keyset-paginated;
* ``DELETE /history``      — clear this profile's history (credential + DELETE).

Every item is built by one function from a fixed list of fields, so nothing a
future change adds to the stored record (and certainly no question or memory
text, which is never stored) can reach a response by accident.

Reads need READ on the workspace when team accounts are on; the summary keeps
dashboard tests out of the numbers unless asked, so trying the tab cannot
change the rate it shows. Errors are one plain sentence — never a traceback.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from superlocalmemory.core import answer_check_history as history
from superlocalmemory.core import answer_check_history_store as store
from superlocalmemory.core.answer_check_history_stats import WINDOWS, outcome_key, summarize
from superlocalmemory.core.rate_limit import TokenBucket
from superlocalmemory.retrieval.answer_check_status import (
    ANSWER_CHECK_STATUSES,
    RECALL_CEILING_S,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3/answer-check", tags=["answer-check"])

CEILING_MS = RECALL_CEILING_S * 1000.0
_CURSOR_RE = re.compile(r"^(\d{1,15}):([0-9a-f]{32})$")
_SUMMARY_TTL_S = 5.0
_READ_FAILED = "Could not read the answer-check history. Try again in a moment."
_RATES = {"live": (2.0, 6.0), "history": (1.0, 5.0), "summary": (1.0, 5.0),
          "clear": (0.1, 1.0)}
_buckets: dict[str, TokenBucket] = {}
_bucket_lock = threading.Lock()
_summary_cache: dict[tuple, tuple[float, dict]] = {}
_cache_lock = threading.Lock()


# -- guards ----------------------------------------------------------------------

def _profile() -> str:
    from superlocalmemory.server.routes.helpers import get_active_profile

    return get_active_profile()


def _require_read(request: Request) -> str:
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.rbac_enforce import require_permission

    profile = _profile()
    require_permission(request, Permission.READ, profile=profile)
    return profile


def _require_clear(request: Request) -> str:
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.rbac_enforce import require_permission
    from superlocalmemory.server.write_identity import require_write_actor

    require_write_actor(request, getattr(request.app.state, "daemon_descriptor", None),
                        actor_kind="answer-check")
    profile = _profile()
    require_permission(request, Permission.DELETE, profile=profile)
    return profile


def _limited(name: str) -> JSONResponse | None:
    rate, burst = _RATES[name]
    with _bucket_lock:
        bucket = _buckets.setdefault(name, TokenBucket(rate, burst))
        if bucket.try_consume():
            return None
        wait_ms = bucket.ms_to_refill()
    return JSONResponse({"error": "Too many requests. Try again in a moment.",
                         "retry_after_ms": wait_ms}, status_code=429,
                        headers={"Retry-After": str(max(1, math.ceil(wait_ms / 1000)))})


def _reset_for_testing() -> None:
    with _bucket_lock:
        _buckets.clear()
    with _cache_lock:
        _summary_cache.clear()


# -- configuration -------------------------------------------------------------------

def _retrieval_config(request: Request) -> Any:
    engine = getattr(request.app.state, "engine", None)
    config = getattr(engine, "_config", None) or getattr(request.app.state, "config", None)
    return getattr(config, "retrieval", None)


def _enabled(request: Request) -> bool:
    return getattr(_retrieval_config(request), "answer_check_history", True) is not False


def _learning_db() -> Path:
    running = store._state.get("learning_db")
    if running is not None:
        return Path(running)
    from superlocalmemory.server.routes.helpers import DB_PATH

    return Path(DB_PATH).parent / "learning.db"


# -- the one item shape ----------------------------------------------------------------

def _iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


def _item(src: Any) -> dict[str, Any]:
    """The only shape any route emits for a check. A fixed field list."""
    row = src if isinstance(src, dict) else store.event_as_row(src)
    total = row.get("total_ms")
    return {
        "id": row["event_id"], "at": _iso(row["occurred_ms"]),
        "outcome": outcome_key(row["status"], row["detail"], bool(row["abstained"]),
                               int(row["result_count"] or 0)),
        "status": row["status"], "detail": row["detail"], "judge": row["backend"],
        "origin": "dashboard" if row["origin"] == history.ORIGIN_DASHBOARD else "other",
        "abstained": bool(row["abstained"]), "abstention_reason": row["abstention_reason"],
        "answer_confidence": row["answer_confidence"], "threshold": row["threshold"],
        "reordered": bool(row["reordered"]), "result_count": int(row["result_count"] or 0),
        "query_type": row["query_type"], "retrieval_ms": row["retrieval_ms"],
        "judge_ms": row["judge_ms"], "total_ms": total,
        "embed_ms": row.get("embed_ms"), "rerank_ms": row.get("rerank_ms"),
        "over_ceiling": bool(total is not None and total > CEILING_MS),
    }


def answer_check_block(response: Any) -> dict[str, Any]:
    """What ``POST /api/v3/recall/trace`` adds: this recall's check, with timings."""
    trace = getattr(response, "answer_check_trace", None)

    def timing(name: str) -> float | None:
        return getattr(trace, name, None) if trace is not None else None
    return {
        "status": getattr(response, "answer_check_status", "") or "",
        "detail": getattr(trace, "detail", "") if trace is not None else "",
        "judge": getattr(trace, "backend", "") if trace is not None else "",
        "abstained": bool(getattr(response, "abstained", False)),
        "abstention_reason": getattr(response, "abstention_reason", None),
        "answer_confidence": getattr(response, "answer_confidence", None),
        "threshold": timing("threshold"),
        "reordered": bool(timing("reordered")),
        "retrieval_ms": timing("retrieval_ms"), "judge_ms": timing("judge_ms"),
        "total_ms": timing("total_ms"), "ceiling_ms": CEILING_MS,
    }


def _disabled(**extra: Any) -> dict[str, Any]:
    return {"enabled": False, "items": [], **extra}


# -- routes ------------------------------------------------------------------------------

@router.get("/history/live")
def history_live(request: Request, after_seq: int = Query(0, ge=0),
                 limit: int = Query(20, ge=1, le=50)):
    profile = _require_read(request)
    if (limited := _limited("live")) is not None:
        return limited
    if not _enabled(request):
        return _disabled(boot_id=history.boot_id(), last_seq=0)
    entries, last = history.recent(profile, after_seq=after_seq, limit=limit)
    return {"enabled": True, "boot_id": history.boot_id(), "last_seq": last,
            "items": [{"seq": seq, **_item(ev)} for seq, ev in entries]}


def _summary_rows(profile: str, window: str, include_dashboard: bool) -> list[dict]:
    since_ms = int(time.time() * 1000) - WINDOWS[window]
    rows = store.read_window(_learning_db(), profile, since_ms=since_ms,
                             include_dashboard=include_dashboard)
    seen = {row["event_id"] for row in rows}
    for ev in history.unsaved_for(profile):
        if ev.event_id in seen or ev.occurred_ms < since_ms:
            continue
        if not include_dashboard and ev.origin == history.ORIGIN_DASHBOARD:
            continue
        rows.append(dict(store.event_as_row(ev)))
    return rows


def _recording() -> dict[str, Any]:
    info, c = store.writer_info(), history.counters()
    return {"boot_id": history.boot_id(), "writer_running": info["running"],
            "last_saved_at": _iso(info["last_saved_ms"]),
            "retention_days": info["retention_days"], "max_rows": info["max_rows"],
            "unsaved": c["unsaved"],
            "since_start": {k: c[k] for k in ("recorded", "saved", "dropped_before_save",
                                               "save_failures", "erased_unsaved")}}


@router.get("/history/summary")
def history_summary(request: Request, window: str = Query("7d", pattern="^(24h|7d|30d)$"),
                    include_dashboard: bool = False):
    profile = _require_read(request)
    if (limited := _limited("summary")) is not None:
        return limited
    if not _enabled(request):
        return {"enabled": False, "profile": profile, "window": window}
    key = (profile, window, include_dashboard)
    with _cache_lock:
        hit = _summary_cache.get(key)
    if hit is not None and time.monotonic() - hit[0] < _SUMMARY_TTL_S:
        return {**hit[1], "recording": _recording()}
    try:
        rows = _summary_rows(profile, window, include_dashboard)
    except Exception as exc:  # noqa: BLE001
        logger.warning("answer-check summary failed: %s", type(exc).__name__)
        return JSONResponse({"error": _READ_FAILED}, status_code=500)
    body = {"enabled": True, "profile": profile, "window": window,
            "since": _iso(int(time.time() * 1000) - WINDOWS[window]),
            "include_dashboard": include_dashboard,
            **summarize(rows, ceiling_ms=CEILING_MS)}
    with _cache_lock:
        _summary_cache[key] = (time.monotonic(), body)
    return {**body, "recording": _recording()}


def _parse_cursor(cursor: str | None) -> tuple[int, str] | None | JSONResponse:
    if not cursor:
        return None
    match = _CURSOR_RE.match(cursor)
    if match is None:
        return JSONResponse({"error": "That page link is not valid."}, status_code=400)
    return int(match.group(1)), match.group(2)


def _first_page(profile: str, rows: list[dict], *, status: str | None, since_ms: int,
                limit: int, more: bool) -> tuple[list[dict], str | None]:
    seen = {row["event_id"] for row in rows}
    merged = list(rows)
    for ev in history.unsaved_for(profile):
        if ev.event_id in seen or ev.occurred_ms < since_ms or (status and ev.status != status):
            continue
        merged.append(dict(store.event_as_row(ev)))
    merged.sort(key=lambda r: (r["occurred_ms"], r["event_id"]), reverse=True)
    page = merged[:limit]
    if (more or len(merged) > limit) and page:
        return page, f"{page[-1]['occurred_ms']}:{page[-1]['event_id']}"
    return page, None


@router.get("/history")
def history_page(request: Request, limit: int = Query(50, ge=1, le=200),
                 cursor: str | None = Query(None, max_length=64),
                 status: str | None = Query(None, max_length=16),
                 window: str = Query("7d", pattern="^(24h|7d|30d|all)$")):
    profile = _require_read(request)
    if (limited := _limited("history")) is not None:
        return limited
    if status is not None and status not in ANSWER_CHECK_STATUSES:
        return JSONResponse({"error": "Unknown status filter."}, status_code=400)
    parsed = _parse_cursor(cursor)
    if isinstance(parsed, JSONResponse):
        return parsed
    if not _enabled(request):
        return _disabled(next_cursor=None)
    since_ms = 0 if window == "all" else int(time.time() * 1000) - WINDOWS[window]
    try:
        rows, next_cursor = store.read_page(_learning_db(), profile, cursor=parsed,
                                            limit=limit, status=status, since_ms=since_ms)
        if parsed is None:
            rows, next_cursor = _first_page(profile, rows, status=status, since_ms=since_ms,
                                            limit=limit, more=next_cursor is not None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("answer-check history read failed: %s", type(exc).__name__)
        return JSONResponse({"error": _READ_FAILED}, status_code=500)
    return {"enabled": True, "items": [_item(r) for r in rows], "next_cursor": next_cursor}


@router.delete("/history")
def history_clear(request: Request):
    profile = _require_clear(request)
    if (limited := _limited("clear")) is not None:
        return limited
    try:
        cleared = store.clear_profile(_learning_db(), profile)
    except Exception as exc:  # noqa: BLE001
        logger.warning("answer-check history clear failed: %s", type(exc).__name__)
        return JSONResponse({"error": "Could not clear the saved history. Try again in a moment."},
                            status_code=500)
    with _cache_lock:
        _summary_cache.clear()
    return {"cleared": cleared}


__all__ = ["answer_check_block", "router"]
