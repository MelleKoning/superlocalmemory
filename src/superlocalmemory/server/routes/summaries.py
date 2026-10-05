# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com

"""The readable summary layer over HTTP (issue #113).

    GET /api/summary            a day, a project, or one session
    GET /api/summary/projects   projects SLM has seen, for the picker
    GET /api/summary/sessions   sessions with memories, for the picker

Moved out of ``routes/memories.py`` in 4.1.21 (that module was over twice the
size limit) when two things were added: the caller's time-zone offset, so
"today" means the caller's today, and the session list, without which a session
summary could not be asked for from the dashboard at all.

Every route needs READ on the workspace when team accounts are on
(``server/read_gates.py`` lists ``/api/summary`` as a sensitive prefix): the
summaries quote memories, and the pickers name projects and sessions.

Every summary carries ``coverage`` and ``source_fact_ids``. A summary that hides
how much it covered is the opaque generic summary issue #113 warned against.
Nothing here runs during remember or recall.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request

from .helpers import dict_factory, get_active_profile, get_db_connection

logger = logging.getLogger("superlocalmemory.routes.summaries")
router = APIRouter()

_KINDS = ("day", "project", "session")

#: tool_events is capped; at the cap the project list is a recent window, not history.
_TOOL_EVENT_RING_SIZE = 2000

_Offset = Annotated[int, Query(ge=-840, le=840)]


def _internal_error(detail: str) -> HTTPException:
    """Log the traceback here; never send it (or a path, or SQL) to the caller."""
    logger.exception(detail)
    return HTTPException(status_code=500, detail=detail)


def _memory_db():
    from superlocalmemory.infra.data_root import state_path

    db_path = state_path("memory.db")
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="no memory database")
    return db_path


def _load_config():
    """The loaded config, so Mode B/C write the summary. None = extractive.

    Omitting it would silently force the extractive path for every caller
    regardless of mode.
    """
    try:
        from superlocalmemory.core.config import SLMConfig

        return SLMConfig.load()
    except Exception:  # noqa: BLE001 - a deterministic summary beats an error
        logger.warning("summary: config could not be loaded; using extractive")
        return None


def _caller_day(target: str, tz_offset_minutes: int) -> str:
    """``today`` / ``yesterday`` / blank resolved in the CALLER's time zone."""
    day = (target or "").strip()
    local_today = (datetime.now(UTC) + timedelta(minutes=tz_offset_minutes)).date()
    if not day or day == "today":
        return local_today.isoformat()
    if day == "yesterday":
        return (local_today - timedelta(days=1)).isoformat()
    try:
        return date.fromisoformat(day).isoformat()
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_date",
                    "message": "Use a date like 2026-10-05, 'today' or 'yesterday'."},
        ) from exc


@router.get("/api/summary")
async def get_summary(request: Request, kind: str = "day", target: str = "",
                      tz_offset_minutes: _Offset = 0):
    """Readable summary of memories: a day, a project, or one session (#113).

    ``tz_offset_minutes`` is the caller's offset east of UTC (330 for India).
    The dashboard sends the browser's; without it a day is the UTC day.
    """
    kind = (kind or "day").strip().lower()
    if kind not in _KINDS:
        raise HTTPException(status_code=400, detail=f"unknown summary kind '{kind}'")
    profile = get_active_profile()
    db_path = _memory_db()
    cfg = _load_config()
    try:
        if kind == "day":
            from superlocalmemory.summaries import generate_daily_reflection

            result = generate_daily_reflection(
                db_path, _caller_day(target, tz_offset_minutes), profile, cfg,
                tz_offset_minutes=tz_offset_minutes)
        elif kind == "project":
            if not (target or "").strip():
                raise HTTPException(status_code=400, detail="project requires target")
            from superlocalmemory.summaries import generate_project_work_log

            result = generate_project_work_log(db_path, target.strip(), profile, cfg)
        else:
            if not (target or "").strip():
                raise HTTPException(status_code=400, detail="session requires target")
            from superlocalmemory.summaries import generate_session_summary

            result = generate_session_summary(db_path, target.strip(), profile, cfg)
    except HTTPException:
        raise
    except Exception:
        raise _internal_error("Summary generation error")

    # Say why, not just what: someone seeing a plainer result than expected
    # learns that a written summary needs a model, instead of reading the
    # feature as broken.
    from superlocalmemory.core.mode_capability import llm_capability

    generated = str(getattr(result, "generated_by", "") or "")
    return {
        "kind": result.kind,
        "profile_id": result.profile_id,
        "summary": result.content,
        "coverage": result.coverage,
        "generated_by": result.generated_by,
        "source_fact_ids": result.source_fact_ids,
        "source_count": len(result.source_fact_ids),
        "metadata": result.metadata,
        "capability": llm_capability(
            cfg,
            # This call has just been made, so its outcome is the honest answer
            # about availability -- better than asking the configuration again.
            llm_reachable=generated.startswith("llm"),
        ),
    }


@router.get("/api/summary/projects")
async def get_summary_projects(request: Request):
    """Projects SuperLocalMemory has actually observed, for the summary picker.

    The dashboard is a browser tab with no working directory, so there is no
    "this project" from the server's point of view; a list of the projects seen
    is the right control. Scope comes from ``tool_events.project_path`` — the
    directory an agent was working in when it called SLM (see the note at the
    top of summaries/project_work_log.py).

    ``tool_events`` is a bounded ring buffer, so this lists recently active
    projects; ``truncated`` says so instead of implying the list is complete.
    """
    profile = get_active_profile()
    try:
        conn = get_db_connection()
        # A SHARED read connection: other handlers set row_factory on it, so
        # read columns by name, never by position.
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT project_path AS path, COUNT(*) AS events
              FROM tool_events
             WHERE project_path IS NOT NULL AND project_path != ''
               AND profile_id = ?
             GROUP BY project_path
             ORDER BY events DESC, project_path ASC
             LIMIT 50
            """,
            (profile,),
        )
        rows = cursor.fetchall()
        total = cursor.execute("SELECT COUNT(*) AS n FROM tool_events").fetchone()["n"]
    except Exception:
        raise _internal_error("Project list error")

    projects = [
        {"path": r["path"], "events": r["events"], "label": _project_label(r["path"])}
        for r in rows
    ]
    return {
        "projects": projects,
        "profile_id": profile,
        "truncated": total >= _TOOL_EVENT_RING_SIZE,
        "event_rows": total,
    }


@router.get("/api/summary/sessions")
async def get_summary_sessions(request: Request,
                               limit: Annotated[int, Query(ge=1, le=100)] = 20):
    """Sessions with at least one visible memory, newest first, for the picker."""
    from superlocalmemory.summaries.sessions import list_recent_sessions

    profile = get_active_profile()
    try:
        sessions = list_recent_sessions(_memory_db(), profile, limit)
    except HTTPException:
        raise
    except Exception:
        raise _internal_error("Session list error")
    return {"sessions": sessions, "profile_id": profile}


def _project_label(path: str) -> str:
    """Short, human label for a project path: its last two segments."""
    parts = [p for p in str(path).replace("\\", "/").split("/") if p]
    if not parts:
        return str(path)
    return "/".join(parts[-2:]) if len(parts) > 1 else parts[-1]


__all__ = ["router"]
