# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com

"""Saved views over HTTP — the routes the dashboard and ``slm view`` both use.

    GET  /api/v3/views                 list the active profile's views
    GET  /api/v3/views/show?name=      one view's definition
    GET  /api/v3/views/run?name=       run it: recall's answer, with memory ids
    POST /api/v3/views                 create   {name, query, filters?, limit?}
    POST /api/v3/views/rename          rename   {name, new_name}
    POST /api/v3/views/delete          delete   {name}

Names travel in the query string or body, never in the path: a name may hold
a "/" and must still reach the right view.

WHO MAY DO WHAT
---------------
Reads (list, show, run) need READ on the workspace — the same permission as
recall — and are listed as sensitive reads in ``server/read_gates.py``.
Writes need a credential this product issued (``require_write_actor``) and
WRITE on the workspace, so a viewer can run views but not change them. Every
call is scoped to the active profile; nothing here takes a profile from the
caller.

RUNNING IS RECALL
-----------------
``run`` calls ``engine.recall`` with exactly the arguments
``views.runner.recall_arguments`` derives — the same call the Recall Lab makes —
and serialises the answer with the shared recall serializer. Each run is its
recall under a synthetic ``view:`` session id, which continuity ignores, so one
run never biases the next: the same view on an unchanged store returns the same
memories in the same order.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from superlocalmemory.access.rbac import Permission
from superlocalmemory.views import (
    ViewError,
    ViewStore,
    default_store,
    recall_arguments,
    shape_run,
    view_session_id,
)
from superlocalmemory.views import model

logger = logging.getLogger("superlocalmemory.routes.views")
router = APIRouter(prefix="/api/v3/views", tags=["views"])

_Name = Annotated[str, Query(min_length=1, max_length=model.MAX_NAME_CHARS * 4)]

#: Refusal code -> HTTP status. "Not ready" is a 409 (state), not a 503, so a
#: client never mistakes it for a daemon that is down and retries.
_STATUS = {model.VIEW_NOT_FOUND: 404, model.VIEW_EXISTS: 409,
           model.TOO_MANY_VIEWS: 409, model.VIEWS_UNAVAILABLE: 409}


class CreateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Bounds here are generous outer limits for the request size; the exact
    # rules (and their plain-English refusals) live in views.model.
    name: Any = Field(...)
    query: Any = Field(...)
    filters: dict[str, Any] | None = Field(None, max_length=16)
    limit: Any = None


class RenameBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Any = Field(...)
    new_name: Any = Field(...)


class DeleteBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Any = Field(...)


# -- gates and helpers --------------------------------------------------------


def _profile() -> str:
    from superlocalmemory.server.routes.helpers import get_active_profile

    return get_active_profile()


def _read_gate(request: Request) -> str:
    from superlocalmemory.server.rbac_enforce import require_permission

    profile = _profile()
    require_permission(request, Permission.READ, profile=profile)
    return profile


def _write_gate(request: Request) -> str:
    from superlocalmemory.server import write_identity
    from superlocalmemory.server.rbac_enforce import require_permission

    write_identity.require_write_actor(
        request, getattr(request.app.state, "daemon_descriptor", None),
        actor_kind="saved-views")
    profile = _profile()
    require_permission(request, Permission.WRITE, profile=profile)
    return profile


def _store() -> ViewStore:
    return default_store()


def _refusal(exc: ViewError) -> JSONResponse:
    status = 422 if exc.code in model.INPUT_CODES else _STATUS.get(exc.code, 409)
    if status == 409:
        # The CLI's daemon client reads a 409's ``detail`` as text.
        return JSONResponse({"detail": exc.message, "code": exc.code}, status_code=status)
    return JSONResponse({"detail": exc.as_dict()}, status_code=status)


def _internal_error() -> JSONResponse:
    logger.exception("saved views: request failed")
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


# -- reads --------------------------------------------------------------------


@router.get("")
def list_views(request: Request):
    profile = _read_gate(request)
    try:
        views = _store().list(profile)
    except ViewError as exc:
        return _refusal(exc)
    except Exception:  # noqa: BLE001 — typed refusals above; never a traceback
        return _internal_error()
    return {"profile": profile, "views": [v.to_dict() for v in views],
            "count": len(views), "limits": _limits()}


@router.get("/show")
def show_view(request: Request, name: _Name):
    profile = _read_gate(request)
    try:
        return {"profile": profile, "view": _store().get(profile, name).to_dict()}
    except ViewError as exc:
        return _refusal(exc)
    except Exception:  # noqa: BLE001
        return _internal_error()


@router.get("/run")
async def run_view(request: Request, name: _Name):
    profile = _read_gate(request)
    try:
        view = _store().get(profile, name)
    except ViewError as exc:
        return _refusal(exc)
    except Exception:  # noqa: BLE001
        return _internal_error()

    from superlocalmemory.server.routes.helpers import get_engine_lazy

    engine = get_engine_lazy(request.app.state)
    if engine is None:
        raise HTTPException(503, detail="The memory engine is starting; try again shortly.")
    try:
        response = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _recall(engine, view, profile))
        return shape_run(view, _serialize(engine, response, view.limit, profile))
    except Exception:  # noqa: BLE001
        return _internal_error()


def _recall(engine: Any, view: model.SavedView, profile: str) -> Any:
    """``engine.recall`` with the view's arguments, outside any conversation.

    ``profile_id`` is the profile the view was read from, named explicitly so a
    profile switch between reading the view and running it cannot run one
    profile's view against another profile's memories.
    """
    from superlocalmemory.core.recall_gate import begin_recall, end_recall
    from superlocalmemory.core.recall_pipeline import resolve_hot_path_fast
    from superlocalmemory.retrieval.facets import Facets

    args = recall_arguments(view)
    facets = Facets.of(kind=args.get("kind"))
    begin_recall()
    try:
        return engine.recall(
            args["query"], profile_id=profile, limit=args["limit"],
            # Synthetic, so continuity ignores it: a conversation's working set
            # would bias a view's second run toward what its first run showed.
            session_id=view_session_id(view),
            agent_id="saved-view",
            fast=resolve_hot_path_fast(None, getattr(engine, "_config", None)),
            window=args.get("window"), as_of=args.get("as_of"),
            **({} if facets.empty else {"facets": facets}),
        )
    finally:
        end_recall()


def _serialize(engine: Any, response: Any, limit: int, profile: str) -> dict[str, Any]:
    from superlocalmemory.server.recall_serializer import (
        recall_response_metadata,
        serialize_recall_response,
    )
    from superlocalmemory.server.routes.answer_check_history import answer_check_block

    retrieval = getattr(getattr(engine, "_config", None), "retrieval", None)
    results, no_confident_match = serialize_recall_response(
        response, limit=limit,
        per_fact_max=getattr(retrieval, "recall_per_fact_max_chars", 2400),
        total_max=getattr(retrieval, "recall_total_max_chars", 12000),
    )
    return {
        "profile": profile,
        "results": results,
        "no_confident_match": no_confident_match,
        "query_type": getattr(response, "query_type", ""),
        "retrieval_time_ms": round(float(getattr(response, "retrieval_time_ms", 0) or 0), 1),
        **recall_response_metadata(response),
        "answer_check": answer_check_block(response),
    }


def _limits() -> dict[str, Any]:
    """The rules a client builds its form from, so the form cannot drift from them."""
    from superlocalmemory.storage.memory_kinds import LABELS

    return {"max_views": model.MAX_VIEWS_PER_PROFILE, "max_name_chars": model.MAX_NAME_CHARS,
            "max_query_chars": model.MAX_QUERY_CHARS, "max_results": model.MAX_LIMIT,
            "default_results": model.DEFAULT_LIMIT, "filters": sorted(model.FILTERS),
            "kinds": [{"value": kind.value, "label": label} for kind, label in LABELS.items()]}


# -- writes -------------------------------------------------------------------


@router.post("")
def create_view(request: Request, body: CreateBody):
    profile = _write_gate(request)
    try:
        view = _store().create(profile, name=body.name, query=body.query,
                               filters=body.filters, limit=body.limit)
    except ViewError as exc:
        return _refusal(exc)
    except Exception:  # noqa: BLE001
        return _internal_error()
    return {"profile": profile, "view": view.to_dict(),
            "message": f"Saved the view {view.name!r}."}


@router.post("/rename")
def rename_view(request: Request, body: RenameBody):
    profile = _write_gate(request)
    try:
        view = _store().rename(profile, body.name, body.new_name)
    except ViewError as exc:
        return _refusal(exc)
    except Exception:  # noqa: BLE001
        return _internal_error()
    return {"profile": profile, "view": view.to_dict(),
            "message": f"Renamed the view to {view.name!r}."}


@router.post("/delete")
def delete_view(request: Request, body: DeleteBody):
    profile = _write_gate(request)
    try:
        view = _store().delete(profile, body.name)
    except ViewError as exc:
        return _refusal(exc)
    except Exception:  # noqa: BLE001
        return _internal_error()
    return {"profile": profile, "deleted": view.to_dict(),
            "message": f"Deleted the view {view.name!r}. No memory was changed."}


__all__ = ["router"]
