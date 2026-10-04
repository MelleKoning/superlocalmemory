# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Read permission for dashboard reads when team accounts are on.

Moved out of ``unified_daemon.py`` (which re-exports the old private names) in
4.1.20, when this gap was closed:

* **Reads sent as POST.** The READ check ran for GET only, so the dashboard
  search (``POST /api/search``) and the memory chat
  (``POST /api/v3/chat/stream``) returned memory to a caller without READ on
  the workspace. :data:`READ_ONLY_POST_PATHS` puts them under the same check.

Every gate is a no-op until team accounts exist (single-operator installs are
unchanged) and fails closed (503) when the account store cannot be read.
"""

from __future__ import annotations

_SENSITIVE_READ_PREFIXES = (
    "/api/memories", "/api/facts", "/api/clusters", "/api/graph",
    "/api/v3/associations", "/api/v3/core-memory",
    "/api/v3/soft-prompts", "/api/v3/dashboard", "/api/v3/mode",
    "/api/v3/embedding/config", "/api/v3/scope/config",
    "/api/v3/storage/config", "/api/v3/daemon/config",
    "/api/v3/mesh/config", "/api/v3/trust/config",
    "/api/v3/forgetting/config", "/api/v3/mcp/profiles",
    "/api/learning", "/api/behavioral",
    # Event stream, agent activity, trust signals, and v3 profiling data
    # expose cross-agent coordination signals and behavioral profiles.
    "/events", "/api/events", "/api/agents", "/api/trust/",
    "/api/v3/abstraction", "/api/v3/insights",
)
_SENSITIVE_READ_EXACT_PATHS = (
    "/api/search", "/api/v3/recall/trace", "/api/patterns",
    "/api/feedback/stats", "/api/stats", "/api/timeline",
    # L3-01: project/agent names and per-bucket memory counts — the same
    # cross-tenant metadata the prefixes above already gate.
    "/api/v3/facets",
)
#: POST routes that only read and return memory content (4.1.20).
READ_ONLY_POST_PATHS = frozenset({"/api/search", "/api/v3/chat/stream"})


def is_sensitive_dashboard_read(method: str, path: str) -> bool:
    if method == "POST":
        return path in READ_ONLY_POST_PATHS
    return (
        method == "GET"
        and (
            path.startswith(_SENSITIVE_READ_PREFIXES)
            or path in _SENSITIVE_READ_EXACT_PATHS
            or path.startswith("/api/v3/recall")
        )
    )


def _json(status: int, message: str):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status, content={"error": message})


def _session_token(request) -> str:
    return (request.headers.get("x-slm-user-session", "")
            or (request.cookies.get("slm_session", "") if request.cookies else ""))


def _rbac_active(rbac) -> bool | None:
    """True/False, or None when the account store could not be read."""
    try:
        return rbac.user_count() > 0
    except Exception:  # noqa: BLE001
        return None


def rbac_read_gate(request, app_state, *, profile: str | None = None,
                   machine_principal: bool = False):
    """READ on ``profile`` (default: the active one). ``None`` allows; else the
    refusal response.

    ``machine_principal`` is a caller that proved it is one of this computer's
    own agents (the daemon capability) or a mesh node (the shared secret):
    a program, not a person, so company mode's "every person signs in" does
    not apply to it. A session it presents is still checked.
    """
    rbac = getattr(app_state, "rbac", None)
    if rbac is None:
        return None
    active = _rbac_active(rbac)
    if active is None:
        # Fail CLOSED: if we cannot determine RBAC state we must not silently
        # allow reads (a DB error would otherwise open the whole read surface).
        return _json(503, "authorization temporarily unavailable")
    if not active:
        return None  # single-operator install — reads are open
    token = _session_token(request)
    user = rbac.resolve_session(token) if token else None
    if user is None:
        if rbac.require_login() and not machine_principal:
            return _json(401, "Login required to read memory.")
        return None  # owner/operator, personal mode
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.routes.helpers import get_active_profile

    if rbac.has_permission(user["user_id"], profile or get_active_profile(),
                           Permission.READ):
        return None
    return _json(403, "Your role cannot read this workspace.")


__all__ = [
    "READ_ONLY_POST_PATHS",
    "is_sensitive_dashboard_read",
    "rbac_read_gate",
]
