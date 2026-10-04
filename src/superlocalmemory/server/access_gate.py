# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Who may reach this computer's memory from another computer.

A caller on this computer (the socket peer is loopback and no proxy carried
the request) is the local user, as before. Every other caller must prove who
it is before any route runs, for reads as well as writes. Before 4.1.20 a read
from the LAN needed nothing: ``GET /api/memories`` from any machine that could
reach the port returned memories.

A network caller is let through when it presents one of:

* ``X-SLM-API-Key`` matching the configured API key;
* ``X-SLM-Daemon-Capability`` for this exact daemon instance;
* a team-account session (``X-SLM-User-Session`` or the ``slm_session``
  cookie from the dashboard sign-in);
* the mesh shared secret, on ``/mesh/*`` only;
* or it comes from an address the operator allowlisted for LAN use
  (``SLM_REMOTE=1`` with ``SLM_MCP_ALLOWED_HOSTS``).

Paths that carry no memory are open to everyone (:data:`PUBLIC_PATHS`,
:data:`PUBLIC_PREFIXES`); two path families answer for themselves
(:data:`SELF_GATED_PREFIXES`). ``tests/test_security/test_nonlocal_read_gate.py``
walks the application's route table and fails when a route is reachable
without credentials and is not on these reviewed lists.

The install token is not a network credential: the dashboard page carries it,
so anyone who can load the page can read it.
"""

from __future__ import annotations

import hmac
import json
import logging
from typing import Any

from superlocalmemory.server.loopback import is_loopback

logger = logging.getLogger(__name__)

#: Reviewed: the dashboard shell and the liveness probe. No memory content.
PUBLIC_PATHS = frozenset({"/", "/health", "/favicon.ico"})
#: Reviewed: packaged static files (HTML/JS/CSS/icons). No memory content.
PUBLIC_PREFIXES = ("/static/",)
#: Reviewed: these paths run their own credential check for network callers.
#: ``/mcp`` requires HTTPS and a remote key (``server.remote_access``).
#: ``/v1``, ``/v1beta`` are the optional LLM proxy: the caller brings its own
#: provider key and the route returns the provider's answer, not memory.
SELF_GATED_PREFIXES = ("/mcp/", "/v1/", "/v1beta/")

REFUSAL_MESSAGE = (
    "This SuperLocalMemory answers other computers only when they sign in. "
    "To sign in, send the API key in the X-SLM-API-Key header (the key is the "
    "file 'api_key' in the SLM data folder) or a team-account session in "
    "X-SLM-User-Session. To let a trusted LAN computer in without a key, start "
    "SLM with SLM_REMOTE=1 and list its address in SLM_MCP_ALLOWED_HOSTS. On "
    "the SLM computer itself, use the 127.0.0.1 address. "
    "See docs/distributed-deployment.md."
)
_REFUSAL_BODY = json.dumps(
    {"error": "remote_auth_required", "message": REFUSAL_MESSAGE},
).encode("utf-8")
#: WebSocket close code for "not authorised" (the dashboard socket uses it too).
_WS_UNAUTHORISED = 4001


def path_policy(path: str) -> str:
    """``public``, ``self-gated`` or ``credentials`` for a network caller."""
    if path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES):
        return "public"
    if path == "/mcp" or path.startswith(SELF_GATED_PREFIXES):
        return "self-gated"
    return "credentials"


def is_local_peer(scope: dict[str, Any]) -> bool:
    """The socket peer is this computer and no proxy carried the request."""
    if scope.get("slm_forwarded_demoted") or scope.get("slm_remote_listener"):
        return False
    host = (scope.get("client") or ("", 0))[0]
    if is_loopback(host):
        return True
    # Starlette's in-process test client, exactly as write_identity allows it.
    from superlocalmemory.server import write_identity

    return host == "testclient" and write_identity._TEST_ISOLATION_ALLOWED


def _headers(scope: dict[str, Any]) -> dict[str, str]:
    return {name.decode("latin-1").lower(): value.decode("latin-1")
            for name, value in scope.get("headers") or ()}


def _cookie(headers: dict[str, str], name: str) -> str:
    from http.cookies import SimpleCookie

    jar: SimpleCookie = SimpleCookie()
    try:
        jar.load(headers.get("cookie", ""))
    except Exception:  # noqa: BLE001 — a malformed cookie is no cookie
        return ""
    morsel = jar.get(name)
    return morsel.value if morsel is not None else ""


def _api_key_ok(headers: dict[str, str]) -> bool:
    from superlocalmemory.infra.auth_middleware import verify_api_key

    return verify_api_key(headers.get("x-slm-api-key", ""))


def _capability_ok(headers: dict[str, str], app_state: Any) -> bool:
    presented = headers.get("x-slm-daemon-capability", "")
    descriptor = getattr(app_state, "daemon_descriptor", None)
    if not presented or descriptor is None:
        return False
    expected_instance = str(getattr(descriptor, "instance_id", ""))
    return (hmac.compare_digest(presented, str(getattr(descriptor, "capability", "")))
            and hmac.compare_digest(headers.get("x-slm-target-instance", ""),
                                    expected_instance))


def _session_ok(headers: dict[str, str], app_state: Any) -> bool:
    rbac = getattr(app_state, "rbac", None)
    token = headers.get("x-slm-user-session", "") or _cookie(headers, "slm_session")
    if rbac is None or not token:
        return False
    return rbac.resolve_session(token) is not None


def mesh_secret_ok(headers: dict[str, str], app_state: Any) -> bool:
    """The caller presented this node's mesh shared secret."""
    broker = getattr(app_state, "mesh_broker", None)
    secret = getattr(broker, "_shared_secret", None) if broker is not None else None
    if not secret:
        return False
    presented = (headers.get("x-mesh-secret", "")
                 or headers.get("authorization", "").removeprefix("Bearer ").strip())
    return bool(presented) and hmac.compare_digest(presented, secret)


def has_network_credential(scope: dict[str, Any], app_state: Any) -> bool:
    """Whether a caller on another computer proved who it is. Fails closed."""
    headers = _headers(scope)
    host = (scope.get("client") or ("", 0))[0]
    try:
        from superlocalmemory.core.remote_mode import is_lan_client_allowed

        if not scope.get("slm_forwarded_demoted") and is_lan_client_allowed(host):
            return True
        if _api_key_ok(headers) or _capability_ok(headers, app_state):
            return True
        if scope.get("path", "").startswith("/mesh/") and mesh_secret_ok(headers, app_state):
            return True
        return _session_ok(headers, app_state)
    except Exception as exc:  # noqa: BLE001 — a check that failed is a no
        logger.warning("network credential check failed (%s); refusing", exc)
        return False


def network_caller_allowed(scope: dict[str, Any], app_state: Any) -> bool:
    """The whole decision, for one HTTP or WebSocket request."""
    if is_local_peer(scope):
        return True
    if path_policy(scope.get("path", "")) != "credentials":
        return True
    return has_network_credential(scope, app_state)


class NonLocalAccessGate:
    """Pure ASGI: refuses a network caller without credentials before any route,
    middleware-installed or not, runs. Covers HTTP and WebSocket."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        app = scope.get("app")
        if network_caller_allowed(scope, getattr(app, "state", None)):
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": _WS_UNAUTHORISED})
            return
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [(b"content-type", b"application/json"),
                        (b"content-length", str(len(_REFUSAL_BODY)).encode())],
        })
        await send({"type": "http.response.body", "body": _REFUSAL_BODY})


__all__ = [
    "NonLocalAccessGate",
    "PUBLIC_PATHS",
    "PUBLIC_PREFIXES",
    "REFUSAL_MESSAGE",
    "SELF_GATED_PREFIXES",
    "has_network_credential",
    "is_local_peer",
    "mesh_secret_ok",
    "network_caller_allowed",
    "path_policy",
]
