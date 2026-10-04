# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""The gate for MCP requests from other computers.

Every request to ``/mcp`` that is not from this computer passes, in order:

1. **HTTPS.** Over plain HTTP it is refused (``remote_requires_tls``). The
   remote listener only speaks TLS; on the main listener an operator may accept
   plain HTTP from other computers with ``SLM_REMOTE_ALLOW_PLAINTEXT=1`` (logged).
2. **Company mode.** With team accounts that require sign-in, remote keys are
   refused (``remote_mcp_company_mode``): a key is not bound to a user's role.
3. **A key.** ``Authorization: Bearer slmr_...`` (a named remote key), or the
   older ``X-SLM-API-Key`` (treated as a write key). The install token, the hook
   token and the daemon capability are never remote credentials and are ignored.
   A named key made before 4.1.20 that is not yet bound to a profile is refused
   (``remote_key_unbound``).
4. The tool policy (:mod:`server.remote_tool_policy`) then decides per tool,
   and :mod:`server.remote_profile_binding` keeps the call inside the key's
   profile.

"This computer" means :func:`server.access_gate.is_local_peer` and the request
did not arrive on the remote listener: the remote listener treats every caller
as remote, even one on 127.0.0.1.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from superlocalmemory.server.remote_keys import RemoteKeyStore, default_store

logger = logging.getLogger("superlocalmemory.remote")

PLAINTEXT_ENV = "SLM_REMOTE_ALLOW_PLAINTEXT"
#: Scope key under which the authenticated principal travels to the MCP app.
PRINCIPAL_SCOPE_KEY = "slm_remote_principal"
#: Set on every request that arrived on the TLS remote listener.
REMOTE_LISTENER_SCOPE_KEY = "slm_remote_listener"


@dataclass(frozen=True)
class RemotePrincipal:
    kind: Literal["remote-key", "legacy-api-key"]
    key_id: str
    name: str
    scope: Literal["read", "write"]
    #: The one profile this caller may reach. Always set for a named key. The
    #: older SLM API key has no stored binding, so it reaches only the profile
    #: active on this computer at the time of each call (``None``).
    profile: str | None = None

    @property
    def actor_id(self) -> str:
        return f"remote-key:{self.key_id}" if self.kind == "remote-key" else "api-key:remote"


@dataclass(frozen=True)
class GateDecision:
    status: int
    body: dict[str, str] | None = None
    principal: RemotePrincipal | None = None

    @property
    def allowed(self) -> bool:
        return self.principal is not None


def is_trusted_local_peer(scope: Mapping[str, Any]) -> bool:
    """This computer, not proxied, and not on the remote listener."""
    if scope.get(REMOTE_LISTENER_SCOPE_KEY):
        return False
    from superlocalmemory.server.access_gate import is_local_peer

    return is_local_peer(dict(scope))


def request_is_tls(scope: Mapping[str, Any]) -> bool:
    return scope.get("scheme") in ("https", "wss")


def plaintext_allowed(env: Mapping[str, str] | None = None) -> bool:
    return (env if env is not None else os.environ).get(PLAINTEXT_ENV, "").strip() == "1"


def principal_from_scope(scope: Mapping[str, Any]) -> RemotePrincipal | None:
    principal = scope.get(PRINCIPAL_SCOPE_KEY)
    return principal if isinstance(principal, RemotePrincipal) else None


def authenticate_remote(headers: Mapping[str, str],
                        store: RemoteKeyStore | None = None) -> RemotePrincipal | None:
    """The principal for these headers, or ``None``. Header names are lower-case."""
    from superlocalmemory.infra.auth_middleware import verify_api_key
    from superlocalmemory.server.remote_keys import KEY_PREFIX

    auth = headers.get("authorization", "") or ""
    if auth[:7].lower() == "bearer ":
        presented = auth[7:].strip()
        if presented.startswith(KEY_PREFIX):
            key = (store or default_store()).verify(presented)
            if key is None:
                return None
            return RemotePrincipal("remote-key", key.key_id, key.name, key.scope,
                                   key.profile)
        # The SLM API key may travel as a Bearer token too: clients strip
        # Authorization on a redirect to another site, unlike X-SLM-API-Key.
        if verify_api_key(presented):
            return RemotePrincipal("legacy-api-key", "api_key", "api_key", "write")
        return None
    legacy = headers.get("x-slm-api-key", "") or ""
    if legacy and verify_api_key(legacy):
        return RemotePrincipal("legacy-api-key", "api_key", "api_key", "write")
    return None


def company_mode_active(app_state: Any) -> bool:
    """Team accounts with required sign-in, or an enterprise deployment. Fails closed."""
    try:
        from superlocalmemory.core.admission import _resolve_deployment

        if _resolve_deployment().is_enterprise:
            return True
        rbac = getattr(app_state, "rbac", None)
        return bool(rbac is not None and rbac.require_login())
    except Exception as exc:  # noqa: BLE001 — an unknown deployment is not personal
        logger.warning("could not read the deployment mode (%s); refusing remote MCP", exc)
        return True


# -- refusal bodies -------------------------------------------------------------------

TLS_REQUIRED = {
    "error": "remote_requires_tls",
    "message": (
        "Remote memory access needs HTTPS. Set it up on the SLM computer with "
        "'slm remote enable' (see docs/hermes.md#remote). If a key was just sent "
        "over plain HTTP, revoke it: slm remote keys revoke <name>."
    ),
}
COMPANY_MODE = {
    "error": "remote_mcp_company_mode",
    "message": ("Company mode does not accept remote MCP keys in this release; remote "
                "keys are not bound to a user role."),
}
KEY_UNBOUND = {
    "error": "remote_key_unbound",
    "message": ("This remote key was made before keys were tied to one profile and is "
                "not bound yet. On the SLM computer run 'slm remote keys list' (it binds "
                "the key to the active profile), or make a new key with "
                "'slm remote keys add <name> --profile <profile>'."),
}
AUTH_REQUIRED = {
    "error": "remote_auth_required",
    "message": ("Present 'Authorization: Bearer slmr_...' (create one on the SLM "
                "computer with: slm remote keys add <name>)."),
}

_auth_fail_log: dict[str, float] = {}
_auth_fail_lock = threading.Lock()
_plaintext_warned = False


def _log_refusal(peer: str, code: str) -> None:
    """At most one line per peer per minute. Never header values."""
    now = time.monotonic()
    with _auth_fail_lock:
        if now - _auth_fail_log.get(peer, -120.0) < 60.0:
            return
        _auth_fail_log[peer] = now
        if len(_auth_fail_log) > 4096:
            _auth_fail_log.clear()
    logger.warning("remote MCP refused: peer=%s reason=%s", peer, code)


def _peer(scope: Mapping[str, Any]) -> str:
    return str(scope.get("slm_remote_peer") or (scope.get("client") or ("?", 0))[0])


def gate_remote_mcp(scope: Mapping[str, Any], headers: Mapping[str, str], app_state: Any,
                    store: RemoteKeyStore | None = None) -> GateDecision:
    """Decide one ``/mcp`` request from another computer."""
    global _plaintext_warned
    peer = _peer(scope)
    if not request_is_tls(scope):
        if scope.get(REMOTE_LISTENER_SCOPE_KEY) or not plaintext_allowed():
            _log_refusal(peer, TLS_REQUIRED["error"])
            return GateDecision(403, TLS_REQUIRED)
        if not _plaintext_warned:
            _plaintext_warned = True
            logger.warning("%s=1: accepting remote MCP over plain HTTP (first use from %s). "
                           "Keys and memory are readable on the network.", PLAINTEXT_ENV, peer)
    if company_mode_active(app_state):
        _log_refusal(peer, COMPANY_MODE["error"])
        return GateDecision(403, COMPANY_MODE)
    principal = authenticate_remote(headers, store)
    if principal is None:
        _log_refusal(peer, AUTH_REQUIRED["error"])
        return GateDecision(401, AUTH_REQUIRED)
    if principal.kind == "remote-key" and principal.profile is None:
        _log_refusal(peer, KEY_UNBOUND["error"])
        return GateDecision(403, KEY_UNBOUND)
    return GateDecision(200, None, principal)


__all__ = [
    "AUTH_REQUIRED",
    "COMPANY_MODE",
    "GateDecision",
    "KEY_UNBOUND",
    "PLAINTEXT_ENV",
    "PRINCIPAL_SCOPE_KEY",
    "REMOTE_LISTENER_SCOPE_KEY",
    "RemotePrincipal",
    "TLS_REQUIRED",
    "authenticate_remote",
    "company_mode_active",
    "gate_remote_mcp",
    "is_trusted_local_peer",
    "plaintext_allowed",
    "principal_from_scope",
    "request_is_tls",
]
