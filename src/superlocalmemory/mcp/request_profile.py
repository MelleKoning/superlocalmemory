# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""The profile one MCP tool call works on.

Every MCP tool that reads or writes one profile's memory takes ``profile_id``:

* empty (or absent) means the active profile, exactly as before the argument
  existed;
* a name serves this one call for that profile. The active profile is never
  moved, and a name that is not a profile is refused, never created.

Remote access always sets it to the key's own profile
(``server/remote_profile_binding``), which is what lets a key bound to one
profile use every tool while this computer is using another one.
"""

from __future__ import annotations

from typing import Any

UNKNOWN_PROFILE = "unknown_profile"
ROUTING_NEEDS_DAEMON = "routing_needs_daemon"


def requested_profile(profile_id: object) -> str:
    """The profile a call names, or ``""`` for the active one."""
    if profile_id is None:
        return ""
    if not isinstance(profile_id, str):
        raise ValueError("profile_id must be text")
    return profile_id.strip()


def _db(engine: Any) -> Any:
    return getattr(engine, "_db", None) or getattr(engine, "db", None)


def profile_exists(engine: Any, profile_id: str) -> bool:
    """Whether ``profile_id`` is a profile on this computer."""
    return bool(_db(engine).execute(
        "SELECT 1 AS one FROM profiles WHERE profile_id = ?", (profile_id,)))


def unknown_profile_error(profile_id: str) -> dict[str, Any]:
    return {
        "success": False,
        "code": UNKNOWN_PROFILE,
        "retryable": False,
        "error": (f"There is no profile '{profile_id}' on this computer. "
                  "A tool call never creates a profile."),
    }


def routing_needs_daemon_error() -> dict[str, Any]:
    """For a named profile on a path that can only serve the active one."""
    return {
        "success": False,
        "code": ROUTING_NEEDS_DAEMON,
        "retryable": True,
        "error": ("Working on a named profile needs the SLM background service, "
                  "which is not running. Start it with 'slm restart'."),
    }


def tool_profile(engine: Any, profile_id: object) -> tuple[str, dict[str, Any] | None]:
    """``(profile, refusal)`` for one call.

    The named profile when it exists, else the engine's active profile when
    none is named. ``refusal`` is the answer to return instead of running when
    the name is not text or names no profile.
    """
    try:
        named = requested_profile(profile_id)
    except ValueError as exc:
        return "", {"success": False, "code": "invalid_profile_id", "retryable": False,
                    "error": str(exc)}
    if not named:
        return str(engine.profile_id), None
    if not profile_exists(engine, named):
        return named, unknown_profile_error(named)
    return named, None


__all__ = [
    "ROUTING_NEEDS_DAEMON",
    "UNKNOWN_PROFILE",
    "profile_exists",
    "requested_profile",
    "routing_needs_daemon_error",
    "tool_profile",
    "unknown_profile_error",
]
