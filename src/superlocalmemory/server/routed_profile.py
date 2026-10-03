# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Shared pieces of per-request profile routing for HTTP routes.

A request may name a ``profile_id`` to be served against that profile without
moving the daemon's active one. Every route that accepts it reads the value the
same way and answers an unknown profile with the same body, which lives here so
the daemon module and the route modules cannot drift apart.
"""

from __future__ import annotations

from typing import Any


class RoutedProfileError(ValueError):
    """A ``profile_id`` that is present but is not text."""


def routed_profile_id(value: Any) -> str | None:
    """The profile a request is routed to, or None for the active profile.

    Absent, empty and whitespace-only all mean "the active profile", so a
    legacy caller is never treated as routed. Padding is stripped.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise RoutedProfileError("profile_id must be a string")
    return value.strip() or None


def unknown_profile_body(profile_id: str) -> dict:
    """Error body for a per-request route to a profile that does not exist.

    Unknown means 404, no implicit creation, no engine call. This is the
    daemon's route-local structured-error shape (same pattern as the
    ``invalid_as_of`` 400 on /recall); the global FastAPI ``{"detail": ...}``
    format is deliberately untouched.
    """
    return {
        "success": False,
        "error": {
            "code": "unknown_profile",
            "profile_id": profile_id,
            "message": (
                f"profile {profile_id!r} does not exist; per-request routing "
                "never creates a profile implicitly"
            ),
        },
    }


__all__ = ["RoutedProfileError", "routed_profile_id", "unknown_profile_body"]
