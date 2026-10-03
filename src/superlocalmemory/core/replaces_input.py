# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The ``replaces`` value a caller sends with ``remember``: shape check only.

Every surface (the tool interface, HTTP, the command line) checks the value
here before doing any work, so a malformed value is refused the same way
everywhere and never reaches the store. Whether the id names a memory this
profile may replace is decided by the daemon, in
``core/remember_replaces.py``; this module imports nothing heavy so the
command line and the tool process can use it freely.
"""

from __future__ import annotations

import re

#: The value is not a usable id (wrong type, empty, bad characters, too long).
INVALID = "INVALID_REPLACES"
#: No fact or memory with that id is visible to the active profile.
NOT_FOUND = "REPLACES_NOT_FOUND"
#: The id names something this write may not replace (another profile's
#: memory, a different scope, or a write routed to another profile).
NOT_ALLOWED = "REPLACES_NOT_ALLOWED"

#: Fact and memory ids are 16 hex characters today; older stores and tests use
#: uuids and short names. Anything with whitespace, quotes or SQL punctuation
#: is not an id. The 128 cap matches the correction ledger's identifier bound.
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class ReplacesRejected(ValueError):
    """A ``replaces`` value refused before anything was saved."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def as_error(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


def normalize_replaces(value: object) -> str:
    """Return the caller's id with surrounding spaces removed, or refuse it."""
    if not isinstance(value, str):
        raise ReplacesRejected(
            INVALID, "replaces must be the id of an earlier memory, sent as text.")
    text = value.strip()
    if not text:
        raise ReplacesRejected(
            INVALID, "replaces is empty. Pass the id of the memory this one "
                     "replaces, or leave it out.")
    if not _ID.fullmatch(text):
        raise ReplacesRejected(
            INVALID, "replaces does not look like a memory id. Use a fact_id or "
                     "memory_id returned by remember or recall.")
    return text


__all__ = ["INVALID", "NOT_ALLOWED", "NOT_FOUND", "ReplacesRejected", "normalize_replaces"]
