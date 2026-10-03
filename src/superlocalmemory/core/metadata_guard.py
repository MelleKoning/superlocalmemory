# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Strip internally-reserved ``_slm_*`` keys from caller-supplied metadata.

Several metadata keys are reserved for the system's own bookkeeping —
``_slm_memory_kind`` (``storage.memory_kinds.METADATA_KEY``), plus any other
``_slm_*``-prefixed marker a future internal feature adds. A caller's free-form
``metadata`` dict must never be allowed to set one of these directly: letting
``_slm_memory_kind`` through lets a plain HTTP ``POST /remember`` forge a
CONFIRMED memory kind (including a standing rule) through a door that has
none of the memory-kind routes' permission checks (L3-13) — a trusted
``source_type`` (``kind_assignment.KIND_TRUSTED_SOURCES`` includes "http")
reads ``metadata["_slm_memory_kind"]`` as the caller's own declared, confirmed
choice, with no further validation of how it got there.

The *declared* ``kind`` field is unaffected: callers still set a kind the
documented way (``remember(kind=...)``, the HTTP ``RememberRequest.kind``
field), which is parsed, validated, and only then written into this same
metadata slot by the route itself -- after this stripping has already run.
"""

from __future__ import annotations

_RESERVED_PREFIX = "_slm_"


def strip_reserved_metadata(metadata: dict | None) -> dict:
    """A new dict with every ``_slm_*``-prefixed key removed.

    ``None`` or empty input becomes an empty dict. Never mutates the input.
    """
    if not metadata:
        return {}
    return {
        key: value for key, value in metadata.items()
        if not str(key).startswith(_RESERVED_PREFIX)
    }


__all__ = ["strip_reserved_metadata"]
