# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""One HTTP body for "that is not a memory kind" everywhere (L3-12).

Before this, a caller who sent a bad ``kind`` saw three different 422
shapes depending on which route answered: a plain string, ``{"error":
"invalid_kind", ...}``, or ``"Not a memory kind."`` with no machine-readable
code at all. A client written against one of them broke on the others.

Every HTTP door now raises the same body:
``{"detail": {"code": "INVALID_KIND", "message": "..."}}``, 422. MCP keeps
its own ``{"success": False, "code": "INVALID_KIND", ...}`` envelope (that is
MCP's own transport shape, not this one) and the CLI exits 2 either way.
"""

from __future__ import annotations

from fastapi import HTTPException

from superlocalmemory.core.kind_query import InvalidKind


def invalid_kind_http(exc: InvalidKind) -> HTTPException:
    """The one 422 every HTTP route raises for a kind that failed to parse."""
    return HTTPException(422, detail={"code": "INVALID_KIND", "message": str(exc)})


__all__ = ["invalid_kind_http"]
