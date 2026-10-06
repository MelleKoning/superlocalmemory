# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The kind a caller declares on the ``remember`` tool, and how it travels.

A declared kind reaches the daemon ONLY as the ``/remember`` request's own
``kind`` field: that is the one place the daemon accepts a confirmed kind from
(it validates it, then sets the trusted declaration itself). It must never
travel inside free-form metadata. The daemon strips every reserved ``_slm_*``
key a caller sends (``core/metadata_guard.py``), so a kind carried that way is
dropped - which is exactly how 4.1.21 lost every kind declared through the
tool interface while the same request over plain HTTP kept it.

The kind is also part of what was asked. The key the tool derives for a call
without one includes it, so the same words saved as a different kind are a new
request; and a caller's own key reused with a different kind changes the
daemon's request hash, so the daemon refuses it rather than returning the
first memory as if the new kind had been recorded.
"""

from __future__ import annotations

from typing import Any


def parse_declared_kind(kind: str | None) -> tuple[str, dict[str, Any] | None]:
    """``(value, None)`` for a known kind, ``("", None)`` for none, or
    ``("", error reply)`` for an unknown one (refused before anything is sent)."""
    if not (kind or "").strip():
        return "", None
    from superlocalmemory.storage.memory_kinds import MemoryKind, parse_kind

    parsed = parse_kind(kind)
    if parsed is None:
        return "", {
            "success": False,
            "code": "INVALID_KIND",
            "retryable": False,
            "error": "Unknown memory kind. Use one of: "
                     + ", ".join(k.value for k in MemoryKind),
        }
    return parsed.value, None


def kind_key_part(declared: str) -> str:
    """The derived-idempotency-key material for ``declared``.

    Empty when no kind was declared, so every key derived for a call without
    one is byte-identical to before.
    """
    return f"\0kind={declared}" if declared else ""


def with_declared_kind(
    body: dict[str, Any], flags: dict[str, Any], declared: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """New ``(body, request flags)`` carrying ``declared`` as the request field.

    Only when a kind was declared, so a plain call stays byte-identical. The
    daemon answers a request it refuses outright (an idempotency key already
    used for a different kind) with 422; ``preserve_unprocessable`` surfaces
    that refusal instead of letting it read as an outage and be retried.
    """
    if not declared:
        return body, flags
    return {**body, "kind": declared}, {**flags, "preserve_unprocessable": True}


def store_kwargs(declared: str) -> dict[str, str]:
    """Keyword arguments that pass ``declared`` to ``DaemonPoolProxy.store``."""
    return {"kind": declared} if declared else {}


__all__ = ["kind_key_part", "parse_declared_kind", "store_kwargs", "with_declared_kind"]
