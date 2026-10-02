# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Carry every recall metadata field from the daemon to MCP callers.

The daemon's ``/recall`` envelope spreads ``recall_response_metadata`` into
its top level. The MCP tools used to copy a hand-picked subset of it, so
fields added later (who chose the final order, abandoned channels, the
temporal frame) never reached an agent. This module asks
``recall_response_metadata`` itself which fields exist, every call, so a field
added there reaches MCP with no other change.

A field the daemon did not send (an older daemon still running after an
upgrade) gets the value ``recall_response_metadata`` gives an empty recall.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)


class _EmptyRecall:
    """What ``recall_response_metadata`` sees for a recall with nothing in it."""

    results: tuple = ()


class _UnknownRecall(_EmptyRecall):
    """Fallback when a new field has no default: every attribute is None."""

    def __getattr__(self, _name: str) -> None:
        return None


def metadata_defaults() -> dict[str, Any]:
    """Every metadata field, with its value for an empty recall."""
    from superlocalmemory.server import recall_serializer

    try:
        return dict(recall_serializer.recall_response_metadata(_EmptyRecall()))
    except Exception as exc:  # noqa: BLE001 — a field without a default
        logger.debug("recall metadata defaults fell back to None: %s", type(exc).__name__)
        return dict(recall_serializer.recall_response_metadata(_UnknownRecall()))


def forward_recall_metadata(raw: Any) -> dict[str, Any]:
    """The metadata fields of a daemon envelope (dict) or a response object.

    Only fields ``recall_response_metadata`` defines are taken, so nothing the
    HTTP envelope does not already return can appear here.
    """
    defaults = metadata_defaults()
    if isinstance(raw, Mapping):
        return {key: raw.get(key, default) for key, default in defaults.items()}
    carried = getattr(raw, "metadata", None)
    if isinstance(carried, Mapping) and carried:
        return {key: carried.get(key, default) for key, default in defaults.items()}
    return {key: getattr(raw, key, default) for key, default in defaults.items()}


__all__ = ["forward_recall_metadata", "metadata_defaults"]
