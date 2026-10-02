# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The one line every surface adds about the answer check's verdict.

The line is added to the memories, never put in their place: abstention is a
signal for the reader, not a reason to hide what was retrieved. Every
auto-injection surface (``session_init``, AutoRecall behind ``slm://context``
and ``slm session-context --full``, the per-prompt hook) uses this function,
so they say the same thing in the same words.

"" when no judge ran (``calibration_status == "uncalibrated"``), so output is
byte-identical to before for everyone who never turned the check on.
Stdlib only — the hook import chain must stay light.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_NOT_ANSWERED = (
    "Answer check: none of these memories answers the question "
    "(confidence {confidence:.2f}). Say you don't have it, or ask "
    "— don't present these as the answer."
)
_ANSWERED = "Answer check: likely answered (confidence {confidence:.2f})."


def _reader(result: Any):
    if isinstance(result, Mapping):
        return result.get
    return lambda key, default=None: getattr(result, key, default)


def answer_check_line(result: Any) -> str:
    """One plain line for a recall response (object or dict), or ""."""
    get = _reader(result)
    if (get("calibration_status", "uncalibrated") or "uncalibrated") == "uncalibrated":
        return ""
    raw_confidence = get("answer_confidence", None)
    try:
        confidence = float(raw_confidence) if raw_confidence is not None else 0.0
    except (TypeError, ValueError):
        confidence = 0.0
    if get("abstention_reason", None) == "judged_insufficient":
        return _NOT_ANSWERED.format(confidence=confidence)
    if not get("abstained", False):
        return _ANSWERED.format(confidence=confidence)
    return ""


__all__ = ["answer_check_line"]
