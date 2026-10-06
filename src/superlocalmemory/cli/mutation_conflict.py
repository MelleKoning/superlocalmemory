# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Print a daemon's 409 as the decision it is, not as an outage.

A live daemon refusing a change (for example, deleting a memory that correction
history protects) answered HTTP 409, and ``slm delete`` printed
DAEMON_UNAVAILABLE with a hint to restart a daemon that was working fine.
"""

from __future__ import annotations

import sys


def exit_conflict(command: str, reason: str, use_json: bool) -> None:
    """Exit 1 with the daemon's own reason. Not retryable: it will say the same."""
    message = reason or "the daemon refused this change"
    if use_json:
        from superlocalmemory.cli.json_output import json_print

        json_print(command, error={
            "code": "CONFLICT",
            "message": message,
            "retryable": False,
        })
    else:
        print(f"Refused: {message}", file=sys.stderr)
    raise SystemExit(1)


__all__ = ["exit_conflict"]
