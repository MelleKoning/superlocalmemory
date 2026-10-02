# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The pipe between the on-device answer check and its worker process.

JSON lines over stdin/stdout. Two things here that the judge relies on:

* Reads are taken straight from the file descriptor into this module's own
  buffer. A text wrapper's ``readline`` may pull two replies into its buffer
  at once, and a ``select`` on the descriptor then reports nothing to read
  while the second reply sits there — a recall would time out waiting for an
  answer that had already arrived. That matters now that a late reply to an
  abandoned request is drained instead of the worker being killed.
* Every request carries an id and every reply echoes it, so a late reply is
  recognised and discarded, never read as the answer to the next question.
"""

from __future__ import annotations

import json
import os
import selectors
import time

#: A single reply is a few hundred bytes; this bounds a misbehaving worker.
MAX_LINE_BYTES = 1_000_000

KIND_PERMANENT = "permanent"
KIND_TRANSIENT = "transient"

#: For a reply from a worker too old to say ``error_kind`` itself.
_PERMANENT_PREFIXES = ("ImportError", "ModuleNotFoundError", "FileNotFoundError",
                       "NotADirectoryError", "LocalEntryNotFoundError",
                       "RepositoryNotFoundError", "RevisionNotFoundError")


class LineReader:
    """Reads whole lines from one descriptor before a deadline."""

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._buffer = b""

    def readline(self, deadline: float) -> str | None:
        """The next line without its newline, or None at the deadline.

        Raises EOFError when the worker closed its end.
        """
        while b"\n" not in self._buffer:
            if len(self._buffer) > MAX_LINE_BYTES:
                raise ValueError("worker reply too long")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            with selectors.DefaultSelector() as selector:
                selector.register(self._fd, selectors.EVENT_READ)
                if not selector.select(timeout=remaining):
                    return None
            chunk = os.read(self._fd, 65536)
            if not chunk:
                raise EOFError("worker closed its output")
            self._buffer += chunk
        line, _, self._buffer = self._buffer.partition(b"\n")
        return line.decode("utf-8", errors="replace")


def write_request(stream, request: dict) -> None:
    """One request, whole, on the worker's stdin (raises OSError if it is gone)."""
    data = (json.dumps(request) + "\n").encode("utf-8")
    fd = stream.fileno()
    while data:
        written = os.write(fd, data)
        data = data[written:]


def failure_kind(reply: dict | None) -> str:
    """Whether a failed load could ever succeed by retrying the same setup."""
    if not isinstance(reply, dict):
        return KIND_TRANSIENT
    kind = reply.get("error_kind")
    if kind in (KIND_PERMANENT, KIND_TRANSIENT):
        return kind
    error = str(reply.get("error", ""))
    return KIND_PERMANENT if error.startswith(_PERMANENT_PREFIXES) else KIND_TRANSIENT


def cooldown_s(failures: int, *, base_s: float, max_s: float) -> float:
    """How long to wait before the next warm-up after ``failures`` in a row.

    The first failure is retried at once — one crash or one bad cycle is not a
    pattern — then the wait doubles from ``base_s`` up to ``max_s``.
    """
    if failures <= 1:
        return 0.0
    return min(max_s, base_s * (2 ** (failures - 2)))


__all__ = [
    "KIND_PERMANENT",
    "KIND_TRANSIENT",
    "LineReader",
    "MAX_LINE_BYTES",
    "cooldown_s",
    "failure_kind",
    "write_request",
]
