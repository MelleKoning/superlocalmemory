# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""One safe way to put a user-supplied value into a daemon HTTP path.

Issue #148: ``slm ops resolve`` interpolated a raw CLI argument into
``f"/operations/{operation_id}/resolve"`` and handed that straight to
``http.client``. A non-ASCII ID (a pasted placeholder, a typo) survives
string formatting untouched, then blows up deep inside ``http.client``'s
request-line encoding (``str.encode('ascii')``) with an uncaught
``UnicodeEncodeError`` — a raw traceback instead of a CLI error.

Every CLI call site that builds a daemon path from a user-supplied string
must go through one of the two helpers below instead of raw f-string
interpolation:

* :func:`validate_daemon_id` — for values the *daemon* itself generates or
  constrains to a fixed shape (operation IDs, memory-kind backfill run IDs,
  profile names: all ``uuid.uuid4().hex``-shaped or
  ``^[a-zA-Z0-9_-]+$``-validated server-side already). A value outside that
  shape cannot be a real ID, so it is rejected before it ever reaches a
  socket, with a friendly, documented error.
* :func:`quote_path_segment` — for values that are legitimately free-form
  (nothing server-side constrains their charset) and must still reach the
  daemon, e.g. a memory/fact ID a caller wants to echo back unmodified.
  Percent-encoding always succeeds, so this never rejects input.

Either helper keeps a valid, already-ASCII value byte-for-byte identical
on the wire: ``validate_daemon_id`` returns it unchanged, and
``quote_path_segment``/``urllib.parse.quote`` is a no-op for characters in
its default unreserved set.
"""

from __future__ import annotations

import re
import urllib.parse

# Every daemon-issued ID this CLI round-trips (operation IDs, backfill run
# IDs) is `uuid.uuid4().hex` (or `.hex[:16]`) -- lowercase hex, no separators.
# Profile names are validated server-side (see
# ``superlocalmemory.server.routes.helpers.validate_profile_name``) against
# the slightly wider ``^[a-zA-Z0-9_-]+$``. One pattern covers both: anything
# a real ID can be, plus nothing a real ID cannot be.
_DAEMON_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class InvalidDaemonId(ValueError):
    """A CLI argument does not look like a value the daemon would issue.

    ``label`` names the kind of value (``"operation ID"``, ``"run ID"``,
    ``"profile name"``) so each call site can phrase its own one-line error;
    ``value`` is the raw, rejected input.
    """

    def __init__(self, label: str, value: str) -> None:
        self.label = label
        self.value = value
        super().__init__(f"invalid {label} {describe(value)}")


def describe(value: str) -> str:
    """An ASCII-safe, printable rendering of ``value`` for an error message.

    ``ascii()`` backslash-escapes anything outside printable ASCII (a
    literal U+2026, a control character) instead of emitting it raw, so the
    error message itself can never raise ``UnicodeEncodeError`` on a
    narrow/ASCII stream -- the exact failure mode this module exists to
    avoid in the first place.
    """
    return ascii(value)


def validate_daemon_id(value: str, *, label: str) -> str:
    """Return ``value`` unchanged if it matches the daemon's ID shape.

    Raises :class:`InvalidDaemonId` otherwise. Call this before
    interpolating ``value`` into any daemon request path; never pass an
    unvalidated value to an f-string path.
    """
    if not isinstance(value, str) or not _DAEMON_ID_RE.match(value):
        raise InvalidDaemonId(label, value)
    return value


def quote_path_segment(value: str) -> str:
    """Percent-encode ``value`` for use as one path segment.

    Unlike :func:`validate_daemon_id` this never rejects input -- every
    string is a legitimate path segment once encoded, so it never produces
    an "invalid ..." error. Use it only where the value is not constrained
    to the daemon's ID shape (nothing to validate against) but must still
    reach the daemon.
    """
    return urllib.parse.quote(str(value), safe="")


__all__ = [
    "InvalidDaemonId",
    "describe",
    "quote_path_segment",
    "validate_daemon_id",
]
