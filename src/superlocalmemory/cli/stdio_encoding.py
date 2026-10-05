# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Keep CLI output from crashing on streams that cannot encode it.

On Windows, when stdout or stderr is a pipe or a file, Python encodes with the
locale code page (usually cp1252). The CLI prints ``✓``, ``✗`` and ``→`` and
echoes stored memories, which can hold any character, so ``slm ... > out.txt``
or an installer capturing ``slm`` output died with ``UnicodeEncodeError``.

The policy, per stream, applied once at the top of ``main()``:

* Already able to encode the CLI's symbols, or already using a non-raising
  error handler: left alone.
* A terminal: keep its encoding, because that is what the terminal renders,
  and switch to ``errors="replace"`` so a missing glyph prints ``?``. (A
  Windows console is UTF-8 already, so this is POSIX terminals on a legacy
  locale.)
* ``PYTHONIOENCODING`` names an encoding: the caller chose it on purpose, the
  same precedence CPython gives it over UTF-8 mode, so keep it and replace.
* Otherwise, a pipe or file on a locale code page: write UTF-8. Nobody chose
  that code page, UTF-8 loses no memory text where ``replace`` would turn it
  into ``?``, and it is what PEP 686 makes Python's default. JSON output is
  ASCII (``ensure_ascii``), so its bytes are identical either way.
"""

from __future__ import annotations

import os
import sys

# The symbols CLI status lines print. A stream that cannot encode these would
# raise part-way through a command.
_SYMBOLS = "✓✗→"

# Error handlers that never raise on an unencodable character.
_NON_RAISING = frozenset(
    {"replace", "backslashreplace", "xmlcharrefreplace", "namereplace", "ignore"}
)


def _can_encode(encoding: str) -> bool:
    try:
        _SYMBOLS.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _is_tty(stream: object) -> bool:
    try:
        return bool(stream.isatty())  # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError):
        return False


def _make_safe(stream: object) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    encoding = getattr(stream, "encoding", None)
    errors = getattr(stream, "errors", None)
    if reconfigure is None or not encoding:
        return
    if errors in _NON_RAISING or _can_encode(encoding):
        return
    explicit = os.environ.get("PYTHONIOENCODING", "").partition(":")[0].strip()
    try:
        if _is_tty(stream) or explicit:
            reconfigure(errors="replace")
        else:
            reconfigure(encoding="utf-8", errors=errors)
    except (OSError, ValueError) as exc:
        # A closed stream, or one that refuses a change: output keeps its old
        # behaviour, which is no worse than before. Say so on stderr if we can.
        try:
            print(f"slm: could not adjust output encoding: {exc}", file=sys.stderr)
        except Exception:
            pass


def make_stdio_safe() -> None:
    """Make ``sys.stdout`` and ``sys.stderr`` unable to raise on encoding."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            _make_safe(stream)
