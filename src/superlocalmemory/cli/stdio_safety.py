# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Output that cannot be encoded never crashes an ``slm`` command.

When output goes to a pipe or a file on Windows, Python encodes it in the
code page (usually cp1252), which lacks characters SLM prints (✓ ✗ → …). A
command that had already done its work then died with ``UnicodeEncodeError``.

Such a stream keeps its encoding -- whoever reads that pipe decodes in it --
and a character it cannot carry becomes ``?``. A stream that can carry every
character (UTF-8, the Windows console, macOS and Linux) is left untouched.
"""

from __future__ import annotations

import sys
from typing import Any

#: Characters SLM's own messages use; a stream that encodes these is safe.
_PROBE = "✓✗→…—"


def make_unencodable_output_safe(stream: Any) -> None:
    """Replace instead of raise on ``stream`` if it cannot encode SLM's output."""
    encoding = getattr(stream, "encoding", None)
    reconfigure = getattr(stream, "reconfigure", None)
    if not encoding or reconfigure is None:
        return  # no stream (pythonw) or not a text wrapper: nothing to change
    try:
        _PROBE.encode(encoding)
        return
    except UnicodeEncodeError:
        pass
    except LookupError:
        return  # an encoding Python does not know; leave it to fail loudly
    reconfigure(errors="replace")


def protect_standard_streams() -> None:
    """Apply :func:`make_unencodable_output_safe` to stdout and stderr."""
    make_unencodable_output_safe(sys.stdout)
    make_unencodable_output_safe(sys.stderr)


__all__ = ["make_unencodable_output_safe", "protect_standard_streams"]
