# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A short, path-free reason an embedding warmup attempt failed.

4.1.22: a first-run Mode A embedder with no cached model and no network
(``HF_HUB_OFFLINE``) exhausts ``unified_daemon``'s warmup retries and never
becomes warm on its own. The retry loop used to log the exception at DEBUG
and otherwise discard it, so ``/health`` and ``slm warmup`` kept describing
a permanently stuck daemon exactly like one that was merely a few seconds
from ready. This formats that exception once, kept out of its own module so
``unified_daemon.py`` (already over this codebase's per-file size guidance)
does not grow for a function with no dependency on the rest of it.
"""

from __future__ import annotations

from pathlib import Path

#: /health shares loopback-adjacent detail with other local accounts and
#: unauthenticated remote callers (see unified_daemon's own note on that),
#: so the reason kept here is bounded and path-free.
_MAX_LENGTH = 200


def sanitized_warmup_error(exc: BaseException) -> str:
    """The exception type and message, with this machine's home directory
    replaced and the result truncated.

    huggingface_hub's own offline/no-cache message is already generic
    ("Cannot find the requested files in the disk cache and outgoing
    traffic has been disabled"); this only guards against whatever a less
    careful provider might include.
    """
    text = f"{type(exc).__name__}: {exc}"
    try:
        home = str(Path.home())
        if home and home in text:
            text = text.replace(home, "<HOME>")
    except Exception:
        pass
    return text[:_MAX_LENGTH]
