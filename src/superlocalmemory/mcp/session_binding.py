# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Work out which session a tool call belongs to, the same way every time.

``recall`` has resolved this through a four-step ladder since S9-DASH-10 — the
explicit argument, then the environment, then the hook registry, then a stable
per-agent fallback — so an engagement signal lands on the right pending outcome.
``remember`` never had it. It took ``session_id: str = ""`` and stored whatever
it was handed, which for a caller that does not pass one is nothing.

Measured on the author's store: **192 of 3,894 genuine facts carry a session_id
(4.9%)**, and 4 of the 200 most recent (2%). Every one of the rest was written
through a path that could have known and did not.

That is not bookkeeping. ``RetrievalEngine`` promotes results so the top of an
answer spans more than one session (its ``sessions_in_top`` pass), and a fact
with no session_id can never be promoted by it — so the diversity mechanism was
running against a corpus where 95% of rows were indistinguishable. It also means
"what did we discuss in that session" has almost nothing to match on.

One implementation, called by both tools, so the read path and the write path
cannot disagree about which session they are in.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

__all__ = ["resolve_session_id", "SESSION_ENV_VARS"]

#: Checked in order. Hosts set one or the other; SLM's own takes precedence so a
#: user can override a host that sets its variable to something unhelpful.
#: ``CLAUDE_CODE_SESSION_ID`` is the actual variable Claude Code's MCP
#: subprocess environment carries — verified on a live daemon 2026-08-24,
#: this pipeline settled 6013/6039 outcomes at the neutral 0.5 fallback
#: because the old two-name list never matched it, and every recall fell
#: through to the synthetic ``mcp:<agent_id>`` id that step 4 deliberately
#: excludes from matching.
SESSION_ENV_VARS = ("SLM_SESSION_ID", "CLAUDE_SESSION_ID", "CLAUDE_CODE_SESSION_ID")


def resolve_session_id(
    explicit: str = "",
    *,
    agent_id: str = "unknown",
    allow_agent_fallback: bool = True,
) -> str:
    """Best available session id for this call. Never raises.

    Order, most to least specific:

      1. ``explicit`` — what the caller passed. Always wins.
      2. ``SLM_SESSION_ID`` / ``CLAUDE_SESSION_ID`` from the environment.
      3. The hook registry: the session whose parent process is ours, else the
         most recently active one inside 60 seconds. Parent-PID lookup is
         collision-free across parallel host sessions, because each MCP
         server's parent is the editor that spawned it.
      4. ``mcp:<agent_id>`` — stable per agent, and deliberately NOT matched by
         the Stop hook, so the reaper settles those outcomes at a neutral 0.5
         rather than crediting or blaming a session that never existed.

    ``allow_agent_fallback=False`` stops before step 4 and returns "". Use it
    where a synthetic id would be worse than none: grouping memories under
    ``mcp:<agent>`` would put every memory an agent ever wrote into one bucket
    and make session-diversity promotion rank them as a single session, which
    is the opposite of what it is for.
    """
    if explicit and explicit.strip():
        return explicit.strip()

    # Steps 2 and 3 describe sessions on this computer. A caller on another
    # computer or a web app is not one of them, so it skips straight to step 4.
    from superlocalmemory.mcp.remote_caller import current_remote_key_id

    if current_remote_key_id() is not None:
        return f"mcp:{agent_id}" if allow_agent_fallback else ""

    for name in SESSION_ENV_VARS:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()

    try:
        from superlocalmemory.hooks.session_registry import (
            lookup_by_parent,
            most_recent_active,
        )

        found = (
            lookup_by_parent(within_seconds=60)
            or most_recent_active(agent_type="claude", within_seconds=60)
            or ""
        )
        if found:
            return found
    except Exception as exc:  # noqa: BLE001 — a hint must never fail a call
        logger.debug("session registry lookup unavailable: %s", exc)

    if allow_agent_fallback:
        return f"mcp:{agent_id}"
    return ""
