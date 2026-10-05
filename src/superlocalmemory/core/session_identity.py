# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Which session ids name a conversation, and which were invented for one call.

TWO DIFFERENT JOBS, ONE STRING
------------------------------
A session id is used for two unrelated things:

1. **Bookkeeping.** A recall needs *some* handle so a downstream reference can be
   traced back to it. Any unique string does. When a caller supplies none, the
   HTTP and MCP fronts invent one — ``http:<milliseconds>`` per request, or
   ``mcp:<agent_id>`` per client.

2. **Continuity.** The working set carries what a conversation was recently
   shown into its next turn. That needs an id that actually identifies a
   conversation, because it decides what gets ranked higher.

An invented id is fine for the first and wrong for the second, and the two are
wrong in opposite directions:

* ``http:<ms>`` is unique per request, so every dashboard search registered a
  new working set. Enough of them and the registry hits its cap and evicts the
  least-recently-touched entry — which is a real conversation sitting idle
  between turns. The next turn of that conversation is cold, with no error.
* ``mcp:<agent_id>`` is SHARED by every client that did not send an id, so two
  unrelated clients pooled one seven-slot set and promoted each other's
  memories.

Neither mattered while the parameter was ignored, which it was until continuity
was built on it. That is what made this easy to miss.

ONE DEFINITION
--------------
Both the creators and the reader use this module, so a new front cannot invent a
third prefix that continuity silently accepts. Two lists of the same thing is how
one ends up wrong.
"""

from __future__ import annotations

__all__ = [
    "SYNTHETIC_PREFIXES",
    "is_conversation",
    "synthetic_session_id",
]

#: Prefixes marking an id that a front invented rather than received.
#:
#: The colon is deliberate: a real client id is a uuid or a hex string and does
#: not contain one, so a genuine id cannot be mistaken for an invented one.
#: ``engine:`` names the daemon process itself, not a caller.  It is shared by
#: every client that reaches one daemon and changes on every restart, so it is
#: synthetic in exactly the way this module exists to catch: it was minted by
#: hand in ``core/engine.py`` rather than through ``synthetic_session_id``, and
#: so passed ``is_conversation`` for as long as it existed.  While it did, every
#: recall that named no session filed its outcome under a process id that no
#: tool event could ever carry, and the reward pipeline had nothing to join on.
#: ``agent:`` and ``api:`` name the calling agent and the workspace, and are
#: shared by every request from either — ``agent:mcp_client`` alone held 24
#: outcomes from unrelated callers. Both were hand-minted in
#: ``server/routes/v3_api.py``, which is why neither was listed here.
#: ``view:`` (4.1.21) names a saved view being run. A view is a fixed question,
#: not a conversation: continuity must not bias its second run toward what its
#: first run showed, and a burst of view runs must not take working-set slots
#: from real conversations.
SYNTHETIC_PREFIXES: tuple[str, ...] = (
    "http:", "mcp:", "cli:", "probe:", "engine:", "agent:", "api:", "view:",
)


def synthetic_session_id(kind: str, discriminator: str = "") -> str:
    """Build an id for bookkeeping that continuity will correctly ignore.

    ``kind`` names the front that invented it, so a log line says where an
    unattributed recall came from.
    """
    prefix = kind if kind.endswith(":") else f"{kind}:"
    return f"{prefix}{discriminator}"


def is_conversation(
    session_id: str | None, profile_id: str | None = None,
) -> bool:
    """Whether this pair identifies a conversation across turns.

    False for an empty id and for anything a front invented. Continuity engages
    only when this is True, so the default for an unidentified caller is the
    behaviour that existed before continuity: every recall starts cold.

    ``profile_id`` is checked too when given. The working set is keyed on both,
    and an empty profile is not a profile: ``None`` and ``""`` would normalise to
    the same key, so two unidentified callers sharing a session id would share
    one set. Every caller in this codebase resolves a real profile before
    reaching here, which is exactly why the aliasing would go unnoticed if it
    ever stopped being true.
    """
    if not session_id:
        return False
    if profile_id is not None and not profile_id:
        return False
    return not session_id.startswith(SYNTHETIC_PREFIXES)
