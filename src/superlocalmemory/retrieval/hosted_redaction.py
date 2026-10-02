# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Credential redaction for text crossing the hosted-judge trust boundary.

Jev's providers (TypeSafe, OpenRouter) see this text over the network.
Reuses SLM's own secret scanner — ``core.security_primitives.redact_secrets``
at its highest aggression setting, the same one already guarding every other
outbound LLM prompt in this codebase (see
``retrieval.remote_reranker._redact_remote_text``) — then strips the
type/last-four suffix that scanner keeps for local logs. That suffix is
useful to an operator reading logs on this machine; it is four more
characters of a real key than should ever leave the machine, so none of it
survives here. Every recognized credential becomes the literal marker
``[redacted]`` and nothing else.

What counts as a credential (``core.credential_shapes``): known key and token
formats; a password inside a connection URL; a value after a credential label
(``password=…``, ``"api_key": "…"``, "my password is …"); a hex or UUID key
after a key-like label. A commit id, a checksum or a UUID used as a record id,
with no such label in front of it, is ordinary text and is sent as it is.

This module only screens for credential-shaped tokens. It does not redact
PII (email, phone, etc.) — that is a separate, broader policy this call site
was not asked to apply, and applying it silently here would change what the
hosted judge is being asked to read without being asked to.
"""

from __future__ import annotations

import re

from superlocalmemory.core.security_primitives import redact_secrets

#: What ``redact_secrets`` emits locally: ``[REDACTED:TYPE:last4]``. Collapsed
#: to the bare marker below before anything leaves the machine.
_LOCAL_MARKER_RE = re.compile(r"\[REDACTED:[A-Z_]+:[^\]]*\]")

#: The only marker a hosted provider ever sees in place of a credential.
REDACTED_MARKER = "[redacted]"


def redact_for_hosted_judge(text: str) -> str:
    """Return ``text`` with every recognized credential replaced by ``[redacted]``.

    Never raises; a non-string input returns ``""``. Whole-token replacement
    only — no prefix, suffix, or length hint of the original value survives.
    """
    if not isinstance(text, str) or not text:
        return ""
    scanned = redact_secrets(text, aggression="high")
    return _LOCAL_MARKER_RE.sub(REDACTED_MARKER, scanned)


def redact_or_none(text: str) -> str | None:
    """Redact ``text``; return None when nothing but markers/whitespace remains.

    Text that WAS only a credential becomes just ``[redacted]`` once
    screened — that marker carries no information for the hosted judge to
    read, so it is treated the same as an empty value: refuse to send it
    rather than ask a question about a placeholder.
    """
    redacted = redact_for_hosted_judge(text).strip()
    if not redacted.replace(REDACTED_MARKER, "").strip():
        return None
    return redacted


__all__ = ["REDACTED_MARKER", "redact_for_hosted_judge", "redact_or_none"]
