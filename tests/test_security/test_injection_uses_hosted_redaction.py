# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Auto-injected context carries no credential, and no piece of one.

Credentials stay in SLM and an explicit recall returns them. What SLM pushes
into an agent's context on its own — session start, standing rules, the hooks,
the pre-staged context tool, the IDE context files and the dashboard chat — is
screened at the same strength as text leaving the machine: every recognised
credential becomes ``[redacted]``, with no type label and no last four
characters. The weaker storage-time scan left four of these nine samples
verbatim and kept the tail of the rest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.core.injection import (
    InjectableMemory,
    render_context,
    sanitize_untrusted_content,
)

#: (sample as a memory holds it, the part that must not appear)
SAMPLES: tuple[tuple[str, str], ...] = (
    ("db password=Hunt3rTwo!x9", "Hunt3rTwo"),
    ("my password is Hunt3rTwo!x9", "Hunt3rTwo"),
    ("Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345", "abcdefghijklmnop"),
    ('{"api_key": "q8w7e6r5t4y3u2i1o0p9"}', "q8w7e6r5"),
    ("postgres://admin:S3cretPass99@db.example.com:5432/app", "S3cretPass99"),
    ("AWS AKIAABCDEFGHIJKLMNOP", "AKIAABCDEFGH"),
    ("token ghp_" + "a" * 36, "ghp_aaaa"),
    ("SLM_SECRET_KEY_2024=abcd1234efgh5678", "abcd1234efgh5678"),
    ("sk-proj-A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0", "A1b2C3d4"),
)

#: A tail the weaker scan used to keep for each token-shaped sample.
TAILS = ("2345", "MNOP", "aaaa", "S9t0")


@pytest.mark.parametrize(("text", "secret"), SAMPLES, ids=[s[1] for s in SAMPLES])
def test_sanitized_context_holds_no_credential(text: str, secret: str) -> None:
    out = sanitize_untrusted_content(text)
    assert secret not in out
    assert "[redacted]" in out
    assert "[REDACTED:" not in out


def test_no_last_four_characters_survive_injection() -> None:
    rendered = render_context(
        [InjectableMemory(content=t, score=0.9, fact_id=f"f{i}")
         for i, (t, _s) in enumerate(SAMPLES)],
        mode="B", cfg=None, wrap=True,
    )
    for _text, secret in SAMPLES:
        assert secret not in rendered
    for tail in TAILS:
        assert f":{tail}]" not in rendered
    assert "[REDACTED:" not in rendered


def test_ordinary_memory_text_is_injected_as_written() -> None:
    text = "Atlas ships on Friday; the API key rotation policy is weekly."
    assert sanitize_untrusted_content(text) == text


def test_the_prestaged_context_tool_uses_the_strong_screen() -> None:
    from superlocalmemory.mcp.tools_context import _cap_memory

    for text, secret in SAMPLES:
        out = _cap_memory({"id": "1", "text": text, "score": 0.5})["text"]
        assert secret not in out and "[REDACTED:" not in out


def test_ide_context_files_use_the_strong_screen() -> None:
    from superlocalmemory.hooks.context_payload import build_payload

    def _recall(query, limit, profile_id):
        return [{"text": t, "name": t, "score": 0.5} for t, _s in SAMPLES]

    payload = build_payload("default", "project", Path("."), recall_fn=_recall)
    blob = repr(payload)
    for _text, secret in SAMPLES:
        assert secret not in blob
    assert "[REDACTED:" not in blob


def test_the_dashboard_chat_context_uses_the_strong_screen() -> None:
    """Mode B/C chat builds its prompt with ``render_context``; Mode A prints
    ``sanitize_untrusted_content`` — both are covered above. This pins that the
    chat route still uses them rather than a weaker path of its own."""
    from superlocalmemory.server.routes import chat

    assert chat.render_context is render_context
    assert chat.sanitize_untrusted_content is sanitize_untrusted_content
