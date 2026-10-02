# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``memory_protocol_markdown()`` is the single shared body embedded into
Cursor's ``.cursor/rules/slm-active-brain.mdc``, the runtime-synced
``.github/copilot-instructions.md`` (CopilotAdapter), and Antigravity's
``slm-memory-adapter`` skill (AntigravityAdapter) — see cursor_adapter.py,
copilot_adapter.py and antigravity_adapter.py, all three of which call this
one function for their body text. None of the three hosts had any rule
telling the agent that a recalled memory can come back abstained. Fixing
this one function fixes all three at once.
"""

from __future__ import annotations

from superlocalmemory.hooks.memory_protocol import memory_protocol_markdown


def test_protocol_teaches_the_abstained_rule():
    protocol = memory_protocol_markdown()
    assert "abstained" in protocol
    assert "answer_confidence" in protocol
    # The substance of the rule, not just the field names.
    assert "do not answer" in protocol.lower() or "does not answer" in protocol.lower()


def test_protocol_still_has_no_internal_or_competitor_branding():
    protocol = memory_protocol_markdown()
    for banned in ("Qualixar", "Anthropic", "OpenAI"):
        assert banned not in protocol
