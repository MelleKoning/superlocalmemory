# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Memory content can leave the device two ways, not one: the long-documented
online answer check (sends the question and top memories to the chosen
provider), and — newer, and undocumented until this fix — the Jev
memory-kind classifier (sends a memory's content to Jev for typing). The
kind path is double-gated: it only runs when the answer check already uses
Jev (`sufficiency_jev_consent`), and only when the kind classifier's own,
separate `jev_consent` is also the literal boolean True
(encoding/memory_kind_classifier.py, core/memory_kind_config.py). Every
privacy statement describing "the one exception" was therefore false
(L3-18).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Every place the egress boundary is described to a human or an agent.
PRIVACY_STATEMENTS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "plugin" / "CLAUDE.md",
    REPO_ROOT / "plugin-src" / "rules" / "CLAUDE.md.fragment",
    REPO_ROOT / "copilot-plugin" / ".github" / "copilot-instructions.md",
)


def _mentions_kind_egress(text: str) -> bool:
    lowered = text.lower()
    return "jev" in lowered and "kind" in lowered and "consent" in lowered


def test_every_privacy_statement_names_the_jev_kind_egress() -> None:
    missing = [
        str(path.relative_to(REPO_ROOT))
        for path in PRIVACY_STATEMENTS
        if path.exists() and not _mentions_kind_egress(path.read_text(encoding="utf-8"))
    ]
    assert not missing, (
        f"these files describe the egress boundary but do not mention the "
        f"Jev memory-kind path: {missing}"
    )


def test_privacy_statements_do_not_overclaim_a_single_exception() -> None:
    """'the online answer check' described as *the* only exception is now
    false — catch any file still making that narrower claim."""
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in PRIVACY_STATEMENTS
        if path.exists() and "the one exception" in path.read_text(encoding="utf-8").lower()
    ]
    assert not offenders, f"these files still claim a single exception: {offenders}"
