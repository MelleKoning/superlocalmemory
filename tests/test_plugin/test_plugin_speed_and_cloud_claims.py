# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Agent guidance may not promise what the online answer check makes false.

The rules and skills agents read said recall never makes a server-side LLM
round and that SLM makes no cloud calls. Both stop being true the moment the
user turns on the online answer check, so neither may be stated without
saying so — in the sources and in every generated plugin tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TREES = ("plugin-src", "plugin", "codex-plugin", "copilot-plugin", "antigravity-plugin",
         "hermes-plugin")
STALE = (
    "no server-side LLM round on the hot path)",
    "(~1–2s, no server-side LLM round)",
    "All tools run on the user's machine; no cloud calls.",
)


def _documents():
    for tree in TREES:
        base = ROOT / tree
        if base.is_dir():
            yield from (p for p in base.rglob("*") if p.is_file()
                        and p.suffix in {".md", ".fragment", ".json", ".txt"})


@pytest.mark.parametrize("phrase", STALE)
def test_no_agent_guidance_makes_an_unqualified_promise(phrase):
    offenders = [str(p.relative_to(ROOT)) for p in _documents()
                 if phrase in p.read_text(encoding="utf-8", errors="replace")]
    assert not offenders, f"{phrase!r} still in {offenders}"


def test_the_rules_say_when_a_recall_waits_on_a_service():
    text = (ROOT / "plugin-src" / "rules" / "AGENTS.md").read_text(encoding="utf-8")
    assert "online answer check" in text
