# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Cue rules for the nine memory kinds: the always-available, zero-cost suggestion.

The persona examples are the 27 of the 4.1.19 research note (three per kind,
across personas). The bar is 24 of 27, and misses are recorded, not tuned
away: the rules are a fallback suggestion, and a rule set that scores 27/27 on
the very examples it was written against says nothing about real memories.
"""

from __future__ import annotations

import pytest

from superlocalmemory.encoding.memory_kind_rules import (
    MAX_RULE_CHARS,
    RULES_RECIPE,
    suggest_by_rules,
)
from superlocalmemory.storage.memory_kinds import KindAssignment, KindSource, MemoryKind

K = MemoryKind

PERSONA_EXAMPLES: list[tuple[MemoryKind, str]] = [
    (K.SEMANTIC, "The billing service stores money as integer cents in Postgres 16."),
    (K.SEMANTIC, "The lab's mass spectrometer is specified to 0.5 ppm."),
    (K.SEMANTIC, "Our fiscal year starts on 1 April."),
    (K.EPISODIC, "On 3 Sep the staging deploy failed because the DNS TTL was 24 h."),
    (K.EPISODIC, "The 'MCP in five minutes' video passed 10k views two days after posting."),
    (K.EPISODIC, "Met Priya on Tuesday; she agreed to own the on-call rota."),
    (K.STATUS, "Migration branch: 41 of 52 tests passing, blocked on FTS trigger replay."),
    (K.STATUS, "Q4 hiring: 3 of 5 roles filled."),
    (K.STATUS, "Run 7 is still training; loss 0.42 after 18 h."),
    (K.OPINION, "GraphQL federation is more trouble than it is worth for teams under 20."),
    (K.OPINION, "Short punchy hooks beat long intros for this channel."),
    (K.OPINION, "I prefer pytest fixtures to unittest setUp."),
    (K.RULE, "Never run rm -rf with ~ or a glob in this repo."),
    (K.RULE, "Every new service must expose /healthz and log structured JSON."),
    (K.RULE, "Status reports go out every Friday by 5 pm."),
    (K.DECISION, "We chose Postgres over DynamoDB for billing because we need "
                 "multi-row transactions."),
    (K.DECISION, "Decided to move the EU launch to Q2 to finish the GDPR review."),
    (K.DECISION, "Picked XGBoost over the LSTM: a third of the cost, same AUC."),
    (K.PROCEDURE, "To release: bump the version in pyproject, run make test, then npm "
                  "publish from dist/."),
    (K.PROCEDURE, "Upload flow: export 4K, caption in Descript, schedule in Buffer for "
                  "9 am IST."),
    (K.PROCEDURE, "Calibrate: warm up 30 min, run a blank, then the standard mix three "
                  "times."),
    (K.PROSPECTIVE, "Draft the Q1 OKRs by 15 Dec."),
    (K.PROSPECTIVE, "TODO: add retry with backoff to the webhook client next sprint."),
    (K.PROSPECTIVE, "Board review on 20 Nov — prepare the cost slide."),
    (K.CORRECTION, "Correction: the API limit is 100 per minute, not 1,000 as noted "
                   "earlier."),
    (K.CORRECTION, "The earlier note was wrong — the sample was stored at -80 C, not "
                   "-20 C."),
    (K.CORRECTION, "Correction to yesterday's memo: the price is $49, not $39."),
]

#: The bar from work_packages.md WP-3. Raising it by tuning cues to these very
#: sentences is forbidden; record the miss instead.
MIN_PERSONA_MATCHES = 24


def test_persona_examples_match() -> None:
    misses = [(want.value, got.kind.value, text)
              for want, text in PERSONA_EXAMPLES
              if (got := suggest_by_rules(text, "semantic")).kind is not want]
    hits = len(PERSONA_EXAMPLES) - len(misses)
    assert len(PERSONA_EXAMPLES) == 27
    assert hits >= MIN_PERSONA_MATCHES, f"{hits}/27 — misses: {misses}"


@pytest.mark.parametrize(("want", "text"), PERSONA_EXAMPLES,
                         ids=[f"{w.value}-{i % 3}" for i, (w, _) in enumerate(PERSONA_EXAMPLES)])
def test_every_answer_is_a_rules_suggestion(want: MemoryKind, text: str) -> None:
    got = suggest_by_rules(text, "semantic")
    assert isinstance(got, KindAssignment)
    assert got.source is KindSource.RULES
    assert got.confidence is None
    assert got.recipe == RULES_RECIPE


def test_long_text_returns_legacy_kind() -> None:
    # Cue-rich, but over the limit: rules scored 0.17 on long text, so the
    # legacy kind is the honest answer.
    text = ("Never run rm -rf. Correction: we decided to go with Postgres. " * 20)
    assert len(text) > MAX_RULE_CHARS
    assert suggest_by_rules(text, "episodic").kind is K.EPISODIC
    assert suggest_by_rules(text, "temporal").kind is K.PROSPECTIVE
    assert suggest_by_rules(text, "no-such-type").kind is K.SEMANTIC


def test_correction_needs_strong_cue() -> None:
    # Weak cues that the research prototype counted as corrections.
    for text in (
        "It turns out the cache warms in two seconds.",
        "Contrary to the docs, the flag defaults to off.",
        "The new index supersedes the old one.",
        "We previously said the build takes ten minutes.",
    ):
        assert suggest_by_rules(text, "semantic").kind is not K.CORRECTION, text
    for text in (
        "Correction: the port is 8765.",
        "Erratum in the report: table 2 uses medians.",
        "I retract the claim about the cache size.",
        "The figure in the deck was wrong; revenue was 4.1 M.",
        "That address is no longer true.",
    ):
        assert suggest_by_rules(text, "semantic").kind is K.CORRECTION, text


@pytest.mark.parametrize("text", [
    "", " ", "\x00\x00", "a", "🙂" * 50, "(" * 799, "picked " + "x" * 790 + " over",
    "1. a\n2. b\n" * 40, "x" * 10_000_000,
])
@pytest.mark.parametrize("legacy", ["semantic", "", "temporal", "garbage"])
def test_empty_and_huge_inputs_never_raise(text: str, legacy: str) -> None:
    got = suggest_by_rules(text, legacy)
    assert isinstance(got.kind, MemoryKind)


@pytest.mark.parametrize("bad", [None, 42, b"bytes", ["list"], {"d": 1}])
def test_non_string_inputs_never_raise(bad: object) -> None:
    assert isinstance(suggest_by_rules(bad, bad).kind, MemoryKind)  # type: ignore[arg-type]


def test_empty_text_falls_back_to_legacy_kind() -> None:
    assert suggest_by_rules("", "opinion").kind is K.OPINION
