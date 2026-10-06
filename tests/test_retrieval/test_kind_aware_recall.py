# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Recall uses a memory's kind for the question asked - within strict bounds.

The pass may move only what it has evidence for: a memory whose kind matches
the question, compared on the pipeline's own ranking key, within a bounded
margin. Everything else keeps exactly the order the pipeline gave it, and the
exact-lexical hit the pipeline pinned first stays first. Latest-first acts only
on CONFIRMED status/decision memories about the entity the question names.
"""

from __future__ import annotations

import math
import random

import pytest

from superlocalmemory.core.recall_pipeline import _preserve_exact_lexical_evidence
from superlocalmemory.retrieval.kind_aware import (
    DEFAULT_BOOST,
    MAX_BOOST,
    apply_kind_awareness,
    clamp_boost,
    query_intent,
)
from superlocalmemory.storage.memory_kinds import MemoryKind
from superlocalmemory.storage.models import (
    AtomicFact,
    FactType,
    RecallResponse,
    RetrievalResult,
)


def _r(fid, score, kind=None, source="caller", created="2026-10-01", entities=("e_slm",),
       *, ranking=None, bm25=0.0, content=None):
    fact = AtomicFact(fact_id=fid, content=content or fid, fact_type=FactType.SEMANTIC,
                      created_at=created, canonical_entities=list(entities))
    fact.memory_kind, fact.memory_kind_source = kind, (source if kind else None)
    return RetrievalResult(fact=fact, score=score, ranking_score=ranking,
                           channel_scores={"bm25": bm25} if bm25 else {})


def _ids(results):
    return [r.fact.fact_id for r in results]


def _notes(results):
    return {r.fact.fact_id: list(r.evidence_chain or []) for r in results}


@pytest.mark.parametrize("query,kinds", [
    ("What did we decide about the recall ceiling?", {MemoryKind.DECISION}),
    ("How do I upgrade SLM with pipx?", {MemoryKind.PROCEDURE}),
    ("What is the current status of 4.1.19?", {MemoryKind.STATUS}),
    ("What are the rules for releases?", {MemoryKind.RULE}),
    ("What's next on the plan?", {MemoryKind.PROSPECTIVE}),
    ("Which editor do I prefer?", {MemoryKind.OPINION}),
    ("Who maintains the bounded loops server?", set()),
    ("Tell me about Laya", set()),
])
def test_intent_is_read_from_the_question(query, kinds) -> None:
    assert set(query_intent(query).kinds) == kinds


def test_a_question_without_intent_keeps_its_exact_order() -> None:
    results = [_r("a", 0.9), _r("b", 0.8, "decision"), _r("c", 0.7, "rule")]
    assert _ids(apply_kind_awareness(results, "Tell me about Laya")) == ["a", "b", "c"]


def test_a_matching_kind_passes_a_close_neighbour_only() -> None:
    results = [_r("close", 0.90), _r("far", 1.20), _r("decision", 0.85, "decision")]
    results.sort(key=lambda r: -r.score)                          # far, close, decision
    out = _ids(apply_kind_awareness(results, "what did we decide about SLM"))
    assert out == ["far", "decision", "close"]                    # passes 0.90, never 1.20


def test_a_suggested_kind_counts_half() -> None:
    confirmed = [_r("plain", 0.90), _r("d", 0.80, "decision", "caller")]
    suggested = [_r("plain", 0.90), _r("d", 0.80, "decision", "model:laya")]
    q = "what did we decide"
    assert _ids(apply_kind_awareness(confirmed, q)) == ["d", "plain"]    # 0.80*1.15 > 0.90
    assert _ids(apply_kind_awareness(suggested, q)) == ["plain", "d"]    # 0.80*1.075 < 0.90


def test_the_newest_current_state_comes_first_and_the_older_says_so() -> None:
    old = _r("status-old", 0.95, "status", created="2026-09-01")
    new = _r("status-new", 0.60, "status", created="2026-10-02")
    other = _r("other-subject", 0.50, "status", created="2026-10-03", entities=("e_other",))
    out = apply_kind_awareness([old, new, other], "what is the current status of SLM",
                               subject=frozenset({"e_slm"}))
    assert _ids(out) == ["status-new", "status-old", "other-subject"]
    assert "newer:status-new" in out[1].evidence_chain
    assert [r.rank_position for r in out] == [1, 2, 3]


def test_latest_first_never_mixes_kinds_or_subjects() -> None:
    decision = _r("decision-old", 0.9, "decision", created="2026-09-01")
    status = _r("status-new", 0.5, "status", created="2026-10-01")
    out = apply_kind_awareness([decision, status], "current status and what did we decide",
                               subject=frozenset({"e_slm"}))
    assert _ids(out) == ["decision-old", "status-new"]


def test_nothing_is_added_or_removed_and_disabled_means_untouched() -> None:
    results = [_r(f"f{i}", 1.0 - i * 0.01, "decision" if i % 2 else None) for i in range(14)]
    out = apply_kind_awareness(list(results), "what did we decide")
    assert sorted(_ids(out)) == sorted(_ids(results))
    assert _ids(out)[10:] == _ids(results)[10:]                   # below the head: untouched
    assert _ids(apply_kind_awareness(list(results), "what did we decide", enabled=False)) \
        == _ids(results)


# --- L2-06 (r03 A): only a matching kind moves, on the pipeline's ranking key ---

def test_an_intent_query_where_nothing_matches_comes_back_in_exactly_its_input_order() -> None:
    # The order the learned ranker left (ranking_score descending), not raw score.
    res = [_r("A", 0.50, ranking=0.95), _r("B", 0.90, ranking=0.60), _r("C", 0.70, ranking=0.55)]
    out = apply_kind_awareness(list(res), "what is the current status of the deploy")
    assert _ids(out) == ["A", "B", "C"]
    assert all(not notes for notes in _notes(out).values())
    # An order the key does not explain (an earlier guard placed it) is kept too:
    # with no kind evidence the pass has no reason to touch it.
    res = [_r("P", 0.2, ranking=0.20), _r("Q", 0.9, ranking=0.90), _r("R", 0.5, ranking=0.50)]
    out = apply_kind_awareness(list(res), "what is the current status of the deploy")
    assert _ids(out) == ["P", "Q", "R"]


def test_equal_kind_evidence_keeps_the_incoming_order_even_against_the_key() -> None:
    res = [_r("d1", 0.5, "decision", ranking=0.50), _r("d2", 0.6, "decision", ranking=0.60),
           _r("x", 0.4, ranking=0.40)]
    assert _ids(apply_kind_awareness(list(res), "what did we decide")) == ["d1", "d2", "x"]


def test_a_kind_for_another_intent_moves_nothing() -> None:
    res = [_r("A", 0.50, ranking=0.95), _r("d", 0.99, "decision", ranking=0.60),
           _r("C", 0.70, ranking=0.55)]
    out = apply_kind_awareness(list(res), "what is the current status of the deploy")
    assert _ids(out) == ["A", "d", "C"]


def test_a_matching_kind_is_compared_on_the_ranking_key_not_the_raw_score() -> None:
    # High raw score, low ranking key: the learned order holds.
    res = [_r("X", 0.30, ranking=0.90), _r("M", 0.95, "decision", ranking=0.50)]
    assert _ids(apply_kind_awareness(list(res), "what did we decide")) == ["X", "M"]
    # Within the margin on the ranking key it passes, whatever the raw scores say;
    # the non-matching items keep their incoming order (A before B).
    res = [_r("A", 0.50, ranking=0.95), _r("B", 0.90, ranking=0.60),
           _r("D", 0.10, "decision", ranking=0.58)]
    assert _ids(apply_kind_awareness(list(res), "what did we decide")) == ["A", "D", "B"]


def test_a_negative_ranking_key_is_bounded_by_its_magnitude() -> None:
    # A learned ranker can emit negative utilities; a multiplier would push the
    # matching memory DOWN. The margin is relative to |key|, so it stays a lift.
    close = [_r("N", 0.5, ranking=-0.100), _r("M", 0.5, "decision", ranking=-0.105)]
    far = [_r("N", 0.5, ranking=-0.100), _r("M", 0.5, "decision", ranking=-0.200)]
    assert _ids(apply_kind_awareness(close, "what did we decide")) == ["M", "N"]
    assert _ids(apply_kind_awareness(far, "what did we decide")) == ["N", "M"]


def _key(r):
    return r.ranking_score if r.ranking_score is not None else r.score


def test_the_pass_moves_only_matching_items_and_only_within_the_margin() -> None:
    """Invariant check over many random heads (fixed seed, reproducible)."""
    rng = random.Random(4119)
    kinds = (None, None, "decision", "status", "rule")
    for trial in range(400):
        res = []
        for i in range(rng.randint(2, 14)):
            kind = rng.choice(kinds)
            res.append(_r(f"t{trial}-{i}", rng.random(), kind,
                          rng.choice(("caller", "user", "rules", "model:laya")),
                          ranking=rng.uniform(-0.5, 1.5) if rng.random() < 0.8 else None))
        if trial % 2:                                 # half in key order, half not
            res.sort(key=lambda r: -_key(r))
        out = apply_kind_awareness(list(res), "what did we decide")
        before = {r.fact.fact_id: i for i, r in enumerate(res)}
        weight = {r.fact.fact_id: (0.0 if r.fact.memory_kind != "decision" else
                                   1.0 if r.fact.memory_kind_source in ("caller", "user") else 0.5)
                  for r in res}
        assert sorted(_ids(out)) == sorted(before)
        assert _ids(out)[10:] == _ids(res)[10:]
        unmatched = [fid for fid in _ids(out) if not weight[fid]]
        assert unmatched == [fid for fid in _ids(res) if not weight[fid]]
        for hi, x in enumerate(out):
            for y in out[hi + 1:]:
                fx, fy = x.fact.fact_id, y.fact.fact_id
                if before[fx] > before[fy]:          # x was passed over y
                    assert weight[fx] > weight[fy], (trial, fx, fy)
                    kx, ky = _key(x), _key(y)
                    assert ky < kx + DEFAULT_BOOST * weight[fx] * abs(kx) + 1e-12, (trial, fx, fy)


# --- L2-06 (r03 B): the exact-lexical guarantee is never undone ---------------

def test_the_exact_lexical_hit_stays_first_for_an_intent_query() -> None:
    q = "current plan"
    resp = RecallResponse(query=q, results=[
        _r("semantic-noise", 0.80, content="roadmap thoughts"),
        _r("exact", 0.40, content="Our current plan is to ship on Friday", bm25=3.0),
    ])
    _preserve_exact_lexical_evidence(resp, q)
    assert _ids(resp.results) == ["exact", "semantic-noise"]
    assert _ids(apply_kind_awareness(resp.results, q)) == ["exact", "semantic-noise"]


def test_a_matching_kind_never_passes_the_exact_lexical_hit() -> None:
    q = "current plan"
    resp = RecallResponse(query=q, results=[
        _r("status-noise", 0.80, "status", content="roadmap thoughts"),
        _r("exact", 0.75, content="Our current plan is to ship on Friday", bm25=3.0),
    ])
    _preserve_exact_lexical_evidence(resp, q)
    out = apply_kind_awareness(resp.results, q)       # 0.80 x 1.15 would pass 0.75
    assert _ids(out) == ["exact", "status-noise"]


def test_latest_first_never_passes_the_exact_lexical_hit() -> None:
    q = "atlas status"
    old_exact = _r("old-exact", 0.9, "status", created="2026-09-01", entities=("e_atlas",),
                   content="Atlas status: blocked on review", bm25=4.0)
    newer = _r("newer", 0.5, "status", created="2026-10-02", entities=("e_atlas",))
    out = apply_kind_awareness([old_exact, newer], q, subject=frozenset({"e_atlas"}))
    assert _ids(out) == ["old-exact", "newer"]


# --- L2-05 (r03 C): latest-first is CONFIRMED-only, on the query's subject -------

def _atlas_and_kitchen(kitchen_source):
    return [
        _r("atlas-status", 0.92, "status", "caller", created="2026-09-01",
           entities=("ent_varun", "ent_atlas"),
           content="Varun: the Atlas migration is blocked on the schema review"),
        *[_r(f"other{i}", 0.5 - i * 0.01) for i in range(7)],
        _r("kitchen-status", 0.11, "status", kitchen_source, created="2026-10-02",
           entities=("ent_varun", "ent_kitchen"),
           content="Varun: the kitchen renovation is in progress"),
    ]


@pytest.mark.parametrize("subject", [None, frozenset({"ent_atlas"})])
def test_a_suggested_newer_status_never_jumps_a_confirmed_one(subject) -> None:
    q = "what is the current status of the Atlas migration"
    out = apply_kind_awareness(_atlas_and_kitchen("rules"), q, subject=subject)
    assert _ids(out)[0] == "atlas-status"
    assert not any(n.startswith("newer:") for n in out[0].evidence_chain or [])


def test_an_entity_the_query_does_not_name_is_not_a_shared_subject() -> None:
    # Both confirmed now; they share only ent_varun, and the question names Atlas.
    q = "what is the current status of the Atlas migration"
    out = apply_kind_awareness(_atlas_and_kitchen("caller"), q,
                               subject=frozenset({"ent_atlas"}))
    assert _ids(out)[0] == "atlas-status"
    assert not any(n.startswith("newer:") for notes in _notes(out).values() for n in notes)


def test_no_named_subject_means_no_latest_first() -> None:
    old = _r("old", 0.9, "status", created="2026-09-01")
    new = _r("new", 0.5, "status", created="2026-10-02")
    out = apply_kind_awareness([old, new], "what is the current status")
    assert _ids(out) == ["old", "new"]
    assert not any(n.startswith("newer:") for notes in _notes(out).values() for n in notes)


@pytest.mark.parametrize("old_src,new_src", [("rules", "caller"), ("caller", "model:jev"),
                                             ("legacy", "user")])
def test_latest_first_needs_both_memories_confirmed(old_src, new_src) -> None:
    old = _r("old", 0.9, "decision", old_src, created="2026-09-01", entities=("e_atlas",))
    new = _r("new", 0.5, "decision", new_src, created="2026-10-02", entities=("e_atlas",))
    out = apply_kind_awareness([old, new], "what did we decide about Atlas",
                               subject=frozenset({"e_atlas"}))
    assert _ids(out) == ["old", "new"]
    assert not any(n.startswith("newer:") for notes in _notes(out).values() for n in notes)


def test_two_confirmed_statuses_about_the_named_subject_put_the_newer_first() -> None:
    q = "what is the current status of the Atlas migration"
    res = _atlas_and_kitchen("caller")
    res[-1] = _r("atlas-newer", 0.11, "status", "user", created="2026-10-02",
                 entities=("ent_atlas",))
    out = apply_kind_awareness(res, q, subject=frozenset({"ent_atlas"}))
    assert _ids(out)[:2] == ["atlas-newer", "atlas-status"]
    assert "newer:atlas-newer" in out[1].evidence_chain
    assert [n for notes in _notes(out).values() for n in notes
            if n.startswith("newer:")] == ["newer:atlas-newer"]


def test_latest_first_never_jumps_a_newer_memory_about_the_same_subject() -> None:
    # Subjects that do not chain (W~V on Y, U~V on X, W and U unrelated) must
    # neither loop nor put V above the newer U it shares X with.
    w = _r("W", 0.9, "status", created="2026-09-01", entities=("Y",))
    u = _r("U", 0.8, "status", created="2026-09-03", entities=("X",))
    v = _r("V", 0.7, "status", created="2026-09-02", entities=("X", "Y"))
    out = _ids(apply_kind_awareness([w, u, v], "current status", subject=frozenset({"X", "Y"})))
    assert out.index("U") < out.index("V")


# --- L2-16 (r03 D): the boost is clamped to [0, 0.5] -------------------------------

def test_a_boost_of_50_is_clamped_to_half() -> None:
    res4 = [_r("best", 0.95), _r("weak-rule", 0.02, "rule")]
    assert _ids(apply_kind_awareness(res4, "what are the rules", boost=50.0)) \
        == ["best", "weak-rule"]
    # Clamped to 0.5, not to nothing: 0.65 x 1.5 passes 0.90, 0.55 x 1.5 does not.
    assert _ids(apply_kind_awareness([_r("a", 0.90), _r("r", 0.65, "rule")],
                                     "what are the rules", boost=50.0)) == ["r", "a"]
    assert _ids(apply_kind_awareness([_r("a", 0.90), _r("r", 0.55, "rule")],
                                     "what are the rules", boost=50.0)) == ["a", "r"]


@pytest.mark.parametrize("raw,expected", [
    (50, MAX_BOOST), (50.0, 0.5), (0.5, 0.5), (0.2, 0.2), (0, 0.0), (-3, 0.0),
    (math.inf, DEFAULT_BOOST), (math.nan, DEFAULT_BOOST), ("0.3", DEFAULT_BOOST),
    (None, DEFAULT_BOOST), (True, DEFAULT_BOOST),
])
def test_clamp_boost(raw, expected) -> None:
    assert clamp_boost(raw) == expected


def test_a_zero_boost_moves_nothing() -> None:
    res = [_r("a", 0.90), _r("d", 0.89, "decision")]
    assert _ids(apply_kind_awareness(res, "what did we decide", boost=0.0)) == ["a", "d"]


# --- 4.1.22 G10: newer confirmed rules, and questions about a current value ---

@pytest.mark.parametrize("query,current", [
    ("What is the recall latency ceiling?", True),
    ("what's the remember ceiling", True),
    ("Which editor do we use?", True),
    ("How many embedding workers run?", True),
    ("What was the recall ceiling in August?", False),
    ("Tell me about Laya", False),
    ("Who maintains the bounded loops server?", False),
])
def test_a_question_about_a_current_value_is_recognised(query, current) -> None:
    assert query_intent(query).current_value is current


def test_a_current_value_question_puts_the_newer_confirmed_rule_first() -> None:
    old = _r("ceiling-2s", 0.95, "rule", created="2026-08-20")
    new = _r("ceiling-3s", 0.70, "rule", created="2026-10-03")
    out = apply_kind_awareness([old, new], "What is the recall latency ceiling?",
                               subject=frozenset({"e_slm"}))
    assert _ids(out) == ["ceiling-3s", "ceiling-2s"]
    assert "newer:ceiling-3s" in out[1].evidence_chain


def test_a_current_value_question_never_reorders_suggested_kinds() -> None:
    old = _r("ceiling-2s", 0.95, "rule", "rules", created="2026-08-20")
    new = _r("ceiling-3s", 0.70, "rule", "rules", created="2026-10-03")
    out = apply_kind_awareness([old, new], "What is the recall latency ceiling?",
                               subject=frozenset({"e_slm"}))
    assert _ids(out) == ["ceiling-2s", "ceiling-3s"]


def test_a_question_about_the_past_keeps_the_older_rule_where_it_was() -> None:
    old = _r("ceiling-2s", 0.95, "rule", created="2026-08-20")
    new = _r("ceiling-3s", 0.70, "rule", created="2026-10-03")
    out = apply_kind_awareness([old, new], "What was the recall ceiling in August?",
                               subject=frozenset({"e_slm"}))
    assert _ids(out) == ["ceiling-2s", "ceiling-3s"]


def test_a_current_value_question_never_boosts_a_kind_by_itself() -> None:
    results = [_r("plain", 0.90), _r("d", 0.85, "decision")]
    assert _ids(apply_kind_awareness(results, "what is the release process")) == ["plain", "d"]


def test_a_newer_rule_does_not_pass_an_older_decision() -> None:
    decision = _r("decision-old", 0.9, "decision", created="2026-09-01")
    rule = _r("rule-new", 0.5, "rule", created="2026-10-01")
    out = apply_kind_awareness([decision, rule], "what is the recall ceiling",
                               subject=frozenset({"e_slm"}))
    assert _ids(out) == ["decision-old", "rule-new"]
