# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""#146: cross-scope spreading activation seeds are the top_m of ONE scale.

The oracle is the same SYNAPSE algorithm seeded by a single KNN over every
fact the requester may see.  That is what personal-scope recall computes when
every fact is the requester's own, so it is the right answer by construction.
Before the fix the channel seeded from EVERY visible external, on a different
scale from its local seeds, and agreed with this oracle 2 times in 10.
"""

from __future__ import annotations

import numpy as np
import pytest

from superlocalmemory.retrieval.scope_policy import filter_authorized_results
from superlocalmemory.retrieval.spreading_activation import (
    SpreadingActivation,
    SpreadingActivationConfig,
)
from tests.test_retrieval.cross_scope_fixture import (
    REQ,
    PartitionedVS,
    build_store,
    cosine,
)

SCOPES = [(True, True), (True, False), (False, True)]
TOP_K = 10


@pytest.fixture(autouse=True)
def _no_background_writes(monkeypatch):
    import superlocalmemory.storage.deferred_writes as dw
    monkeypatch.setattr(dw, "submit_background", lambda fn: None)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return build_store(tmp_path_factory.mktemp("sa146") / "sa.db")


def _visible(store, include_global: bool, include_shared: bool) -> list[str]:
    return store.ids("L" + ("G" if include_global else "") + ("S" if include_shared else ""))


def _oracle(store, q, cfg, include_global, include_shared):
    """SYNAPSE seeded by one KNN over every visible fact, on the one seed scale."""
    scored = []
    for fid in _visible(store, include_global, include_shared):
        c = cosine(q, store.embs[fid])
        scored.append((fid, max(0.0, c)))
    scored.sort(key=lambda x: (-x[1], x[0]))
    seeds = scored[: cfg.top_m]
    ch = SpreadingActivation(store.db, None, cfg)
    acts = ch._propagate(seeds, REQ, include_global=include_global,
                         include_shared=include_shared)
    if not ch._fok_check(acts):
        return seeds, []
    ranked = sorted(acts.items(), key=lambda x: (-x[1], x[0]))
    return seeds, filter_authorized_results(
        store.db, ranked, REQ, include_global=include_global,
        include_shared=include_shared,
    )[:TOP_K]


class _SeedSpy:
    """Record the seeds handed to _propagate and every neighbour lookup."""

    def __init__(self, monkeypatch) -> None:
        self.seeds: list[list[tuple[str, float]]] = []
        self.lookups = 0
        real_prop = SpreadingActivation._propagate
        real_nb = SpreadingActivation._get_unified_neighbors
        spy = self

        def prop(self_, seeds, *a, **kw):
            spy.seeds.append(list(seeds))
            return real_prop(self_, seeds, *a, **kw)

        def nb(self_, *a, **kw):
            spy.lookups += 1
            return real_nb(self_, *a, **kw)

        monkeypatch.setattr(SpreadingActivation, "_propagate", prop)
        monkeypatch.setattr(SpreadingActivation, "_get_unified_neighbors", nb)


# ---------------------------------------------------------------------------
# 1. Oracle equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("include_global,include_shared", SCOPES)
def test_cross_scope_answer_equals_the_unified_index_oracle(
    store, monkeypatch, include_global, include_shared,
) -> None:
    cfg = SpreadingActivationConfig()
    spy = _SeedSpy(monkeypatch)
    ch = SpreadingActivation(store.db, PartitionedVS(store.embs, store.ids("L")), cfg)
    answered = 0
    for q in store.queries:
        want_seeds, want = _oracle(store, q, cfg, include_global, include_shared)
        got = ch.search(q.tolist(), REQ, top_k=TOP_K, include_global=include_global,
                        include_shared=include_shared)
        assert spy.seeds[-1] == want_seeds
        assert got == want
        answered += bool(got)
    assert answered == len(store.queries), "an empty answer proves nothing"
    externals = sum(1 for s in spy.seeds for fid, _ in s if fid[0] in "GS")
    assert externals > 0, "no external ever seeded; cross-scope signal was lost"


def test_sql_fallback_path_equals_its_oracle(store, monkeypatch) -> None:
    """Without vec0 the seeds are scored on the same max(0, cos) scale (MUSE-4)."""
    cfg = SpreadingActivationConfig()
    spy = _SeedSpy(monkeypatch)
    ch = SpreadingActivation(store.db, None, cfg)
    for q in store.queries[:4]:
        want_seeds, want = _oracle(store, q, cfg, True, True)
        got = ch.search(q.tolist(), REQ, top_k=TOP_K, include_global=True,
                        include_shared=True)
        assert spy.seeds[-1] == want_seeds
        assert got == want and got


# ---------------------------------------------------------------------------
# 2. The bound, and the cost it buys (deterministic, no wall clock)
# ---------------------------------------------------------------------------

def test_at_most_top_m_seeds_and_bounded_neighbour_lookups(store, monkeypatch) -> None:
    """Issue criterion: 5 seeds from many externals, the top 5 by unified score."""
    cfg = SpreadingActivationConfig(top_m=5)
    spy = _SeedSpy(monkeypatch)
    ch = SpreadingActivation(store.db, PartitionedVS(store.embs, store.ids("L")), cfg)
    q = store.queries[3]
    ch.search(q.tolist(), REQ, top_k=TOP_K, include_global=True, include_shared=True)
    seeds, lookups = spy.seeds[0], spy.lookups
    want_seeds, _ = _oracle(store, q, cfg, True, True)
    assert seeds == want_seeds and len(seeds) == 5
    # Lookups are cached per node, and at most top_m nodes are active per
    # round, so a search costs <= top_m * max_iterations of them.  Before the
    # fix it cost one per visible external (here 360+).
    assert 0 < lookups <= cfg.top_m * cfg.max_iterations


def test_a_relevant_local_fact_is_not_evicted_by_unrelated_externals(
    store, monkeypatch,
) -> None:
    """The reporter's clamp on the old mixed scale fails exactly this.

    An unrelated external (cos ~ 0) scored 0.5 on the old external scale and
    outranked a local fact at cos 0.45.  On one scale the local fact stays.
    """
    cfg = SpreadingActivationConfig()
    spy = _SeedSpy(monkeypatch)
    # cos(q, L00007) ~ 0.6; every external sits well below it.
    noise = np.random.default_rng(7).normal(size=store.embs["L00007"].shape[0])
    noise = noise / np.linalg.norm(noise)
    q = (0.6 * store.embs["L00007"] + 0.8 * noise).astype(np.float32)
    ch = SpreadingActivation(store.db, PartitionedVS(store.embs, store.ids("L")), cfg)
    ch.search(q.tolist(), REQ, top_k=TOP_K, include_global=True, include_shared=True)
    seed_ids = [fid for fid, _ in spy.seeds[-1]]
    assert 0.5 < cosine(q, store.embs["L00007"]) < 0.7
    old_scale = sorted(((cosine(q, store.embs[f]) + 1) / 2 for f in store.ids("GS")),
                       reverse=True)
    assert old_scale[cfg.top_m - 1] > cosine(q, store.embs["L00007"]), (
        "fixture no longer exercises the mixed-scale eviction"
    )
    assert seed_ids[0] == "L00007"
    want_seeds, _ = _oracle(store, q, cfg, True, True)
    assert spy.seeds[-1] == want_seeds


# ---------------------------------------------------------------------------
# 3. Authorization first
# ---------------------------------------------------------------------------

def test_a_fact_the_caller_cannot_see_never_seeds_and_never_appears(
    store, monkeypatch,
) -> None:
    """A stale index offers forbidden ids at the very top; they take no slot."""
    cfg = SpreadingActivationConfig(top_m=5)
    spy = _SeedSpy(monkeypatch)
    forbidden = store.ids("DP")
    q = store.embs[forbidden[0]]  # the forbidden fact is the best match there is
    leaky = PartitionedVS(store.embs, store.ids("L") + forbidden)
    ch = SpreadingActivation(store.db, leaky, cfg)
    got = ch.search(q.tolist(), REQ, top_k=TOP_K, include_global=True,
                    include_shared=True)
    assert leaky.search(q, top_k=1)[0][0] == forbidden[0]
    seed_ids = {fid for fid, _ in spy.seeds[-1]}
    assert not seed_ids & set(forbidden)
    assert len(seed_ids) == cfg.top_m, "a forbidden fact starved a real seed"
    assert spy.seeds[-1] == _oracle(store, q, cfg, True, True)[0]
    assert not {fid for fid, _ in got} & set(forbidden)


# ---------------------------------------------------------------------------
# 4. Personal scope is untouched
# ---------------------------------------------------------------------------

def test_personal_scope_is_byte_identical(store, monkeypatch) -> None:
    """The index's own output reaches _propagate unchanged; no external read."""
    cfg = SpreadingActivationConfig()
    spy = _SeedSpy(monkeypatch)
    vs = PartitionedVS(store.embs, store.ids("L"))

    def no_external(*a, **kw):
        raise AssertionError("personal scope read the cross-scope supplement")

    monkeypatch.setattr(store.db, "get_external_visible_embeddings", no_external)
    ch = SpreadingActivation(store.db, vs, cfg)
    for i, q in enumerate(store.queries[:3]):
        spy.lookups = 0
        got = ch.search(q.tolist(), REQ, top_k=TOP_K)
        assert spy.seeds[-1] == vs.search(q, top_k=cfg.top_m)
        assert 0 < spy.lookups <= cfg.top_m * cfg.max_iterations
        ids = [fid for fid, _ in got]
        assert ids == PERSONAL_GOLDEN[i][0]
        assert [s for _, s in got] == pytest.approx(PERSONAL_GOLDEN[i][1], rel=1e-6)


# Captured from 4.1.19 (53ea9996) on this fixture: queries 0-2, personal scope.
PERSONAL_GOLDEN: list[tuple[list[str], list[float]]] = [(['L00023', 'L00035', 'L00017', 'L00046', 'L00002'],
  [0.6748144502157569,
   0.6307753981425253,
   0.623553620607295,
   0.5366946257883112,
   0.5145188655555591]),
 (['L00089',
   'L00117',
   'L00067',
   'L00112',
   'L00013',
   'L00018',
   'L00058',
   'L00072',
   'L00045'],
  [0.5394328979505021,
   0.5391721915918816,
   0.5322421067456772,
   0.52739128729759,
   0.5218416897905173,
   0.5145179179192161,
   0.5145109734672316,
   0.5145032819034272,
   0.5144647081339602]),
 (['L00041', 'L00007', 'L00012', 'L00079'],
  [0.7211928220184252, 0.6540041551666688, 0.5296917662751737, 0.5277701882654858])]


def test_cross_scope_cache_key_changes_with_the_seed_fix_and_personal_does_not() -> None:
    """Activations cached by 4.1.19's cross-scope seeding must not be served
    after the upgrade; personal results were unchanged, so their cache stays."""
    import hashlib

    def old_key(query, profile_id, g, s):  # 4.1.19's formula, verbatim
        scope = f"|g={int(g)}|s={int(s)}".encode()
        return hashlib.sha256(str(query).encode() + profile_id.encode() + scope).hexdigest()[:16]

    ch = SpreadingActivation.__new__(SpreadingActivation)
    q = "what did we decide"
    assert ch._compute_query_hash(q, "p1") == old_key(q, "p1", False, False)
    for g, s in ((True, False), (False, True), (True, True)):
        assert ch._compute_query_hash(
            q, "p1", include_global=g, include_shared=s,
        ) != old_key(q, "p1", g, s)
