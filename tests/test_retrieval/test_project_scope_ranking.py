# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""GitHub #150 on a real store: recall that knows the project.

The fixture is the issue's shape - five on-topic memories saved under one
project (some as a name, some as a full path), and cross-project session
trivia that shares their words. Retrieval is the real pipeline (channels,
fusion, evidence floor, result building); only the embedder is a
deterministic bag-of-words stand-in, so two runs and two machines agree.

The numbers quoted in retrieval/project_scope.py's BOOST comment come from
``test_measured_ranking_before_and_after``.
"""

from __future__ import annotations

import hashlib
import random
import re
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from superlocalmemory.retrieval.facets import Facets

PROJECT_PATH = "/Users/dev/work/acme-billing"
PROJECT_FACTS = (
    ("Invoice retries back off exponentially and stop after six attempts.", "acme-billing"),
    ("Invoice PDFs are rendered by the render service before the invoice email is sent.",
     PROJECT_PATH),
    ("Invoice reconciliation runs nightly at 02:00 UTC against the ledger.", "ACME-Billing"),
    ("Invoice webhooks from Stripe are verified with the signing secret before processing.",
     PROJECT_PATH + "/"),
    ("Invoice numbers are per-tenant and gap-free.", "acme-billing"),
)
TRIVIA = (
    ("[docs-site] session ended 2026-10-04 18:20 | branch: main | files: invoice-faq.md, "
     "pricing.md", "docs-site"),
    ("[newsletter] session ended 2026-10-04 17:02 | recent: fix typo in invoice email footer",
     "newsletter"),
    ("[platform] session ended 2026-10-04 11:40 | files: k8s/upgrade.md", "platform"),
    ("The mobile app caches the user's avatar for a day.", "mobile"),
)
#: Saved with no project at all: legacy memories must stay findable.
UNTAGGED = (
    "Invoice is a word the finance team prefers over bill.",
    "Team offsite lunch was invoiced to the events budget.",
)
OTHER_PROFILE_FACT = "Invoice disputes in the work profile are escalated to legal."


def _embed(text: str) -> list[float]:
    vec = np.zeros(768, dtype=np.float32)
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        vec[int(hashlib.sha256(token.encode()).hexdigest(), 16) % 768] += 1.0
    norm = float(np.linalg.norm(vec))
    return (vec / norm).tolist() if norm else (vec + 1e-3).tolist()


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.engine import MemoryEngine
    from superlocalmemory.core.engine_ingestion import canonical_store, local_trusted_actor_id
    from superlocalmemory.storage.models import Mode

    embedder = MagicMock()
    embedder.embed.side_effect = _embed
    embedder.embed_batch.side_effect = lambda texts: [_embed(t) for t in texts]
    embedder.is_available = True
    embedder.compute_fisher_params.return_value = ([0.0] * 768, [1.0] * 768)
    config = SLMConfig.for_mode(Mode.A, base_dir=tmp_path_factory.mktemp("i150"))
    config.retrieval.use_cross_encoder = False
    engine = MemoryEngine(config)
    with patch("superlocalmemory.core.engine_wiring.init_embedder", return_value=embedder):
        engine.initialize()
        engine._embedder = embedder
    engine._db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('work', 'work')")

    def save(content: str, project: str | None, profile_id: str | None = None) -> None:
        canonical_store(engine, content, source_type="python-api",
                        trusted_actor_id=local_trusted_actor_id("python-api"),
                        metadata={"project": project} if project else {},
                        require_complete=True, profile_id=profile_id)

    # The same ids on every run. Memory and fact ids are random, and recall
    # breaks score ties by id (deterministically, for a given store), so a
    # store rebuilt with fresh ids can rank tied candidates the other way.
    # Unpinned, this fixture lost its "weakly relevant project memory" in 3
    # of 42 id draws (ids 11, 20 and 31 of a seeded sweep), failing
    # test_the_boost_is_bounded and test_measured_ranking_before_and_after.
    # Pinned, every run builds the identical store and asks the same thing.
    ids = random.Random(150)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(uuid, "uuid4", lambda: uuid.UUID(int=ids.getrandbits(128), version=4))
        for content, project in (*PROJECT_FACTS, *TRIVIA):
            save(content, project)
        for content in UNTAGGED:
            save(content, None)
        save(OTHER_PROFILE_FACT, "acme-billing", profile_id="work")
    yield engine
    engine.close()


def _recall(engine, query: str, limit: int = 10, **facets):
    kwargs = {"facets": Facets.of(**facets)} if facets else {}
    return engine.recall(query, limit=limit, fast=True, **kwargs)


def _contents(response) -> list[str]:
    return [r.fact.content for r in response.results]


_PROJECT = {c for c, _ in PROJECT_FACTS}


def _is_project(content: str) -> bool:
    # Enrichment may also store a fact with the "[name] " prefix stripped.
    return content in _PROJECT


# -- criterion 1 -------------------------------------------------------------


def test_project_memories_rank_above_cross_project_trivia(store) -> None:
    query = "what happens to an invoice"
    before = _contents(_recall(store, query, limit=5))
    after = _contents(_recall(store, query, limit=5, prefer_project=PROJECT_PATH))

    # The fixture reproduces the defect: trivia in the top five, one of the
    # project's own memories pushed out of it.
    assert not all(_is_project(c) for c in before), before
    assert sorted(after) == sorted(_PROJECT), after


def test_prefer_project_reports_what_it_did(store) -> None:
    response = _recall(store, "what happens to an invoice", prefer_project="ACME-BILLING")
    prefer = response.project_scope["prefer"]
    assert prefer["key"] == "acme-billing" and prefer["matched"] == 5
    assert prefer["boost"] == pytest.approx(0.25)
    boosted = [r for r in response.results if "same_project" in r.evidence_chain]
    assert {r.fact.content for r in boosted} == _PROJECT


def test_the_boost_moves_rank_only_never_the_reported_score(store) -> None:
    query = "what happens to an invoice"
    plain = {r.fact.fact_id: (r.score, r.relevance_score) for r in _recall(store, query).results}
    boosted = _recall(store, query, prefer_project=PROJECT_PATH).results
    for r in boosted:
        if r.fact.fact_id in plain:
            assert (r.score, r.relevance_score) == plain[r.fact.fact_id]


def test_the_boost_is_bounded(store) -> None:
    """A project memory at under 80% of a better match's score stays below it."""
    query = "how do invoices work"
    plain = {r.fact.content: r.ranking_score for r in _recall(store, query).results}
    after = _contents(_recall(store, query, prefer_project=PROJECT_PATH))
    weak = [c for c in _PROJECT if c in plain
            and any(plain[c] < 0.8 * plain[t] for t in plain if not _is_project(t))]
    assert weak, "fixture lost its weakly relevant project memory"
    for c in weak:
        stronger = [t for t in plain if not _is_project(t) and plain[c] < 0.8 * plain[t]]
        assert all(after.index(t) < after.index(c) for t in stronger), (c, after)


def test_measured_ranking_before_and_after(store) -> None:
    """The fixture holds both sides of the bound, so the BOOST comment's
    claim is checked: every project memory that trivia outranked by less than
    20% is lifted above it, and every one outranked by more stays below.

    With the fixture's ids pinned the ratios are the same on every run:
    0.87-0.98 lifted, 0.65 held. The bands are asserted rather
    than the digits so a change elsewhere in ranking that keeps the bound
    intact does not fail here.
    """
    lifted_seen, held_seen = [], []
    for query in ("what happens to an invoice", "how do invoices work",
                  "invoice retries and numbering"):
        plain = _recall(store, query).results
        after = _contents(_recall(store, query, prefer_project=PROJECT_PATH))
        for i, r in enumerate(plain):
            if not _is_project(r.fact.content):
                continue
            for t in plain[:i]:
                if _is_project(t.fact.content):
                    continue
                ratio = r.ranking_score / t.ranking_score
                order = after.index(r.fact.content) < after.index(t.fact.content)
                (lifted_seen if ratio >= 1 / 1.25 else held_seen).append(ratio)
                assert order is (ratio > 1 / 1.25), (query, ratio, after)
    assert lifted_seen and all(0.8 <= x < 1.0 for x in lifted_seen), lifted_seen
    assert held_seen and all(0.5 <= x < 0.8 for x in held_seen), held_seen


# -- criteria 2 and 5 --------------------------------------------------------


def test_a_project_with_no_saved_memories_changes_nothing(store) -> None:
    query = "what happens to an invoice"
    plain = _recall(store, query)
    preferred = _recall(store, query, prefer_project="/Users/dev/work/ghost-project")
    assert _contents(preferred) == _contents(plain) and _contents(plain)
    assert preferred.project_scope["prefer"]["matched"] == 0


def test_a_project_filter_matching_nothing_falls_back_and_says_so(store) -> None:
    query = "what happens to an invoice"
    plain = _recall(store, query)
    filtered = _recall(store, query, project="ghost-project")
    assert _contents(filtered) == _contents(plain) and _contents(filtered)
    report = filtered.project_scope["filter"]
    assert report["applied"] is False and report["matched"] == 0
    assert "ghost-project" in report["note"] and "not narrowed" in report["note"]


def test_a_value_that_names_no_project_also_falls_back(store) -> None:
    filtered = _recall(store, "what happens to an invoice", project="/")
    assert _contents(filtered)
    assert filtered.project_scope["filter"]["reason"] == "not_a_project"


# -- the filter when it does match --------------------------------------------


def test_project_filter_keeps_that_projects_memories_by_name_or_path(store) -> None:
    for value in ("ACME-billing", PROJECT_PATH, PROJECT_PATH + "/", " acme-billing "):
        response = _recall(store, "what happens to an invoice", project=value)
        assert set(_contents(response)) == _PROJECT, value
        assert response.project_scope["filter"]["applied"] is True


def test_untagged_legacy_memories_stay_findable_under_a_preference(store) -> None:
    after = _contents(_recall(store, "invoice word finance team prefers",
                              prefer_project=PROJECT_PATH))
    assert UNTAGGED[0] in after


def test_another_profiles_memories_never_appear(store) -> None:
    for facets in ({"project": "acme-billing"}, {"prefer_project": "acme-billing"}):
        got = _contents(_recall(store, "invoice disputes escalated to legal", **facets))
        assert OTHER_PROFILE_FACT not in got


# -- determinism --------------------------------------------------------------


@pytest.mark.parametrize("facets", [{"prefer_project": PROJECT_PATH},
                                    {"project": "acme-billing"},
                                    {"project": "ghost-project"}])
def test_the_same_question_and_project_rank_identically(store, facets) -> None:
    runs = [_recall(store, "what happens to an invoice", **facets) for _ in range(3)]
    # ranking_score itself creeps in the 8th decimal between runs (recency
    # decay reads the clock); the order and the reported score must not move.
    shapes = [[(r.fact.fact_id, r.score) for r in run.results] for run in runs]
    assert shapes[0] == shapes[1] == shapes[2]
    assert runs[0].project_scope == runs[1].project_scope == runs[2].project_scope
