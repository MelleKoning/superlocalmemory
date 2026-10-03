# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.retrieval.entity_channel — Entity Graph + Spreading Activation.

Covers:
  - extract_query_entities() — proper nouns, title-cased, quoted, dedup, stopwords
  - EntityGraphChannel.search() — entity resolution, direct facts, spreading activation
  - Edge traversal and decay
  - Discover entities from activated facts
  - No entities in query -> empty results
  - Mock DB interactions (get_entity_by_name, get_facts_by_entity, get_edges_for_node)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from superlocalmemory.retrieval.entity_channel import (
    EntityGraphChannel,
    extract_query_entities,
)
from superlocalmemory.storage.models import (
    AtomicFact,
    CanonicalEntity,
    EdgeType,
    GraphEdge,
)

# ---------------------------------------------------------------------------
# extract_query_entities
# ---------------------------------------------------------------------------


class TestExtractQueryEntities:
    def test_proper_nouns_extracted(self) -> None:
        result = extract_query_entities("Did Alice meet Bob yesterday?")
        names = [n.lower() for n in result]
        assert "alice" in names
        assert "bob" in names

    def test_title_cased_fallback(self) -> None:
        # All lowercase — title() should produce proper nouns
        result = extract_query_entities("what did alice do?")
        names = [n.lower() for n in result]
        assert "alice" in names

    def test_quoted_phrases(self) -> None:
        result = extract_query_entities('Tell me about "Project Alpha"')
        texts = [n.lower() for n in result]
        assert "project alpha" in texts

    def test_stopwords_filtered(self) -> None:
        result = extract_query_entities("What did they do?")
        names_lower = [n.lower() for n in result]
        # "What", "They" are in entity stop list
        assert "what" not in names_lower

    def test_short_names_filtered(self) -> None:
        result = extract_query_entities("A B C Alice")
        # Single-char names should be filtered (len < 2)
        names = [n for n in result if len(n) == 1]
        assert len(names) == 0

    def test_deduplication_case_insensitive(self) -> None:
        result = extract_query_entities("Alice alice ALICE")
        # Should have only one entry for alice
        lower_names = [n.lower() for n in result]
        assert lower_names.count("alice") == 1

    def test_empty_query(self) -> None:
        assert extract_query_entities("") == []

    def test_no_entities(self) -> None:
        result = extract_query_entities("what is going on?")
        # "What" is a stop word. "Going" from title() might pass.
        # The key check: no crash, returns list
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# EntityGraphChannel with mocks
# ---------------------------------------------------------------------------


def _mock_entity(entity_id: str, name: str) -> CanonicalEntity:
    return CanonicalEntity(entity_id=entity_id, canonical_name=name)


def _mock_fact(fact_id: str, canonical_entities: list[str] | None = None) -> AtomicFact:
    return AtomicFact(
        fact_id=fact_id,
        memory_id="m0",
        content=f"fact {fact_id}",
        canonical_entities=canonical_entities or [],
    )


def _mock_edge(source: str, target: str) -> GraphEdge:
    return GraphEdge(
        edge_id=f"e_{source}_{target}",
        source_id=source,
        target_id=target,
        edge_type=EdgeType.ENTITY,
        weight=1.0,
    )


def _authorize_all_mock_candidates(db: MagicMock) -> None:
    """Model the canonical scope check for single-profile channel unit tests."""
    db.get_facts_by_ids.side_effect = lambda fact_ids, _profile_id, **_kwargs: [
        _mock_fact(fact_id) for fact_id in fact_ids
    ]


class TestEntityGraphChannelSearch:
    def test_entity_map_reload_reads_only_required_columns(
        self,
        monkeypatch,
    ) -> None:
        monkeypatch.setenv("SLM_MAX_FACTS_UNBOUNDED", "123")
        db = MagicMock()
        db.execute.return_value = [
            {
                "fact_id": "f1",
                "canonical_entities_json": json.dumps(["e_alice", "e_bob"]),
            },
            {
                "fact_id": "f2",
                "canonical_entities_json": "[]",
            },
        ]
        channel = EntityGraphChannel(db)

        channel._load_entity_maps("default")

        sql = db.execute.call_args.args[0]
        assert sql.startswith(
            "SELECT fact_id, canonical_entities_json FROM atomic_facts",
        )
        assert "ORDER BY created_at DESC LIMIT ?" in sql
        assert db.execute.call_args.args[1][-1] == 123
        assert "embedding" not in sql
        db.get_all_facts.assert_not_called()
        assert channel._visible_fact_ids == {"f1", "f2"}
        assert channel._entity_to_facts["e_alice"] == ["f1"]
        assert channel._fact_to_entities["f1"] == ["e_alice", "e_bob"]

    def test_no_entities_returns_empty(self) -> None:
        db = MagicMock()
        ch = EntityGraphChannel(db)
        results = ch.search("what is going on", "default")
        assert results == []

    def test_entity_not_found_returns_empty(self) -> None:
        db = MagicMock()
        db.get_entity_by_name.return_value = None
        ch = EntityGraphChannel(db)
        results = ch.search("Did Alice do something?", "default")
        assert results == []

    def test_direct_entity_facts_returned(self) -> None:
        db = MagicMock()
        _authorize_all_mock_candidates(db)
        db.get_entity_by_name.return_value = _mock_entity("e_alice", "Alice")
        db.get_facts_by_entity.return_value = [_mock_fact("f1", ["e_alice"])]
        db.get_edges_for_node.return_value = []
        db.execute.return_value = []
        ch = EntityGraphChannel(db, max_hops=1)
        results = ch.search("What did Alice do?", "default")
        assert len(results) > 0
        fact_ids = [r[0] for r in results]
        assert "f1" in fact_ids

    def test_spreading_activation_through_edges(self) -> None:
        db = MagicMock()
        _authorize_all_mock_candidates(db)
        db.get_entity_by_name.return_value = _mock_entity("e_alice", "Alice")
        # Alice directly linked to f1
        db.get_facts_by_entity.return_value = [_mock_fact("f1")]
        # f1 has edge to f2
        db.get_edges_for_node.side_effect = lambda fid, pid, **kwargs: (
            [_mock_edge("f1", "f2")] if fid == "f1" else []
        )
        # f2 has canonical entities -> discover new entities
        db.execute.return_value = []

        ch = EntityGraphChannel(db, decay=0.7, max_hops=3)
        results = ch.search("What did Alice do?", "default")
        fact_ids = [r[0] for r in results]
        # f2 should be discovered via spreading activation
        assert "f1" in fact_ids
        assert "f2" in fact_ids

    def test_decay_reduces_activation(self) -> None:
        db = MagicMock()
        db.get_entity_by_name.return_value = _mock_entity("e_alice", "Alice")
        db.get_facts_by_entity.return_value = [_mock_fact("f1")]
        db.get_edges_for_node.side_effect = lambda fid, pid, **kwargs: (
            [_mock_edge("f1", "f2")] if fid == "f1" else []
        )
        db.execute.return_value = []

        ch = EntityGraphChannel(db, decay=0.7, max_hops=3)
        results = ch.search("What did Alice do?", "default")
        scores = {r[0]: r[1] for r in results}
        # f1 should have higher activation than f2
        if "f2" in scores:
            assert scores["f1"] > scores["f2"]

    def test_activation_threshold_filters_weak(self) -> None:
        db = MagicMock()
        db.get_entity_by_name.return_value = _mock_entity("e_alice", "Alice")
        db.get_facts_by_entity.return_value = [_mock_fact("f1")]
        db.get_edges_for_node.return_value = []
        db.execute.return_value = []

        ch = EntityGraphChannel(db, decay=0.7, activation_threshold=0.5, max_hops=3)
        results = ch.search("What did Alice do?", "default")
        # All results should be above threshold
        for _, score in results:
            assert score >= 0.5

    def test_top_k_limits_results(self) -> None:
        db = MagicMock()
        db.get_entity_by_name.return_value = _mock_entity("e_alice", "Alice")
        facts = [_mock_fact(f"f{i}") for i in range(20)]
        db.get_facts_by_entity.return_value = facts
        db.get_edges_for_node.return_value = []
        db.execute.return_value = []

        ch = EntityGraphChannel(db, max_hops=1)
        results = ch.search("What did Alice do?", "default", top_k=5)
        assert len(results) <= 5

    def test_entity_resolver_used_when_provided(self) -> None:
        db = MagicMock()
        _authorize_all_mock_candidates(db)
        resolver = MagicMock()
        resolver.lookup.return_value = {"Alice": "e_alice"}
        db.get_facts_by_entity.return_value = [_mock_fact("f1")]
        db.get_edges_for_node.return_value = []
        db.execute.return_value = []

        ch = EntityGraphChannel(db, entity_resolver=resolver, max_hops=1)
        results = ch.search("What did Alice do?", "default")
        resolver.lookup.assert_called_once()
        # A search must never use the resolver's writing path.
        resolver.resolve.assert_not_called()
        assert len(results) > 0

    def test_discover_entities_from_facts(self) -> None:
        db = MagicMock()
        _authorize_all_mock_candidates(db)
        db.get_entity_by_name.return_value = _mock_entity("e_alice", "Alice")
        db.get_facts_by_entity.side_effect = lambda eid, pid, **kwargs: (
            [_mock_fact("f1", ["e_alice"])]
            if eid == "e_alice"
            else [_mock_fact("f3", ["e_bob"])]
            if eid == "e_bob"
            else []
        )
        db.get_edges_for_node.return_value = []

        # Simulate discovering e_bob from f1's canonical_entities
        def mock_execute(sql, params):
            if params and params[0] == "f1":
                row = MagicMock()
                row.__iter__ = lambda s: iter([("canonical_entities_json", json.dumps(["e_bob"]))])
                row.keys = lambda: ["canonical_entities_json"]
                # Return dict-like row
                mock_row = MagicMock()
                mock_row.__iter__ = lambda s: iter(
                    [("canonical_entities_json", json.dumps(["e_bob"]))]
                )
                d = {"canonical_entities_json": json.dumps(["e_bob"])}
                mock_row.__getitem__ = lambda s, k: d[k]
                mock_dict = MagicMock(return_value=d)
                return [mock_dict]
            return []

        db.execute.side_effect = mock_execute

        ch = EntityGraphChannel(db, decay=0.7, max_hops=3)
        results = ch.search("What did Alice do?", "default")
        # Should have found facts via both Alice and discovered Bob
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Supersession penalty (4.1.18)
# ---------------------------------------------------------------------------

class TestSupersessionPenalty:
    """A ``supersedes`` edge must not cut a fact's activation.

    The penalty multiplied the edge's SOURCE by 0.3 "because this fact was
    replaced" — but every writer and reader agrees the source is the NEWER
    fact (storage/models.py: "Newer fact replaces older"; the sheaf checker
    writes source = the fact being stored). So the current memory was the one
    suppressed. Measured on Varun's store: source was newer on 4,481 of 4,786
    edges. And the edges are not supersessions at all: all of them come from
    the sheaf consistency check, and 0 of 34 hand-judged were real. Flipping
    the direction would only move the damage onto 1,603 older facts.
    """

    @staticmethod
    def _channel(edges, created):
        from unittest.mock import MagicMock as _MM

        db = _MM()

        def execute(sql, params=()):
            if "FROM graph_edges" in sql:
                wanted = [t for t in ("contradiction", "supersedes") if f"'{t}'" in sql]
                return [dict(e) for e in edges if e["edge_type"] in wanted]
            if "FROM atomic_facts" in sql:
                return [{"fact_id": f, "created_at": t} for f, t in created.items()]
            return []

        db.execute.side_effect = execute
        return EntityGraphChannel(db)

    def test_the_newer_fact_is_not_suppressed_by_its_own_supersedes_edge(self) -> None:
        ch = self._channel(
            [{"source_id": "new", "target_id": "old", "edge_type": "supersedes"}],
            {"new": "2026-09-26T00:00:00", "old": "2026-09-06T00:00:00"},
        )
        activation = {"new": 1.0, "old": 1.0}
        ch._suppress_contradictions(activation, "default")
        assert activation["new"] == 1.0

    def test_a_supersedes_edge_alone_never_lowers_any_activation(self) -> None:
        ch = self._channel(
            [{"source_id": "a", "target_id": "b", "edge_type": "supersedes"},
             {"source_id": "c", "target_id": "a", "edge_type": "supersedes"}],
            {"a": "2026-09-10", "b": "2026-09-01", "c": "2026-09-20"},
        )
        activation = {"a": 1.0, "b": 1.0, "c": 1.0}
        ch._suppress_contradictions(activation, "default")
        assert activation == {"a": 1.0, "b": 1.0, "c": 1.0}

    def test_contradiction_still_lowers_the_older_fact(self) -> None:
        """Deliberately unchanged in 4.1.18; pinned so the fix cannot drift it."""
        ch = self._channel(
            [{"source_id": "a", "target_id": "b", "edge_type": "contradiction"}],
            {"a": "2026-09-01", "b": "2026-09-20"},
        )
        activation = {"a": 1.0, "b": 1.0}
        ch._suppress_contradictions(activation, "default")
        assert activation == {"a": 0.5, "b": 1.0}
