# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Every way of reading a memory, checked against one rule.

WHY THIS FILE EXISTS, WHICH IS THE POINT OF IT
----------------------------------------------
4.0.10 first put the withheld-row filter in ``get_facts_by_ids`` alone. The
reasoning was that every retrieval channel re-authorises its candidates through
that method and the engine drops anything it cannot hydrate, so one clause there
covers the lot. That reasoning was correct. The conclusion was wrong: it covered
the RECALL pipeline, and ``search``, ``list_recent``, ``fetch``, ``export``, the
pinned-context injection and the dashboard's own search box are not the recall
pipeline. They read the table directly.

Two independent audits found it immediately. Measured on a copy of the author's
real store, before the fix:

    search_facts_fts("information available")   50 returned, 20 withheld
    get_all_facts(limit=400)                   400 returned, 66 withheld
    get_fact_count()                           5,093 — inflated by 1,195

So "one place" was the right instinct expressed as the wrong mechanism. There is
no single SQL choke point available here: ``_scope_where`` is the only universal
predicate and it is spliced against ``graph_edges``, ``temporal_events``,
``memories``, ``bm25_tokens``, ``fact_temporal_validity`` and
``correction_cases`` as well as ``atomic_facts``, so it cannot carry a fact
column. The honest form is one CLAUSE plus an enumerated set of call sites --
and an enumeration is only trustworthy if something checks it. That is this
file's whole job.

If you add a read path, add it here. If a test below fails, a caller is showing
the owner a model's non-answer as though it were their own note.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.schema import create_all_tables

_PROFILE = "default"
_WITHHELD_TEXT = (
    "Unfortunately, there is no information available about 'Gateway', "
    "'State', 'Bounded', or 'Claude' in the provided text."
)
_REAL_TEXT = "Varun decided the release ships once both audits are clean."


@pytest.fixture()
def store(tmp_path: Path) -> DatabaseManager:
    db = tmp_path / "memory.db"
    conn = sqlite3.connect(str(db))
    create_all_tables(conn)
    conn.execute(
        "INSERT INTO memories (memory_id, profile_id, content) "
        "VALUES ('m1', ?, 'source')", (_PROFILE,),
    )
    rows = [
        ("keep-1", _REAL_TEXT, 0, 0),
        ("keep-2", "The harness caught two vacuous tests before release.", 0, 1),
        ("hide-1", _WITHHELD_TEXT, 1, 0),
        ("hide-2", "The Pro projects have made significant progress.", 1, 1),
    ]
    for fid, content, quarantined, pinned in rows:
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content,"
            " quarantined, pinned, scope, created_at) "
            "VALUES (?, 'm1', ?, ?, ?, ?, 'global', '2026-08-01T00:00:00+00:00')",
            (fid, _PROFILE, content, quarantined, pinned),
        )
    conn.commit()
    conn.close()
    return DatabaseManager(str(db))


def _withheld(facts) -> list[str]:
    return [f.fact_id for f in facts if f.fact_id.startswith("hide-")]


def _kept(facts) -> list[str]:
    return [f.fact_id for f in facts if f.fact_id.startswith("keep-")]


class TestEveryReadPathThatAnswersAQuestion:
    """One parametrised rule, so adding a path here is a one-line change."""

    def test_full_text_search(self, store: DatabaseManager) -> None:
        """The dashboard search box, the `search` tool and `fetch` land here."""
        facts = store.search_facts_fts("information progress release", _PROFILE, limit=50)
        assert _withheld(facts) == [], "search returned a withheld row"
        assert _kept(facts), "search returned nothing at all, so this proves nothing"

    def test_bulk_read(self, store: DatabaseManager) -> None:
        """`list_recent`, export, the CLI listing and several routes use this."""
        facts = store.get_all_facts(_PROFILE)
        assert _withheld(facts) == []
        assert sorted(_kept(facts)) == ["keep-1", "keep-2"]

    def test_hydration(self, store: DatabaseManager) -> None:
        """The recall pipeline's gate: every channel re-authorises through it."""
        facts = store.get_facts_by_ids(
            ["keep-1", "keep-2", "hide-1", "hide-2"], _PROFILE,
        )
        assert _withheld(facts) == []
        assert sorted(_kept(facts)) == ["keep-1", "keep-2"]

    def test_pinned_context_injection(self, store: DatabaseManager) -> None:
        """The most consequential one: pins go straight into an agent's context.

        A withheld row here is not merely displayed, it is asserted as
        background truth for the whole session.
        """
        facts = store.get_pinned(_PROFILE)
        assert _withheld(facts) == []
        assert _kept(facts) == ["keep-2"], "the pinned real memory disappeared"

    def test_cross_profile_sharing(self, store: DatabaseManager) -> None:
        """A withheld row must not cross a profile boundary as shared memory."""
        facts = store.get_external_visible_facts(
            "someone-else", include_global=True, include_shared=True,
        )
        assert _withheld(facts) == []
        assert _kept(facts), "global facts stopped being visible to other profiles"

    def test_the_count_the_owner_reads(self, store: DatabaseManager) -> None:
        """"All memories 5,093" was counting 1,195 withheld summaries."""
        assert store.get_fact_count(_PROFILE) == 2

    def test_the_daily_summary(self, store: DatabaseManager) -> None:
        """4.1.21 (#113): the summaries hand-roll their SQL and were not on this
        list, so "Today" quoted a withheld refusal as the owner's own note.
        Session and project summaries are covered in
        tests/test_summaries/test_summaries_show_only_visible_memories.py."""
        from superlocalmemory.summaries import generate_daily_reflection

        result = generate_daily_reflection(store.db_path, "2026-08-01", _PROFILE)
        assert [f for f in result.source_fact_ids if f.startswith("hide-")] == []
        assert sorted(result.source_fact_ids) == ["keep-1", "keep-2"]


class TestTheEscapeHatchStillWorks:
    def test_repair_and_erasure_can_see_withheld_rows(
        self, store: DatabaseManager,
    ) -> None:
        """A row nothing can read is a row nothing can fix, export or erase."""
        facts = store.get_facts_by_ids(
            ["hide-1", "hide-2"], _PROFILE, include_quarantined=True,
        )
        assert sorted(f.fact_id for f in facts) == ["hide-1", "hide-2"]

    def test_get_fact_is_deliberately_unfiltered(
        self, store: DatabaseManager,
    ) -> None:
        """It is the "read that row" primitive, not a display path.

        Write paths, correction handling and the repair itself all need to read
        a withheld row they hold the id of. It applies no archive filter either,
        which is the same decision made before quarantine existed. Its docstring
        says so; this test makes the decision visible rather than looking like
        an oversight to the next auditor.
        """
        fact = store.get_fact("hide-1", _PROFILE)
        assert fact is not None and fact.fact_id == "hide-1"


class TestTheClauseIsSharedRatherThanRepeated:
    def test_one_definition(self) -> None:
        """Two copies of this predicate would drift. That is how it started."""
        import inspect

        from superlocalmemory.storage import database

        src = inspect.getsource(database)
        inline = src.count("COALESCE(quarantined, 0) = 0")
        assert inline <= 1, (
            f"{inline} inline copies of the quarantine predicate in "
            "database.py; it belongs in visible_fact_clause"
        )

    def test_it_degrades_rather_than_raises_on_an_unmigrated_store(
        self, tmp_path: Path,
    ) -> None:
        """No column must mean no filter, never an exception on every read.

        A DatabaseManager can be pointed at a store engine init never touched.
        Filtering on an absent column would turn a cosmetic gap into total
        read failure.
        """
        db = tmp_path / "old.db"
        conn = sqlite3.connect(str(db))
        create_all_tables(conn)
        conn.execute(
            "INSERT INTO memories (memory_id, profile_id, content) "
            "VALUES ('m1', ?, 's')", (_PROFILE,),
        )
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content,"
            " created_at) VALUES ('f1','m1',?,?, '2026-01-01T00:00:00+00:00')",
            (_PROFILE, "a memory from before the column existed"),
        )
        conn.commit()
        conn.execute("ALTER TABLE atomic_facts DROP COLUMN quarantined")
        conn.commit()
        conn.close()

        mgr = DatabaseManager(str(db))
        assert mgr._has_quarantine_column() is False
        assert "quarantined" not in mgr.visible_fact_clause()
        assert [f.fact_id for f in mgr.get_all_facts(_PROFILE)] == ["f1"]
        assert mgr.get_fact_count(_PROFILE) == 1

    def test_a_duck_typed_db_does_not_silently_empty_the_result(self) -> None:
        """The authorisation fallback must resolve the clause on the TYPE.

        A MagicMock fabricates any attribute asked of it, so an instance check
        returns a callable whose result lands in the SQL string as a repr and
        makes the query a syntax error -- which this function's except turns
        into an empty authorized set, i.e. every candidate dropped. That is a
        silent recall failure, and it is how spreading activation broke when the
        fallback was first written.
        """
        import inspect

        from superlocalmemory.retrieval import scope_policy

        src = inspect.getsource(scope_policy.authorized_fact_ids)
        assert 'getattr(type(db), "visible_fact_clause"' in src, (
            "the clause is resolved on the instance, so a duck-typed or mocked "
            "db will inject a repr into the SQL and drop every candidate"
        )


class TestChannelsDoNotSpendTheirBudgetOnWithheldRows:
    """Correctness was covered by hydration. This is about answer quality.

    A channel that fills its top_k with rows hydration will discard returns
    fewer real candidates, and the ones it drops are not replaced.
    """

    def test_keyword_channel_filters_early(self) -> None:
        import inspect

        from superlocalmemory.retrieval import bm25_channel

        src = inspect.getsource(bm25_channel)
        assert "visible_fact_clause" in src

    def test_entity_map_excludes_them(self) -> None:
        """These rows carry their whole cluster's pooled entity list.

        That is precisely why they out-ranked real memories here: more entity
        links than any single genuine fact.
        """
        import inspect

        from superlocalmemory.retrieval import entity_channel

        src = inspect.getsource(entity_channel)
        assert "visible_fact_clause" in src

    def test_temporal_recency_fallback_excludes_them(self) -> None:
        """Where they won before: no temporal_events, so created_at alone."""
        import inspect

        from superlocalmemory.retrieval import temporal_channel

        src = inspect.getsource(temporal_channel)
        assert "visible_fact_clause" in src

    def test_semantic_index_excludes_them(self) -> None:
        import inspect

        from superlocalmemory.retrieval import vector_store

        src = inspect.getsource(vector_store)
        assert "COALESCE(af.quarantined, 0) = 0" in src

    def test_cognitive_consolidation_excludes_them(self) -> None:
        """The one that could still cause damage after the repair.

        CCQ clusters on entity overlap and ARCHIVES its sources. A withheld
        summary naming ten entities joins a cluster of real memories and takes
        them down with it -- at retention scores the repair's restore would not
        bring back.
        """
        import inspect

        from superlocalmemory.encoding import cognitive_consolidator

        src = inspect.getsource(cognitive_consolidator)
        assert src.count("COALESCE(f.quarantined, 0) = 0") >= 1
        assert src.count("COALESCE(quarantined, 0) = 0") >= 1


class TestExportDoesNotSpreadThem:
    def test_the_export_query_excludes_withheld_rows(self) -> None:
        """Import re-ingests, so an export/import round trip resurrects them.

        Every imported record goes through the normal pipeline and comes out
        with a fresh memory_id and quarantined = 0 -- so a backup taken from a
        poisoned store would recreate all 1,195 rows on a machine where nothing
        had gone wrong, and the repair would have to run there too.
        """
        import inspect

        from superlocalmemory.server.routes import data_io

        src = inspect.getsource(data_io.export_memories)
        assert "COALESCE(quarantined, 0) = 0" in src
