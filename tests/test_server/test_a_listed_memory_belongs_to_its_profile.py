# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""The lists that answer "what is in my memory", checked by asking them.

WHY THIS FILE EXISTS, WHICH IS THE POINT OF IT
----------------------------------------------
4.0.10 moved model-authored summaries out of the retrieval corpus, excluded
them from every ``DatabaseManager`` read path and every retrieval channel, and
enumerated those paths in a test. That enumeration was of *methods*. Two HTTP
routes never call one — they build their own SQL against ``atomic_facts`` — so
neither was covered, and both shipped still serving withheld rows.

Measured against a live store on released 4.0.10:

    GET /api/memories?limit=50      24 of 50 rows withheld, page ONE
                                    total reported 5,218 · real memories 3,919
    GET /api/v3/timeline/?range=30d  2 of 200 events withheld

The first is the dashboard's main memory list. The release note for 4.0.10 was
"your memories, not the summarizer's", and that list was showing the
summarizer's, above the fold, with the count inflated by exactly the 1,299 rows
the release had withheld.

WHY THIS TEST CALLS THE SURFACE INSTEAD OF GREPPING IT. The obvious cheap guard
is to assert that no module selects from ``atomic_facts`` without the clause.
That guard would have been useless here and actively misleading: 68 files run
unguarded SQL against that table and almost all of them are *supposed* to —
migrations, GDPR erasure, lifecycle sweeps, the repair itself. Meanwhile
``mcp/tools_active.py`` selects from it directly and was already clean. Presence
of raw SQL predicts nothing. What a reader cares about is what arrived on their
screen, so that is what is asserted: seed a withheld row, call the route, look
at the response.

If you add a surface that lists or counts memories, add it here.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

_WITHHELD = (
    "Unfortunately, there is no information available about 'Gateway', "
    "'State', 'Bounded', or 'Claude' in the provided text."
)
_REAL = "Varun decided the release ships once both audits are clean."

# Two of each in this profile's own scope, so a total can be wrong in either
# direction and still be caught: an unfiltered count reads 4, a
# double-filtered one reads 0.
#
# The last two exist because of a vacuity check. With only personal-scope rows,
# the ``scope=global`` and ``scope=shared`` parametrisations below PASSED
# against the unfixed handler — those views returned nothing at all, so they
# asserted nothing. A withheld row is seeded into each of those scopes to make
# all four WHERE-clause branches carry a real assertion.
#
#   (fact_id, content, quarantined, scope, shared_with)
_ROWS = (
    ("keep-1", _REAL, 0, "personal", None),
    ("keep-2", "The harness caught two vacuous tests before release.",
     0, "personal", None),
    ("hide-1", _WITHHELD, 1, "personal", None),
    ("hide-2", "The Pro projects have made significant progress.",
     1, "personal", None),
    ("hide-global", _WITHHELD, 1, "global", None),
    ("hide-shared", _WITHHELD, 1, "shared", '["default"]'),
)

#: Rows in this profile's default view — what ``total`` must report.
_VISIBLE_IN_MINE = 2


def _daemon_headers(app) -> dict[str, str]:
    d = app.state.daemon_descriptor
    return {
        "X-SLM-Daemon-Capability": d.capability,
        "X-SLM-Target-Instance": d.instance_id,
    }


@pytest.fixture()
def client(engine_with_mock_deps):
    from superlocalmemory.server.profile_runtime import bind_profile_runtime
    from superlocalmemory.server.unified_daemon import create_app

    engine = engine_with_mock_deps
    engine.profile_id = "default"
    engine._config.active_profile = "default"
    engine._db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('default','default')"
    )
    engine._db.execute(
        "INSERT INTO memories (memory_id, profile_id, content, session_id, "
        " speaker, role, created_at, metadata_json, scope) "
        "VALUES ('m1','default','source','s1','user','user',"
        " '2026-01-01T00:00:00Z','{}','personal')"
    )
    for fid, content, quarantined, scope, shared_with in _ROWS:
        engine._db.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, "
            " lifecycle, created_at, scope, shared_with, quarantined) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (fid, "m1", "default", content, "active",
             "2026-01-01T00:00:00Z", scope, shared_with, quarantined),
        )

    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    bind_profile_runtime(app.state, engine, engine._config)
    yield TestClient(app), _daemon_headers(app)


def _served(payload: dict) -> list[str]:
    return [m["id"] for m in payload.get("memories", [])]


class TestTheAllMemoriesList:
    """``GET /api/memories`` — the dashboard's main list, and the leak."""

    def test_no_withheld_row_is_listed(self, client) -> None:
        tc, h = client
        r = tc.get("/api/memories?limit=50", headers=h)
        assert r.status_code == 200, r.text
        ids = _served(r.json())
        assert [i for i in ids if i.startswith("hide-")] == [], (
            f"the list served a withheld row: {ids}"
        )
        assert sorted(i for i in ids if i.startswith("keep-")) == ["keep-1", "keep-2"], (
            f"the profile's real memories disappeared: {ids}"
        )

    def test_the_count_a_user_reads_excludes_them(self, client) -> None:
        """"5,218 memories" was 3,919 real ones plus 1,299 withheld."""
        tc, h = client
        r = tc.get("/api/memories?limit=50", headers=h)
        assert r.json()["total"] == _VISIBLE_IN_MINE

    def test_the_count_matches_the_rows_when_paging(self, client) -> None:
        """A filtered page against an unfiltered total is its own bug.

        It renders as "showing 2 of 4" with two rows missing and nothing to
        click, which reads as data loss to the user rather than as a filter.
        """
        tc, h = client
        body = tc.get("/api/memories?limit=50&offset=0", headers=h).json()
        assert len(_served(body)) == body["total"]

    @pytest.mark.parametrize("scope", ["global", "shared", "all"])
    def test_every_scope_view_filters(self, client, scope: str) -> None:
        """The scope switch rebuilds the WHERE clause — all four branches.

        Mine/shared/global/all are separate string concatenations in this
        handler, so fixing one proves nothing about the others.
        """
        tc, h = client
        r = tc.get(f"/api/memories?limit=50&scope={scope}", headers=h)
        assert r.status_code == 200, r.text
        body = r.json()
        served = _served(body)
        assert [i for i in served if i.startswith("hide")] == [], (
            f"scope={scope} served a withheld row: {served}"
        )
        assert body["total"] == len(served), (
            f"scope={scope} counted {body['total']} and served {len(served)}"
        )


class TestTheCreationTimeline:
    def test_no_withheld_row_becomes_an_event(self, client) -> None:
        tc, h = client
        # 365d is the widest range this route accepts (1d/7d/30d/90d/365d), and
        # the seeded facts are dated inside it.
        r = tc.get("/api/v3/timeline/?range=365d&limit=200", headers=h)
        assert r.status_code == 200, r.text
        ids = [e.get("id") for e in r.json().get("events", [])]
        assert [i for i in ids if isinstance(i, str) and i.startswith("hide-")] == []
        assert any(isinstance(i, str) and i.startswith("keep-") for i in ids), (
            "the timeline returned no real facts either, so this proves nothing"
        )


class TestTheClauseIsOneDefinitionNotTwo:
    def test_the_routes_resolve_it_from_storage(self) -> None:
        """A second copy of the predicate is how the first gap opened."""
        import inspect

        from superlocalmemory.server.routes import memories, timeline

        for mod in (memories, timeline):
            src = inspect.getsource(mod)
            assert "visible_fact_clause_for_connection" in src, (
                f"{mod.__name__} does not use the shared clause"
            )
            assert "COALESCE(quarantined" not in src, (
                f"{mod.__name__} has its own inline copy of the predicate"
            )

    def test_it_is_imported_at_module_level_not_per_request(self) -> None:
        """A request-time import is a 500 waiting for an install to move.

        4.0.10 shipped exactly that bug twice: ``asset_versions`` imported
        inside a route handler took the whole dashboard down when the package
        was reinstalled underneath the running daemon, and ``recall_serializer``
        — imported inside ``recall_trace`` — turned the Recall Lab into an
        Internal Server Error when the daemon could not read the source file it
        was pointed at. Neither module was at fault; the import site was.
        """
        import ast

        for path in (
            "src/superlocalmemory/server/routes/memories.py",
            "src/superlocalmemory/server/routes/timeline.py",
        ):
            tree = ast.parse(open(path, encoding="utf-8").read())
            top = {
                alias.name
                for node in tree.body
                if isinstance(node, ast.ImportFrom)
                for alias in node.names
            }
            assert "visible_fact_clause_for_connection" in top, (
                f"{path} imports the clause inside a function"
            )


class TestItDegradesInsteadOfFailing:
    """A route must not 500 on a store the engine has never opened."""

    @pytest.mark.parametrize("factory", ["tuple", "row", "dict"])
    def test_the_column_probe_survives_any_row_factory(self, factory: str) -> None:
        """The first caller sets ``dict_factory``, so ``row[1]`` raises there.

        Found before this shipped, and worth pinning: the probe's own
        ``except sqlite3.Error`` would not have caught a KeyError, so the
        failure mode was "silently decide the column is absent and drop the
        filter" — the same leak, now with a guard in front of it.
        """
        import sqlite3

        from superlocalmemory.storage.database import (
            visible_fact_clause_for_connection,
        )

        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE atomic_facts "
            "(fact_id TEXT, quarantined INTEGER, archive_status TEXT)"
        )
        if factory == "row":
            conn.row_factory = sqlite3.Row
        elif factory == "dict":
            conn.row_factory = lambda cur, row: {
                d[0]: row[i] for i, d in enumerate(cur.description)
            }

        clause = visible_fact_clause_for_connection(conn)
        assert "quarantined" in clause, f"filter dropped under {factory} factory"
        assert "archive_status" in clause

    def test_an_absent_column_means_no_filter_not_an_exception(self) -> None:
        import sqlite3

        from superlocalmemory.storage.database import (
            visible_fact_clause_for_connection,
        )

        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE atomic_facts (fact_id TEXT)")
        assert visible_fact_clause_for_connection(conn) == ""

    def test_the_escape_hatch_is_still_reachable(self) -> None:
        """Repair, erasure and export must be able to see a withheld row."""
        import sqlite3

        from superlocalmemory.storage.database import (
            visible_fact_clause_for_connection,
        )

        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE atomic_facts (fact_id TEXT, quarantined INTEGER)"
        )
        assert "quarantined" not in visible_fact_clause_for_connection(
            conn, include_quarantined=True,
        )

    def test_the_manager_and_the_connection_agree(self, tmp_path) -> None:
        """Two entry points, one string. If they drift, one surface leaks."""
        import sqlite3

        from superlocalmemory.storage.database import (
            DatabaseManager,
            visible_fact_clause_for_connection,
        )
        from superlocalmemory.storage.schema import create_all_tables

        db_file = tmp_path / "memory.db"
        conn = sqlite3.connect(str(db_file))
        create_all_tables(conn)
        conn.commit()

        mgr = DatabaseManager(str(db_file))
        for prefix in ("", "af"):
            assert mgr.visible_fact_clause(prefix) == (
                visible_fact_clause_for_connection(conn, prefix)
            ), f"the two clause builders disagree at prefix={prefix!r}"
        conn.close()
