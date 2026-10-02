# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M043 — take model-written summaries out of the retrieval corpus.

Runs unattended on upgrade. Three steps, in this order, all idempotent:

  1. PRESERVE — copy every consolidator-authored row's text into
     ``consolidated_summaries``, the display-only table, so nothing the user
     could see today disappears from view.
  2. WITHHOLD — set ``atomic_facts.quarantined = 1`` on those rows. They stay
     on disk, with their provenance, and stop being answers.
  3. RESTORE — un-hide genuine memories whose retention row says they are
     maximally retained but whose zone says 'archive'.

The restore runs when the migration is applied, and never again: once this
store's log records it complete, the standing guard (``repair()``) preserves and
withholds only, because every memory archived after that was archived on
purpose by the running product.

WHY THIS IS A MIGRATION AND NOT A ONE-OFF SCRIPT
------------------------------------------------
About three quarters of this product's users are not engineers. A repair that
requires reading a runbook, cloning a repository or running SQL is a repair
they will not get. The store is the whole point of the product, so a defective
store has to fix itself on the next start, whether they installed by pip, npm
or from source.

DEFERRED, NOT EAGER
-------------------
``atomic_facts`` is bootstrapped by ``MemoryEngine.initialize()``, which runs
*after* ``apply_all``. Every existing migration that touches it — M011, M013,
M015, M016 — is deferred for that reason and this one is no different. Deferred
migrations are snapshotted: ``apply_deferred`` takes a verified backup lazily,
immediately before the first migration that will actually apply
(``migration_runner.py``, ``_ensure_snapshot``), using the sqlite3 backup API
rather than a file copy, which cannot snapshot a live WAL database.

WHAT IT IS REPAIRING, MEASURED
------------------------------
On the author's 5,089-fact store, before this ran:

    rows written by the consolidator into atomic_facts     1,195
      ...retrieval-eligible (retention zone not archive)     307
      ...that are summaries of other summaries              353
      ...carrying a temporal_events row                        0
      ...violating the declared FK to memories             1,195
    genuine memories archived by consolidation              528
      ...archived by anything other than consolidation        0
      ...whose retention score is 1.0 (i.e. "keep")         528

Asked "what am I working on", ranks 1, 2 and 3 all read "Unfortunately, there
is no information available about 'Gateway', 'State', 'Bounded', or 'Claude' in
the provided text."

THE THREE PREDICATES, AND WHY EACH IS EXACT
-------------------------------------------
**Which rows are consolidator-authored.** ``memory_id = ''`` *and* present in
``fact_consolidations``. Verified to agree exactly on the real store: 1,195
either way, zero rows on either difference, and zero genuine facts with a
dangling ``memory_id`` that could be swept up by accident. Both halves are
required so that a future fact with an empty ``memory_id`` — or a consolidation
record pointing at a real fact — is left alone.

Content matching was considered and rejected. Only 34 of the 307
retrieval-eligible rows read as refusals; the other 273 are fluent, plausible,
generic prose that no honest predicate separates from a real summary. Repairing
by content would have cleared a ninth of the problem and looked finished.

**Which memories to un-hide.** ``lifecycle_zone IN ('archive','forgotten')``
together with ``retention_score > 0.8``. That combination is a contradiction on
its own terms: ``math/ebbinghaus.py:lifecycle_zone`` maps any score above 0.8 to
'active', so a row scoring 1.0 and filed under 'archive' was put there by
something that did not consult the score. Consolidation was that something, but
the predicate does not need to know it — which is what makes it safe to re-run
forever, and self-limiting: it describes an inconsistency, so it empties itself.

The restored zone is *recomputed* from the score with the same thresholds, not
guessed. Guessing 'warm' would have quietly demoted 528 facts that the maths
already called 'active'.

**Which summaries to preserve.** All of them, including the refusals. A reader
looking at their dashboard should see what their store actually generated;
silently dropping the embarrassing ones would hide the problem this repair
exists to fix. W5 renders them, and its own tests cover not showing junk as
though it were insight.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

NAME = "M043_quarantine_display_summaries"
DB_TARGET = "memory"

#: Kept short and STABLE. The runner hashes this text and fails a completed
#: migration whose hash has drifted, so it must not be edited to track changes
#: in ``apply()``. The work lives in ``apply()`` because it is conditional data
#: movement that one DDL script cannot express.
DDL = "-- M043: see apply(); preserve, withhold, restore"


#: Retention thresholds, mirroring ``math/ebbinghaus.py::lifecycle_zone`` and
#: ``ForgettingConfig`` defaults (archive_threshold 0.2, forget_threshold 0.05).
#: A migration is SQL and cannot read the user's config, so these are the
#: shipped defaults.
def _zone_from_score(score: str) -> str:
    return f"""
    CASE
        WHEN {score} > 0.8  THEN 'active'
        WHEN {score} > 0.5  THEN 'warm'
        WHEN {score} > 0.2  THEN 'cold'
        WHEN {score} > 0.05 THEN 'archive'
        ELSE 'forgotten'
    END
"""


_ZONE_FROM_SCORE = _zone_from_score("retention_score")

#: Rows the fact consolidator wrote directly into the retrieval corpus.
_CONSOLIDATOR_ROWS = """
    SELECT af.fact_id
      FROM atomic_facts af
     WHERE af.memory_id = ''
       AND af.fact_id IN (
             SELECT consolidated_fact_id FROM fact_consolidations
           )
"""

#: A genuine memory that consolidation hid.
#:
#: BY PROVENANCE, NOT BY SCORE. The first version of this required
#: ``retention_score > 0.8``, reasoning that a score above 0.8 maps to 'active'
#: so zone 'archive' must be a contradiction. Every one of the 528 hidden
#: memories on the author's store scored exactly 1.0, so it worked there — and
#: that store is the LUCKY shape. Those rows scored 1.0 because they had no
#: prior retention row and the schema default is 1.0.
#:
#: On a store where the forgetting scheduler has been running (the default,
#: ``ForgettingConfig.enabled = True``) a consolidation victim carries whatever
#: score it had drifted to — typically 0.21 to 0.80 — because
#: ``set_fact_lifecycle_zone`` moves the zone and leaves the score alone. The
#: score-based predicate does not see those rows at all: they stay archived,
#: ``verify()`` returns true because it cannot see them either, and ``doctor``
#: reports healthy while the memories remain unreachable. A repair that is
#: silent about what it failed to repair is worse than one that fails loudly.
#:
#: So the test is the same one the withhold step uses: was this row a
#: consolidation source? That is recorded, and it does not depend on a number
#: that something else was free to change afterwards.
#:
#: Self-correcting rather than indiscriminate: the zone is RECOMPUTED from the
#: score, and only a row whose recomputed zone is reachable is selected. A
#: source that has genuinely faded is not selected at all -- so it is neither
#: counted as "restored" while it stays where it is, nor left looking like a
#: repair that never finishes. (Selecting it, as an earlier version did, made
#: ``verify()`` false for ever on any store holding one, and every start then
#: re-ran the repair and reported the migration failed.)
_WRONGLY_HIDDEN_BASE = """
    SELECT r.fact_id
      FROM fact_retention r
      JOIN atomic_facts af ON af.fact_id = r.fact_id
     WHERE r.lifecycle_zone IN ('archive', 'forgotten')
       AND af.memory_id <> ''
       AND ({extra})
       AND (""" + _zone_from_score("r.retention_score") + """)
           NOT IN ('archive', 'forgotten')
"""

#: The provenance half. Only usable when the ledger exists, which is why this
#: is composed at call time rather than being one constant: a first version
#: baked the subquery in and ``verify()`` then raised "no such table:
#: fact_consolidations" on a store without a ledger -- caught by the test for
#: exactly that store shape.
#:
#: Only the OLD path's ledger rows count. That path wrote summaries into recall
#: and archived their sources; today's consolidator writes display-only
#: summaries, records them as ``display_summary``, and archives nothing
#: (``core/fact_consolidator.py``). A memory such a summary lists was hidden, if
#: at all, by something else -- the tier manager, the forgetting curve, the
#: owner -- and is not this repair's to undo.
_ARCHIVED_BY_CONSOLIDATION = """
    af.fact_id IN (
        SELECT je.value
          FROM fact_consolidations fc, json_each(fc.source_fact_ids) je
         WHERE json_valid(fc.source_fact_ids){only_old_path}
    )
"""

#: The score half: hidden while the retention maths says to keep.
_SCORED_TO_KEEP = "r.retention_score > 0.8"


def _wrongly_hidden(conn: sqlite3.Connection) -> str:
    """The restore predicate, using whichever halves this store supports."""
    if _table_exists(conn, "fact_consolidations"):
        only_old_path = (
            " AND COALESCE(fc.strategy, 'entity_cluster') <> 'display_summary'"
            if _has_column(conn, "fact_consolidations", "strategy") else ""
        )
        provenance = _ARCHIVED_BY_CONSOLIDATION.format(only_old_path=only_old_path)
        return _WRONGLY_HIDDEN_BASE.format(
            extra=f"{_SCORED_TO_KEEP} OR {provenance}"
        )
    return _WRONGLY_HIDDEN_BASE.format(extra=_SCORED_TO_KEEP)


def apply(conn: sqlite3.Connection) -> None:
    """Preserve, withhold, restore. Atomic, and safe to run again."""
    _run(conn, restore=True)


def _run(conn: sqlite3.Connection, *, restore: bool) -> None:
    if not _table_exists(conn, "atomic_facts"):
        # Nothing to repair on a store whose corpus does not exist yet. Not an
        # error: a fresh install reaches this migration with the table created
        # moments earlier, and a store older than the table has no rows to fix.
        logger.debug("M043: atomic_facts absent, nothing to repair")
        return
    if not _has_column(conn, "atomic_facts", "quarantined"):
        # storage.schema.create_all_tables adds it at every engine init, so
        # reaching here means engine init has not run against this store.
        # Adding it is cheap and keeps the migration independent of that order.
        conn.execute(
            "ALTER TABLE atomic_facts ADD COLUMN quarantined "
            "INTEGER NOT NULL DEFAULT 0"
        )
    if not _table_exists(conn, "consolidated_summaries"):
        # UNCONDITIONALLY, because verify() requires it and verify() is what the
        # runner consults on every later start.
        #
        # A first draft created it only inside _preserve, which is skipped when
        # there is no provenance ledger. On a store with no ledger the migration
        # therefore applied, recorded 'complete', and then failed verify() on
        # the NEXT start -- and since repair() then re-ran apply(), it failed
        # again and was reported as a failed migration forever. Two existing
        # idempotency tests caught it. The lesson generalises: everything
        # verify() asserts has to be produced on every path through apply(),
        # not only on the path that happens to need it.
        #
        # The DDL is imported, not restated, so the two definitions cannot
        # drift apart.
        from superlocalmemory.storage.schema import CONSOLIDATED_SUMMARIES_DDL

        conn.executescript(CONSOLIDATED_SUMMARIES_DDL)

    # No provenance ledger means no consolidator-authored rows can be
    # identified, so there is nothing to preserve or withhold. The restore step
    # is independent of it and still runs — a store can have wrongly-hidden
    # memories without having the ledger that explains how they got that way.
    #
    # Written as a skip rather than an error after the first draft raised here
    # and broke three existing migration-runner tests, which drive
    # apply_deferred against minimal fixtures. A repair that only works on a
    # fully-bootstrapped store is a repair that will not run on the store that
    # needs it most.
    has_ledger = _table_exists(conn, "fact_consolidations")

    conn.execute("BEGIN IMMEDIATE")
    try:
        preserved = _preserve(conn) if has_ledger else 0
        withheld = _withhold(conn) if has_ledger else 0
        restored = _restore(conn) if restore else 0
        conn.execute("COMMIT")
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:  # pragma: no cover — best effort
            pass
        raise

    # INFO, not debug. This changes what the user's next recall returns, so it
    # belongs in the log they can actually see.
    if preserved or withheld or restored:
        logger.info(
            "M043 memory repair: %d summaries preserved for display, "
            "%d withheld from recall, %d memories restored to recall",
            preserved, withheld, restored,
        )


def _preserve(conn: sqlite3.Connection) -> int:
    """Copy consolidator rows into the display table. Returns rows added."""
    before = _count(conn, "SELECT COUNT(*) FROM consolidated_summaries")
    # The entity a summary was written about is not recoverable from the row —
    # entities_json holds the whole cluster's pool — so entity_id is left empty
    # and entity_name carries the summary's own leading label when it has one.
    # source_fact_ids comes from the provenance ledger, which does know.
    conn.execute("""
        INSERT INTO consolidated_summaries
            (summary_id, profile_id, entity_id, entity_name, content,
             source_fact_ids, source_count, char_count, generated_by,
             scope, shared_with, source_earliest, source_latest, created_at)
        SELECT af.fact_id,
               af.profile_id,
               '',
               '',
               af.content,
               COALESCE(
                   (SELECT fc.source_fact_ids FROM fact_consolidations fc
                     WHERE fc.consolidated_fact_id = af.fact_id
                       AND json_valid(fc.source_fact_ids)
                     ORDER BY fc.created_at DESC LIMIT 1),
                   '[]'
               ),
               -- Count only sources that are the owner's OWN memories.
               --
               -- Two wrong answers were tried first. evidence_count reached 450
               -- on one row, because the old reinforce-or-insert path bumped it
               -- every time it saw the same summary text again. The length of
               -- the provenance array is 10, which is closer but still a lie:
               -- 353 of these summaries are summaries OF SUMMARIES, and on the
               -- author's store the first row's ten "sources" are all
               -- consolidator rows written 10-20 seconds apart. Reporting that
               -- as "10 memories summarised" tells the owner this is a digest
               -- of ten things they said, when it is a digest of ten things the
               -- summarizer said.
               --
               -- So: count the sources that have a parent memory. A
               -- summary-of-summaries therefore reports 0, which is the honest
               -- number, and the endpoint ranks it below one that covers real
               -- memories.
               -- json_valid guards every json_each in this file. Without it
               -- ONE malformed source_fact_ids anywhere in the ledger makes
               -- json_each raise, apply() roll back, verify() stay false, and
               -- repair() re-raise on every start for the life of the store —
               -- a single bad row taking the whole repair down with it.
               (SELECT COUNT(*) FROM atomic_facts src
                 WHERE src.memory_id <> ''
                   AND src.fact_id IN (
                       SELECT je.value FROM fact_consolidations fc,
                              json_each(fc.source_fact_ids) je
                        WHERE fc.consolidated_fact_id = af.fact_id
                          AND json_valid(fc.source_fact_ids))),
               LENGTH(af.content),
               'migrated',
               COALESCE(af.scope, 'personal'),
               af.shared_with,
               -- Coverage window over those same genuine sources only.
               -- A first draft spanned every source and produced
               -- "2026-08-18 to 2026-08-18" on a row whose sources were ten
               -- summaries written seconds apart -- the span of a summarisation
               -- run, presented as the stretch of work it covers. NULL when
               -- there are no genuine sources, and the dashboard then shows no
               -- span rather than a false one.
               (SELECT MIN(src.created_at) FROM atomic_facts src
                 WHERE src.memory_id <> ''
                   AND src.fact_id IN (
                       SELECT je.value FROM fact_consolidations fc,
                              json_each(fc.source_fact_ids) je
                        WHERE fc.consolidated_fact_id = af.fact_id
                          AND json_valid(fc.source_fact_ids))),
               (SELECT MAX(src.created_at) FROM atomic_facts src
                 WHERE src.memory_id <> ''
                   AND src.fact_id IN (
                       SELECT je.value FROM fact_consolidations fc,
                              json_each(fc.source_fact_ids) je
                        WHERE fc.consolidated_fact_id = af.fact_id
                          AND json_valid(fc.source_fact_ids))),
               af.created_at
          FROM atomic_facts af
         WHERE af.fact_id IN (""" + _CONSOLIDATOR_ROWS + """)
        -- UNTARGETED. `ON CONFLICT (profile_id, entity_id, content)` names one
        -- constraint and raises on any other, and this table has two: that
        -- unique triple, and summary_id as primary key. summary_id is the
        -- source fact_id, so a row whose CONTENT changed between runs conflicts
        -- on the primary key, which the targeted form does not catch --
        -- reproduced as `IntegrityError: UNIQUE constraint failed:
        -- consolidated_summaries.summary_id`. Because the runner calls
        -- repair() (which is apply()) whenever verify() fails, that failure
        -- would then repeat on every single start, forever.
        --
        -- Untargeted DO NOTHING absorbs both. Losing the newer text is the
        -- right trade: this is a display copy of a row that is about to be
        -- withheld, and the first copy taken is the one the owner has already
        -- been shown.
        ON CONFLICT DO NOTHING
    """)
    return _count(conn, "SELECT COUNT(*) FROM consolidated_summaries") - before


def _withhold(conn: sqlite3.Connection) -> int:
    """Mark consolidator rows quarantined. Returns rows changed."""
    cur = conn.execute(
        "UPDATE atomic_facts SET quarantined = 1 "
        "WHERE COALESCE(quarantined, 0) = 0 "
        "  AND fact_id IN (" + _CONSOLIDATOR_ROWS + ")"
    )
    return int(cur.rowcount or 0)


def _restore(conn: sqlite3.Connection) -> int:
    """Un-hide memories consolidation archived. Returns rows changed.

    Needs both tables: ``fact_retention`` to hold the zone, and
    ``fact_consolidations`` for the provenance half of the predicate. Without
    the ledger it falls back to the score-only test, which is strictly weaker
    but is all a store with no ledger can support.
    """
    if not _table_exists(conn, "fact_retention"):
        return 0
    cur = conn.execute(
        "UPDATE fact_retention SET lifecycle_zone = (" + _ZONE_FROM_SCORE + ") "
        "WHERE fact_id IN (" + _wrongly_hidden(conn) + ")"
    )
    changed = int(cur.rowcount or 0)
    if changed:
        _sync_lifecycle_mirror(conn)
    return changed


def _sync_lifecycle_mirror(conn: sqlite3.Connection) -> None:
    """Bring atomic_facts.lifecycle back in line with the canonical zone.

    Same mapping core/lifecycle_state.py applies: archive/forgotten map to
    'archived', every other zone keeps its own name.
    """
    conn.execute("""
        UPDATE atomic_facts
           SET lifecycle = (
                 SELECT CASE
                          WHEN r.lifecycle_zone IN ('archive', 'forgotten')
                               THEN 'archived'
                          ELSE r.lifecycle_zone
                        END
                   FROM fact_retention r
                  WHERE r.fact_id = atomic_facts.fact_id
               )
         WHERE lifecycle = 'archived'
           AND fact_id IN (
                 SELECT fact_id FROM fact_retention
                  WHERE lifecycle_zone NOT IN ('archive', 'forgotten')
               )
    """)


def unmet(conn: sqlite3.Connection) -> str:
    """Which check does not hold, named. Empty string when all of them do.

    ``verify()`` returns a bare boolean, so when a completed migration stops
    verifying the runner can only say "safe repair did not restore M043". This
    checks five separate things, and that sentence names none of them -- a user
    hitting it had to come back and ask which, and so did we. This is the same
    gap that ``migration_failure_reasons`` closed one level up, left open one
    level down.
    """
    if not _table_exists(conn, "atomic_facts"):
        return ""
    if not _has_column(conn, "atomic_facts", "quarantined"):
        return "atomic_facts has no 'quarantined' column"
    if not _table_exists(conn, "consolidated_summaries"):
        return "the consolidated_summaries display table is missing"
    if _table_exists(conn, "fact_consolidations"):
        n = _count(
            conn,
            "SELECT COUNT(*) FROM atomic_facts WHERE COALESCE(quarantined, 0) = 0 "
            "  AND fact_id IN (" + _CONSOLIDATOR_ROWS + ")",
        )
        if n:
            return f"{n} model-written summaries are not withheld from recall"
        n = _count(conn, """
            SELECT COUNT(*) FROM atomic_facts af
             WHERE af.fact_id IN (""" + _CONSOLIDATOR_ROWS + """)
               AND NOT EXISTS (
                     SELECT 1 FROM consolidated_summaries cs
                      WHERE cs.profile_id = af.profile_id
                        AND (cs.summary_id = af.fact_id
                             OR cs.content = af.content)
                   )
        """)
        if n:
            return f"{n} withheld summaries have no display copy"
    # Once this store's own log records the repair complete, which memories are
    # hidden is decided by the running product -- consolidation archiving the
    # sources of a gist, the tier manager, the forgetting curve, the owner. It
    # is no longer a check on this migration, or every start would undo those
    # decisions (see repair()).
    if _table_exists(conn, "fact_retention") and not _recorded_complete(conn):
        n = _count(conn, "SELECT COUNT(*) FROM (" + _wrongly_hidden(conn) + ")")
        if n:
            return f"{n} real memories are hidden from recall and should not be"
    return ""


def _recorded_complete(conn: sqlite3.Connection) -> bool:
    """Whether this database's own migration_log records M043 complete."""
    if not _table_exists(conn, "migration_log"):
        return False
    row = conn.execute(
        "SELECT status FROM migration_log WHERE name = ?", (NAME,),
    ).fetchone()
    return row is not None and row[0] == "complete"


def blocks_serving(conn: sqlite3.Connection) -> bool:
    """Should a daemon refuse to serve while this check does not hold?

    Only when the SCHEMA is missing. The two schema conditions here -- the
    column and the display table -- mean queries would hit something that is not
    there, so refusing is right. The other three are about DATA: a summary that
    should be withheld is not withheld, or a real memory is hidden. Those make
    some answers worse; they do not stop the store working.

    The distinction matters because this ``verify()`` is a standing guard over
    data that ordinary use can re-violate -- a consolidation pass hiding one more
    memory is enough. Treating that like a missing table meant one drifted row
    could return 503 on every route indefinitely, with a manual restart the only
    way out. That is an outage caused by a quality check, which is worse than the
    thing the check is for.

    Reported as #125, where a user's daemon sat unusable on exactly this.
    """
    if not _table_exists(conn, "atomic_facts"):
        return False
    if not _has_column(conn, "atomic_facts", "quarantined"):
        return True
    return not _table_exists(conn, "consolidated_summaries")


def verify(conn: sqlite3.Connection) -> bool:
    """Whether the repair's end-state holds.

    Called on every start for an already-complete migration. Returning False
    routes to ``repair()``, which makes this a standing guard: if pollution ever
    reappears, the next daemon start withholds it without anyone asking.

    Thin wrapper over ``unmet()`` so the two can never disagree about what
    "verified" means.
    """
    return not unmet(conn)


def repair(conn: sqlite3.Connection) -> None:
    """The standing guard for a completed store: preserve and withhold only.

    The runner calls this only when this store's log already records M043
    complete. It must not restore. That step un-hides archived memories, and
    on a completed store every archived memory was put there by the running
    product after the upgrade: re-running it on every start un-archived the
    sources of each consolidation gist and each memory the tier manager had
    retired -- 3,452 of them on one real store, on one start -- and counted a
    memory the owner had forgotten as "restored" while leaving it where it was.
    The restore ran once, when the migration was first applied, which is what
    it is for. Whether learning.db exists has no bearing on any of this.
    """
    _run(conn, restore=False)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,),
    ).fetchone() is not None


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(
        row[1] == column for row in conn.execute(f"PRAGMA table_info({table})")
    )


def _count(conn: sqlite3.Connection, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    return int(row[0]) if row else 0
