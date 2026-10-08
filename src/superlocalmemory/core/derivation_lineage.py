# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Capture derivation lineage without inventing source spans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


def _lineage_id(profile_id: str, object_type: str, object_id: str, operation_id: str) -> str:
    value = f"{profile_id}\0{object_type}\0{object_id}\0{operation_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _table_exists(db: Any, table: str) -> bool:
    return bool(db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,),
    ))


_COMPARED = ("derivation_version", "source_status", "source_start", "source_end",
             "source_text_sha256", "source_fact_ids_json", "unresolved_reason")


def _row_values(record: dict[str, Any]) -> tuple[Any, ...]:
    return (record["derivation_version"], record["source_status"],
            record.get("source_start"), record.get("source_end"),
            record.get("source_text_sha256", ""),
            json.dumps(list(record.get("source_fact_ids", ())), separators=(",", ":")),
            record.get("unresolved_reason", ""))


def _stored_rows(db: Any, profile_id: str, operation_id: str) -> dict[str, tuple[Any, ...]]:
    """This operation's lineage rows as stored now (one indexed read)."""
    cols = ",".join(_COMPARED)
    return {str(row["lineage_id"]): tuple(row[c] for c in _COMPARED) for row in db.execute(
        f"SELECT lineage_id,{cols} FROM derivation_lineage "
        "WHERE operation_id=? AND profile_id=?", (operation_id, profile_id))}


def _record(
    db: Any,
    *,
    profile_id: str,
    object_type: str,
    object_id: str,
    operation_id: str,
    derivation_version: str,
    source_status: str,
    source_start: int | None = None,
    source_end: int | None = None,
    source_text_sha256: str = "",
    source_fact_ids: tuple[str, ...] = (),
    unresolved_reason: str = "",
) -> None:
    db.execute(
        "INSERT OR REPLACE INTO derivation_lineage "
        "(lineage_id,profile_id,object_type,object_id,operation_id,"
        "derivation_version,source_status,source_start,source_end,"
        "source_text_sha256,source_fact_ids_json,unresolved_reason) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            _lineage_id(profile_id, object_type, object_id, operation_id),
            profile_id, object_type, object_id, operation_id,
            derivation_version, source_status, source_start, source_end,
            source_text_sha256,
            json.dumps(list(source_fact_ids), separators=(",", ":")),
            unresolved_reason,
        ),
    )


def _fact_ids(value: Any) -> tuple[str, ...]:
    try:
        decoded = json.loads(value or "[]")
    except (TypeError, ValueError):
        return ()
    if not isinstance(decoded, list):
        return ()
    return tuple(str(item) for item in decoded)


#: Bound parameters per query (SQLite's default limit is 999).
_CHUNK = 400


def _chunks(values: tuple[str, ...]) -> list[tuple[str, ...]]:
    return [values[i:i + _CHUNK] for i in range(0, len(values), _CHUNK)]


def _content_sha(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LineagePlan:
    """Everything one checkpoint will write, worked out with reads only.

    Built before the checkpoint's write transaction opens and applied inside
    it (``apply_operation_lineage``), so the write lock is held for the writes
    alone. On a 22k-fact store the reads (every graph edge, summary, scene and
    lexical row of the profile) held it 2.6-4.4 s per remember.
    """

    operation_id: str
    profile_id: str
    raw_content: str
    fact_ids: tuple[str, ...]
    derivation_version: str
    present: bool
    records: tuple[dict[str, Any], ...] = ()
    #: ``(fact_id, reason)`` to withhold, atomically with the lineage rows.
    withhold: tuple[tuple[str, str], ...] = ()
    #: What each fact said when the plan was made (``None`` = missing).
    fact_hashes: tuple[tuple[str, str | None], ...] = ()

    def matches(self, *, operation_id: str, profile_id: str, raw_content: str,
                fact_ids: tuple[str, ...], derivation_version: str) -> bool:
        return (self.operation_id, self.profile_id, self.raw_content, self.fact_ids,
                self.derivation_version) == (operation_id, profile_id, raw_content,
                                             tuple(fact_ids), derivation_version)


def _fact_contents(db: Any, profile_id: str, fact_ids: tuple[str, ...]) -> dict[str, str]:
    contents: dict[str, str] = {}
    for chunk in _chunks(tuple(dict.fromkeys(fact_ids))):
        marks = ",".join("?" for _ in chunk)
        for row in db.execute(
            f"SELECT fact_id, content FROM atomic_facts WHERE profile_id=? "
            f"AND fact_id IN ({marks})", (profile_id, *chunk),
        ):
            contents[str(row["fact_id"])] = str(row["content"])
    return contents


def _derived_rows(
    db: Any, *, table: str, id_column: str, object_type: str, profile_id: str,
    operation_id: str, derivation_version: str, operation_fact_ids: frozenset[str],
) -> list[dict[str, Any]]:
    """Lineage of the summaries/scenes built from this operation's facts.

    Only rows whose JSON mentions one of the operation's ids are read (a
    superset, prefiltered in SQLite); each is then decoded and matched exactly
    as before. It used to decode every row of the profile.
    """
    if not operation_fact_ids or not _table_exists(db, table):
        return []
    found: dict[str, Any] = {}
    for chunk in _chunks(tuple(sorted(operation_fact_ids))):
        mentions = " OR ".join("instr(fact_ids_json, ?) > 0" for _ in chunk)
        for row in db.execute(
            f'SELECT "{id_column}",fact_ids_json FROM "{table}" '
            f"WHERE profile_id=? AND ({mentions})", (profile_id, *chunk),
        ):
            found[str(row[id_column])] = row["fact_ids_json"]
    out = []
    for object_id, fact_ids_json in found.items():
        source_ids = tuple(fid for fid in _fact_ids(fact_ids_json) if fid in operation_fact_ids)
        if source_ids:
            out.append(dict(
                profile_id=profile_id, object_type=object_type, object_id=object_id,
                operation_id=operation_id, derivation_version=derivation_version,
                source_status="derived_from_facts", source_fact_ids=source_ids,
                unresolved_reason="direct_span_not_applicable"))
    return out


def _edge_rows(db: Any, *, profile_id: str, operation_id: str, derivation_version: str,
               operation_fact_ids: frozenset[str]) -> list[dict[str, Any]]:
    """Lineage of the graph edges touching this operation's facts (indexed reads)."""
    if not operation_fact_ids or not _table_exists(db, "graph_edges"):
        return []
    edges: dict[str, tuple[str, str]] = {}
    for chunk in _chunks(tuple(sorted(operation_fact_ids))):
        marks = ",".join("?" for _ in chunk)
        for column in ("source_id", "target_id"):
            for row in db.execute(
                f"SELECT edge_id,source_id,target_id FROM graph_edges "
                f"WHERE profile_id=? AND {column} IN ({marks})", (profile_id, *chunk),
            ):
                edges[str(row["edge_id"])] = (str(row["source_id"]), str(row["target_id"]))
    out = []
    for edge_id, ends in edges.items():
        source_ids = tuple(value for value in ends if value in operation_fact_ids)
        if source_ids:
            out.append(dict(
                profile_id=profile_id, object_type="graph_edge", object_id=edge_id,
                operation_id=operation_id, derivation_version=derivation_version,
                source_status="derived_from_facts", source_fact_ids=source_ids,
                unresolved_reason="direct_span_not_applicable"))
    return out


def _indexed_ids(db: Any, profile_id: str, fact_ids: frozenset[str]) -> set[str]:
    if not fact_ids or not _table_exists(db, "bm25_tokens"):
        return set()
    out: set[str] = set()
    for chunk in _chunks(tuple(sorted(fact_ids))):
        marks = ",".join("?" for _ in chunk)
        out.update(str(row["fact_id"]) for row in db.execute(
            f"SELECT fact_id FROM bm25_tokens WHERE profile_id=? AND fact_id IN ({marks})",
            (profile_id, *chunk)))
    return out


def _fact_record(base: dict[str, Any], fact_id: str, content: str | None,
                 raw_content: str, db: Any) -> tuple[dict[str, Any], tuple[str, str] | None]:
    """One fact's lineage row, and ``(fact_id, reason)`` when it must be withheld."""
    record = dict(base, object_type="fact", object_id=fact_id)
    if content is None:
        return dict(record, source_status="unresolved",
                    unresolved_reason="final_fact_missing"), None
    start = raw_content.find(content)
    if start >= 0:
        return dict(record, source_status="exact", source_start=start,
                    source_end=start + len(content),
                    source_text_sha256=_content_sha(content)), None
    # A derived fact must still say what the source said; one that
    # turns 2004.6 ms into a date is withheld, other doubts are marked.
    from superlocalmemory.core.source_fidelity_guard import judge_fidelity

    reason, withhold = judge_fidelity(db, profile_id=base["profile_id"], fact_id=fact_id,
                                      content=content, raw_content=raw_content)
    reason = reason or "no_exact_span_in_raw_source"
    return (dict(record, source_status="unresolved", unresolved_reason=reason),
            (fact_id, reason) if withhold else None)


def plan_operation_lineage(
    db: Any,
    *,
    operation_id: str,
    profile_id: str,
    raw_content: str,
    fact_ids: tuple[str, ...],
    derivation_version: str,
) -> LineagePlan:
    """Work out an operation's lineage with reads only (no write lock needed)."""
    fact_ids = tuple(fact_ids)
    head = dict(operation_id=operation_id, profile_id=profile_id, raw_content=raw_content,
                fact_ids=fact_ids, derivation_version=derivation_version)
    if not _table_exists(db, "derivation_lineage"):
        return LineagePlan(**head, present=False)
    base = dict(profile_id=profile_id, operation_id=operation_id,
                derivation_version=derivation_version)
    contents = _fact_contents(db, profile_id, fact_ids)
    records: list[dict[str, Any]] = []
    withhold: list[tuple[str, str]] = []
    by_fact: dict[str, dict[str, Any]] = {}
    for fact_id in fact_ids:
        record, held = _fact_record(base, fact_id, contents.get(fact_id), raw_content, db)
        records.append(record)
        by_fact[fact_id] = record
        if held is not None:
            withhold.append(held)
    records.append(dict(base, object_type="profile", object_id=profile_id,
                        source_status="not_applicable",
                        unresolved_reason="profile_scope_not_derived_from_source_span"))
    operation_fact_ids = frozenset(fact_ids)
    derived = dict(profile_id=profile_id, operation_id=operation_id,
                   derivation_version=derivation_version, operation_fact_ids=operation_fact_ids)
    records.extend(_derived_rows(db, table="entity_profiles", id_column="profile_entry_id",
                                 object_type="entity_summary", **derived))
    records.extend(_derived_rows(db, table="memory_scenes", id_column="scene_id",
                                 object_type="memory_scene", **derived))
    records.extend(_edge_rows(db, profile_id=profile_id, operation_id=operation_id,
                              derivation_version=derivation_version,
                              operation_fact_ids=operation_fact_ids))
    for fact_id in sorted(_indexed_ids(db, profile_id, operation_fact_ids)):
        lineage = by_fact.get(fact_id, {})
        records.append(dict(
            base, object_type="index_bm25", object_id=fact_id,
            source_status=str(lineage.get("source_status", "unresolved")),
            source_start=lineage.get("source_start"), source_end=lineage.get("source_end"),
            source_text_sha256=str(lineage.get("source_text_sha256", "")),
            source_fact_ids=(fact_id,),
            unresolved_reason=str(lineage.get("unresolved_reason",
                                              "source_fact_lineage_missing"))))
    hashes = tuple((fid, _content_sha(contents[fid]) if fid in contents else None)
                   for fid in fact_ids)
    return LineagePlan(**head, present=True, records=tuple(records),
                       withhold=tuple(withhold), fact_hashes=hashes)


def _still_current(db: Any, plan: LineagePlan) -> bool:
    """Whether the facts still say what they said when ``plan`` was made.

    Read inside the write transaction (one indexed read per fact), so an edit,
    an erasure or a release that landed in between is never overwritten by a
    stale plan: the plan is then simply worked out again, under the lock.
    """
    contents = _fact_contents(db, plan.profile_id, plan.fact_ids)
    now = tuple((fid, _content_sha(contents[fid]) if fid in contents else None)
                for fid in plan.fact_ids)
    if now != plan.fact_hashes:
        return False
    if plan.withhold:
        from superlocalmemory.core.source_fidelity_guard import is_released

        return not any(is_released(db, profile_id=plan.profile_id, fact_id=fid)
                       for fid, _reason in plan.withhold)
    return True


def apply_operation_lineage(db: Any, plan: LineagePlan) -> None:
    """Write a plan: withheld facts and every lineage row (writes only).

    Call inside the checkpoint's write transaction so the withholding of a
    ``number_became_date`` fact commits atomically with its lineage and the
    checkpoint itself, exactly as when it was all done in there.
    """
    if not plan.present:
        return
    if not _still_current(db, plan):
        plan = plan_operation_lineage(
            db, operation_id=plan.operation_id, profile_id=plan.profile_id,
            raw_content=plan.raw_content, fact_ids=plan.fact_ids,
            derivation_version=plan.derivation_version)
    from superlocalmemory.core.source_fidelity_guard import withhold_fact

    for fact_id, reason in plan.withhold:
        withhold_fact(db, profile_id=plan.profile_id, fact_id=fact_id, reason=reason)
    # Each materialization checkpoints twice with the same facts; the second
    # rewrote every row (500-1,300 on a large store) for nothing. Only rows
    # that are new or different are written.
    stored = _stored_rows(db, plan.profile_id, plan.operation_id)
    for record in plan.records:
        lineage_id = _lineage_id(record["profile_id"], record["object_type"],
                                 record["object_id"], record["operation_id"])
        if stored.get(lineage_id) != _row_values(record):
            _record(db, **record)


def capture_operation_lineage(
    db: Any,
    *,
    operation_id: str,
    profile_id: str,
    raw_content: str,
    fact_ids: tuple[str, ...],
    derivation_version: str,
) -> None:
    """Persist operation lineage while distinguishing spans from derivations."""
    apply_operation_lineage(db, plan_operation_lineage(
        db, operation_id=operation_id, profile_id=profile_id, raw_content=raw_content,
        fact_ids=fact_ids, derivation_version=derivation_version))


def lineage_plan_for(db: Any, operation: Any, fact_ids: tuple[str, ...],
                     derivation_version: str) -> LineagePlan:
    """The plan a checkpoint of ``operation`` with these final facts will write."""
    return plan_operation_lineage(
        db, operation_id=operation.operation_id, profile_id=operation.profile_id,
        raw_content=operation.raw_content, fact_ids=tuple(fact_ids),
        derivation_version=derivation_version)


def apply_checkpoint_lineage(db: Any, operation: Any, plan: LineagePlan | None) -> None:
    """Inside a checkpoint: write ``plan`` if it is this checkpoint's, else work it out."""
    if plan is None or not plan.matches(
            operation_id=operation.operation_id, profile_id=operation.profile_id,
            raw_content=operation.raw_content, fact_ids=tuple(operation.final_fact_ids),
            derivation_version=operation.derivation_version):
        plan = lineage_plan_for(db, operation, operation.final_fact_ids,
                                operation.derivation_version)
    apply_operation_lineage(db, plan)


def checkpoint_with_lineage(repository: Any, operation: Any, *,
                            final_fact_ids: tuple[str, ...], derivation_version: str,
                            **checkpoint: Any) -> Any:
    """A materializer checkpoint: lineage read first, then ONE short write.

    The checkpoint row, any withheld fact and every lineage row commit
    together, instead of one commit per row with the reads in between.
    Called outside any transaction (the materializer is a saga of short
    database operations, never one long transaction).
    """
    plan = lineage_plan_for(repository.db, operation, final_fact_ids, derivation_version)
    with repository.db.transaction():
        return repository.checkpoint_enriching(
            operation.operation_id, final_fact_ids=final_fact_ids,
            derivation_version=derivation_version, lineage_plan=plan, **checkpoint)


__all__ = ["LineagePlan", "apply_checkpoint_lineage", "checkpoint_with_lineage", "apply_operation_lineage",
           "capture_operation_lineage", "lineage_plan_for", "plan_operation_lineage"]
