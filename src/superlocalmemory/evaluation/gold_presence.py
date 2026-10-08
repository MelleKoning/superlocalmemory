# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Is the labelled answer actually in the store being measured?

A question whose labelled memory is not in the store cannot be answered by
any search: counting it as a retrieval miss blames recall for a missing write.
Before scoring, every answerable question is checked against the store (read
only): which of its labelled ids are present in the measured profile, and when
they were written.

Statuses:
  stored            every labelled id is present in the measured profile
  partially_stored  at least one labelled id is present, not all
  not_stored        none is present — "answer not stored", never a miss

Only ids, statuses and timestamps are read; never memory text.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from superlocalmemory.evaluation.answer_quality import GoldQuestion

STORED = "stored"
PARTIALLY_STORED = "partially_stored"
NOT_STORED = "not_stored"


@dataclass(frozen=True)
class Presence:
    status: str
    present_ids: tuple[str, ...]
    missing_ids: tuple[str, ...]
    earliest_created_at: str | None
    latest_created_at: str | None

    def as_row(self) -> dict:
        return {"status": self.status, "present": len(self.present_ids),
                "missing_ids": list(self.missing_ids),
                "earliest_created_at": self.earliest_created_at,
                "latest_created_at": self.latest_created_at}


def _placeholders(ids: Sequence[str]) -> str:
    return ",".join("?" for _ in ids)


#: Ids bound per statement: far below every SQLite build's variable limit.
_CHUNK = 500


def _rows(conn: sqlite3.Connection, table: str, key: str, ids: Sequence[str],
          profile_id: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for start in range(0, len(ids), _CHUNK):
        part = ids[start:start + _CHUNK]
        rows = conn.execute(
            f"SELECT {key}, created_at FROM {table} "
            f"WHERE profile_id = ? AND {key} IN ({_placeholders(part)})",
            (profile_id, *part)).fetchall()
        out.update({str(r[0]): str(r[1] or "") for r in rows})
    return out


def _memory_rows(conn: sqlite3.Connection, ids: Sequence[str],
                 profile_id: str) -> dict[str, str]:
    return _rows(conn, "memories", "memory_id", ids, profile_id)


def _fact_rows(conn: sqlite3.Connection, ids: Sequence[str],
               profile_id: str) -> dict[str, str]:
    return _rows(conn, "atomic_facts", "fact_id", ids, profile_id)


def _presence(question: GoldQuestion, memories: Mapping[str, str],
              facts: Mapping[str, str]) -> Presence:
    # One id labelled as both a memory and a fact is one labelled answer.
    labelled = list(dict.fromkeys(sorted(question.gold_memory_ids)
                                  + sorted(question.gold_fact_ids)))
    found = {**{m: memories[m] for m in question.gold_memory_ids if m in memories},
             **{f: facts[f] for f in question.gold_fact_ids if f in facts}}
    present = tuple(i for i in labelled if i in found)
    missing = tuple(i for i in labelled if i not in found)
    if not present:
        status = NOT_STORED
    elif missing:
        status = PARTIALLY_STORED
    else:
        status = STORED
    dates = sorted(d for d in found.values() if d)
    return Presence(status=status, present_ids=present, missing_ids=missing,
                    earliest_created_at=dates[0] if dates else None,
                    latest_created_at=dates[-1] if dates else None)


def check_presence(db_path: Path, questions: Iterable[GoldQuestion],
                   profile_id: str) -> dict[str, Presence]:
    """Presence of every answerable question's labelled ids, read only."""
    answerable = [q for q in questions if q.answerable]
    memory_ids = sorted({m for q in answerable for m in q.gold_memory_ids})
    fact_ids = sorted({f for q in answerable for f in q.gold_fact_ids})
    conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    try:
        memories = _memory_rows(conn, memory_ids, profile_id)
        facts = _fact_rows(conn, fact_ids, profile_id)
    finally:
        conn.close()
    return {q.qid: _presence(q, memories, facts) for q in answerable}


def presence_summary(presence: Mapping[str, Presence],
                     ranks: Mapping[str, int | None]) -> dict:
    """Counts per status; misses split into "answer not stored" and real misses."""
    counts = {STORED: 0, PARTIALLY_STORED: 0, NOT_STORED: 0}
    for item in presence.values():
        counts[item.status] += 1
    not_stored = sorted(q for q, p in presence.items() if p.status == NOT_STORED)
    missed = sorted(q for q in presence if ranks.get(q) is None)
    return {
        "checked": len(presence),
        **counts,
        "answer_not_stored": not_stored,
        "retrieval_misses": [q for q in missed if presence[q].status != NOT_STORED],
        "misses_explained_by_storage": [q for q in missed if presence[q].status == NOT_STORED],
    }
