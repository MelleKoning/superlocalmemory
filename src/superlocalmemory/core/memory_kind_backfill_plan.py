# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""What a classification run decides for each memory, and which backend it uses.

Pure decisions plus two read-only queries. No thread, no lock, no write.

The per-fact decision (``decide``, in encoding/memory_kind_classifier.py so
saving a memory uses the same rule) — rules against a model:

* A **strong cue** is a rules cue that actually fired on short text
  (``memory_kind_rules.cue_kind``: under 800 characters, an explicit phrase
  such as "never push", "we decided", "status:"). On text like that the cue
  rules scored 0.887 on the persona set against 0.48 for the real Laya model,
  and no model confidence is calibrated yet (``KIND_CALIBRATIONS``: behaviour
  threshold ``None`` everywhere). So a model answer that disagrees with a
  strong cue does not replace it: the rules' kind is kept.
* Where no cue fired (long text, or a plain statement the rules could only map
  from its old type), the model's answer is used: that is the gap a model is
  there to fill.
* When the model agrees with the cue, the model's answer is kept, because it
  carries a confidence the dashboard can show.
* ``correction`` is the one exception: there the model's answer already went
  through the separate yes/no check (cue AND verify >= 0.8, LLD §3.4), which
  exists precisely because the correction cue alone was right only 35 % of
  the time. Whatever that check produced stands.

Every result is a suggestion; nothing here can produce a confirmed kind.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from superlocalmemory.encoding.memory_kind_classifier import (
    MAX_MODEL_FACTS,
    MODEL_SOURCES,
    decide,
)
from superlocalmemory.encoding.memory_kind_recipe import KINDS_V1, LLM_RECIPE, RULES_RECIPE
from superlocalmemory.encoding.memory_kind_rules import suggest_by_rules
from superlocalmemory.storage.memory_kind_store import KindCandidate, KindChange
from superlocalmemory.storage.memory_kinds import (
    AUTHORITY,
    KindAssignment,
    KindSource,
    MemoryKind,
)

#: Plain-language reasons the dashboard and CLI show beside the backend.
REASONS = {
    "off": "Memory kinds are turned off in Settings.",
    "rules": "Typed on this device by simple rules.",
    "laya": "Typed on this device by Laya, the model your answer check already runs.",
    "jev": "Typed online by Jev: memory text is sent to the Jev service.",
    "llm_saved": ("Your local or cloud model types new memories as they are saved; "
                  "existing memories are typed on this device by rules."),
    "laya_not_running": "Laya is chosen, but it is not the running answer check; using rules.",
    "jev_not_ready": ("Jev typing needs Jev as the answer check and its own consent; "
                      "using rules."),
    "llm_not_ready": "The language model is not available in this mode; using rules.",
}


@dataclass(frozen=True, slots=True)
class BackendChoice:
    configured: str
    active: str            # "off" | "rules" | "laya" | "jev"
    reason: str
    leaves_device: bool


def resolve_backfill_backend(cfg: Any, mode: Any, judge: Any,
                             llm_available: bool) -> BackendChoice:
    """The backend a run over existing memories uses right now. Never raises."""
    from superlocalmemory.encoding.memory_kind_classifier import (
        KindBackend,
        resolve_kind_backend,
    )

    configured = getattr(cfg, "backend", "auto")
    configured = configured if isinstance(configured, str) else "auto"
    try:
        backend = resolve_kind_backend(cfg, mode, judge, llm_available)
    except Exception:  # noqa: BLE001 — unknown state reads as the safe backend
        backend = KindBackend.RULES
    if backend is KindBackend.OFF:
        return BackendChoice(configured, "off", REASONS["off"], False)
    if backend is KindBackend.LLM:
        # The extraction call is the only LLM typing path (no extra calls).
        return BackendChoice(configured, "rules", REASONS["llm_saved"], False)
    if backend is KindBackend.JEV:
        return BackendChoice(configured, "jev", REASONS["jev"], True)
    if backend is KindBackend.LAYA:
        return BackendChoice(configured, "laya", REASONS["laya"], False)
    reason = {"laya": REASONS["laya_not_running"], "jev": REASONS["jev_not_ready"],
              "llm": REASONS["llm_not_ready"]}.get(configured, REASONS["rules"])
    return BackendChoice(configured, "rules", reason, False)


#: The longest a batch write may hold the store's write lock (LLD G4).
WRITE_TARGET_MS = 50.0
#: Below this, fixed per-batch costs (a connection, a commit) dominate.
MIN_BATCH = 10
#: Writes per decision. The median ignores a lone spike: SQLite's automatic
#: checkpoint runs inside COMMIT every few batches, after the write lock is
#: released, and its cost does not depend on the batch size.
WINDOW = 3


def next_batch_size(size: int, recent_ms: Sequence[float], configured: int) -> int | None:
    """The next batch size from the last ``WINDOW`` writes at ``size``, or None
    to keep ``size`` and keep collecting. Halve over target, double well under."""
    if len(recent_ms) < WINDOW:
        return None
    median = sorted(recent_ms[-WINDOW:])[WINDOW // 2]
    if median > WRITE_TARGET_MS and size > MIN_BATCH:
        return max(MIN_BATCH, size // 2)
    if median < WRITE_TARGET_MS / 3 and size < configured:
        return min(configured, size * 2)
    return None


def recipe_for(backend: str) -> str:
    return {"laya": KINDS_V1.recipe_id, "jev": KINDS_V1.recipe_id,
            "llm": LLM_RECIPE}.get(backend, RULES_RECIPE)


def _changed(candidate: KindCandidate, new: KindAssignment) -> bool:
    return (candidate.memory_kind != new.kind.value
            or candidate.memory_kind_source != new.source.value)


def _outranks(candidate: KindCandidate, new: KindAssignment) -> bool:
    """Never replace a kind someone confirmed, nor any higher-authority kind."""
    old = candidate.memory_kind_source
    if old is None or candidate.memory_kind is None:
        return True
    try:
        old_source = KindSource(old)
    except ValueError:
        return False          # a source from a newer build: leave it alone
    if old_source in (KindSource.USER, KindSource.CALLER):
        return False
    return AUTHORITY[new.source] >= AUTHORITY[old_source]


def plan_changes(candidates: Sequence[KindCandidate],
                 assignments: Sequence[KindAssignment | None],
                 old_confidence: Mapping[int, float | None],
                 ) -> tuple[list[KindChange], int]:
    """(changes to write, how many candidates need none)."""
    changes: list[KindChange] = []
    for candidate, new in zip(candidates, assignments):
        if new is None or not _outranks(candidate, new) or not _changed(candidate, new):
            continue
        changes.append(KindChange(
            rowid=candidate.rowid, fact_id=candidate.fact_id,
            old_kind=candidate.memory_kind, old_source=candidate.memory_kind_source,
            old_confidence=old_confidence.get(candidate.rowid),
            fact_type=candidate.fact_type, new=new,
        ))
    return changes, len(candidates) - len(changes)


@dataclass(frozen=True, slots=True)
class _Fact:
    """What the classifier reads from a candidate row."""

    content: str
    fact_type: str
    memory_kind: str | None
    memory_kind_source: str | None


def ask_in_chunks(classifier: Any, batch: Sequence[KindCandidate],
                  backend: str) -> list[KindAssignment | None] | None:
    """The classifier's answer per candidate, ``MAX_MODEL_FACTS`` at a time.

    None when a model was due (Laya, Jev) and a chunk came back without a
    model answer: the batch is retried later rather than half-typed. Blank
    text never goes to a model (the classifier would refuse the whole chunk).
    """
    if classifier is None:
        return [suggest_by_rules(c.content, c.fact_type) for c in batch]
    facts = [_Fact(c.content, c.fact_type, c.memory_kind, c.memory_kind_source) for c in batch]
    expect_model = backend in ("laya", "jev")
    size = MAX_MODEL_FACTS if expect_model else max(1, len(facts))
    out: list[KindAssignment | None] = []
    for start in range(0, len(facts), size):
        chunk = facts[start:start + size]
        blank = {i for i, f in enumerate(chunk) if not f.content.strip()}
        ask = [f for i, f in enumerate(chunk) if i not in blank]
        got = list(classifier.suggest(ask, caller_kind=None)) if ask else []
        if len(got) != len(ask):
            return None
        if expect_model and ask and not any(
                a is not None and a.source in MODEL_SOURCES for a in got):
            return None
        answers = iter(got)
        out.extend(suggest_by_rules(f.content, f.fact_type) if i in blank else next(answers)
                   for i, f in enumerate(chunk))
    return out


# ---------------------------------------------------------------------------
# Read-only queries the store does not offer
# ---------------------------------------------------------------------------

def _mode_clause(mode: str) -> tuple[str, tuple[Any, ...]]:
    from superlocalmemory.storage.memory_kind_store import _REFRESH_ELIGIBLE_SOURCES

    if mode == "refresh":
        marks = ", ".join("?" for _ in _REFRESH_ELIGIBLE_SOURCES)
        return (f"(memory_kind IS NULL OR memory_kind_source IN ({marks}))",
                tuple(_REFRESH_ELIGIBLE_SOURCES))
    return "memory_kind IS NULL", ()


def count_candidates(db: Any, profile_id: str, mode: str) -> int:
    """How many facts a run of ``mode`` would look at (same filter as select_batch)."""
    clause, params = _mode_clause(mode)
    tombstones = ""
    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                  "AND name='projection_tombstones'"):
        tombstones = (" AND NOT EXISTS (SELECT 1 FROM projection_tombstones pt "
                      "WHERE pt.profile_id = atomic_facts.profile_id "
                      "AND pt.fact_id = atomic_facts.fact_id)")
    rows = db.execute(
        "SELECT COUNT(*) AS n FROM atomic_facts WHERE profile_id = ? "
        f"AND COALESCE(quarantined, 0) = 0 AND {clause}{tombstones}",
        (profile_id, *params))
    return int(dict(rows[0])["n"]) if rows else 0


def old_confidences(db: Any, profile_id: str,
                    candidates: Sequence[KindCandidate]) -> dict[int, float | None]:
    """The stored confidence of candidates that already carry a kind, so the
    history row (and therefore an undo) restores it exactly."""
    typed = [c.rowid for c in candidates if c.memory_kind is not None]
    if not typed:
        return {}
    marks = ", ".join("?" for _ in typed)
    rows = db.execute(
        "SELECT rowid, memory_kind_confidence FROM atomic_facts "
        f"WHERE profile_id = ? AND rowid IN ({marks})", (profile_id, *typed))
    return {int(dict(r)["rowid"]): dict(r)["memory_kind_confidence"] for r in rows}


def history_rows(db: Any, profile_id: str) -> int:
    rows = db.execute("SELECT COUNT(*) AS n FROM memory_kind_history WHERE profile_id = ?",
                      (profile_id,))
    return int(dict(rows[0])["n"]) if rows else 0


__all__ = ["MAX_MODEL_FACTS", "MIN_BATCH", "MODEL_SOURCES", "REASONS", "WINDOW",
           "WRITE_TARGET_MS", "BackendChoice", "ask_in_chunks", "count_candidates", "decide",
           "history_rows", "next_batch_size", "old_confidences", "plan_changes", "recipe_for",
           "resolve_backfill_backend"]
