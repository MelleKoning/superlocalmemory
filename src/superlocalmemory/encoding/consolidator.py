# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Memory Consolidator.

Mem0-style ADD/UPDATE/SUPERSEDE/NOOP classification for incoming facts.

UPDATE and SUPERSEDE are now *proposal classifications*: they persist the
incoming fact but never rewrite, archive, lower trust for, or otherwise mutate
the matched fact.  A reviewed correction owner may later apply a case through
the correction ledger.  This keeps ingestion useful while preventing an
automatic model judgement from changing the historical record.

Mode A: keyword-based contradiction detection (zero LLM).
Mode B/C: LLM-assisted contradiction detection when available.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
import math
from typing import Any, Protocol

from superlocalmemory.core.config import EncodingConfig
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import (
    AtomicFact,
    ConsolidationAction,
    ConsolidationActionType,
    EdgeType,
    GraphEdge,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocols (avoid tight coupling to concrete classes)
# ---------------------------------------------------------------------------

class Embedder(Protocol):
    """Anything that produces an embedding vector from text."""

    def encode(self, text: str) -> list[float]: ...


class LLM(Protocol):
    """Anything that can generate text from a prompt."""

    def generate(self, prompt: str, system: str = "") -> str: ...

    def is_available(self) -> bool: ...


# ---------------------------------------------------------------------------
# Negation patterns for Mode A keyword contradiction detection
# ---------------------------------------------------------------------------

_NEGATION_MARKERS: frozenset[str] = frozenset({
    "not", "no longer", "never", "stopped", "quit", "left",
    "changed", "moved", "divorced", "fired", "resigned",
    "broke up", "ended", "cancelled", "dropped", "switched",
    "former", "ex-", "previously", "used to", "no more",
})

# Score thresholds (match EncodingConfig defaults)
_NOOP_THRESHOLD = 0.95
_MATCH_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# MemoryConsolidator
# ---------------------------------------------------------------------------

class MemoryConsolidator:
    """Decides ADD/UPDATE/SUPERSEDE/NOOP for each incoming fact.

    For each new fact the consolidator:
    1. Finds candidate matches via entity overlap + semantic similarity.
    2. Scores each candidate (Jaccard entity + cosine embedding).
    3. Classifies the relationship and executes the action.
    4. Logs every decision in ``consolidation_log``.

    Thread-safe — delegates all DB ops to DatabaseManager which holds a lock.
    """

    def __init__(
        self,
        db: DatabaseManager,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        config: EncodingConfig | None = None,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._llm = llm
        self._cfg = config or EncodingConfig()
        #: Exact nearest-neighbour source (sqlite-vec or the in-memory index),
        #: shared with recall. None: fall back to reading every fact.
        self._vectors: Any = None

    def use_vector_source(self, vectors: Any) -> None:
        """Search neighbours through recall's vector source instead of reading
        and decoding every fact of the profile, once per new fact (measured on
        a 21,847-fact store: about 7.7 s per fact, twice per fact)."""
        self._vectors = vectors

    def _similar(
        self, new_fact: AtomicFact, profile_id: str, k: int, exclude: set[str],
    ) -> list[tuple[AtomicFact, float]]:
        """The ``k`` most similar visible facts with cosine > 0.5, best first.

        Through the vector source: an exact top-k, then each hit loaded through
        the profile's visibility rules (``get_facts_by_ids``), so a withheld or
        deleted fact is never a candidate. The same set the full scan returns
        whenever that scan saw every fact (stores under its 50,000 cap).
        """
        skip = set(exclude) | {new_fact.fact_id}
        vectors = self._vectors
        if vectors is not None and getattr(vectors, "available", True) and k > 0:
            try:
                hits = vectors.search(new_fact.embedding, top_k=k + len(skip),
                                      profile_id=profile_id)
            except Exception as exc:  # noqa: BLE001 — fall back, never fail a save
                logger.debug("consolidation neighbour search fell back (%s)",
                             type(exc).__name__)
            else:
                ids = [fid for fid, _ in hits if fid not in skip]
                facts = {f.fact_id: f for f in self._db.get_facts_by_ids(ids, profile_id)}
                out = []
                for fid in ids:
                    fact = facts.get(fid)
                    if fact is None or fact.embedding is None:
                        continue
                    sim = _compute_similarity(new_fact.embedding, fact.embedding)
                    if sim > 0.5:
                        out.append((fact, sim))
                out.sort(key=lambda t: (-t[1], t[0].fact_id))
                return out[:k]
        scored = []
        for fact in self._db.get_all_facts(profile_id):
            if fact.fact_id in skip:
                continue
            sim = _compute_similarity(new_fact.embedding, fact.embedding)
            if sim > 0.5:
                scored.append((fact, sim))
        scored.sort(key=lambda t: (-t[1], t[0].fact_id))
        return scored[:k]

    # -- Public API ---------------------------------------------------------

    def consolidate(
        self,
        new_fact: AtomicFact,
        profile_id: str,
        *,
        exclude_fact_ids: set[str] | frozenset[str] | None = None,
    ) -> ConsolidationAction:
        """Consolidate *new_fact* against existing knowledge.

        Returns a ``ConsolidationAction`` describing what was done.

        ``exclude_fact_ids`` is reserved for an ingestion operation's own
        queryable projection.  That projection is evidence being promoted,
        not pre-existing knowledge, and must not suppress its derived facts as
        near-duplicates.
        """
        candidates = self._find_candidates(
            new_fact,
            profile_id,
            exclude_fact_ids=exclude_fact_ids,
        )

        if not candidates:
            return self._execute_add(new_fact, profile_id, reason="no matching facts")

        best_fact, best_score = candidates[0]

        if best_score > _NOOP_THRESHOLD:
            # Even near-duplicates might be contradictions (e.g. negation).
            # Check contradiction BEFORE declaring NOOP.
            if self._is_contradicting(new_fact, best_fact):
                return self._execute_supersede(
                    new_fact, best_fact, profile_id,
                    reason=f"contradiction detected (score={best_score:.3f})",
                )
            return self._execute_noop(
                new_fact, best_fact, profile_id,
                reason=f"near-duplicate (score={best_score:.3f})",
            )

        if best_score > _MATCH_THRESHOLD:
            if self._is_contradicting(new_fact, best_fact):
                return self._execute_supersede(
                    new_fact, best_fact, profile_id,
                    reason=f"contradiction detected (score={best_score:.3f})",
                )
            if new_fact.fact_type == best_fact.fact_type:
                return self._execute_update(
                    new_fact, best_fact, profile_id,
                    reason=f"refines existing (score={best_score:.3f})",
                )

        return self._execute_add(
            new_fact, profile_id,
            reason=f"new information (best_score={best_score:.3f})",
        )

    def get_consolidation_history(
        self, profile_id: str, limit: int = 50,
    ) -> list[ConsolidationAction]:
        """Recent consolidation actions for audit/debugging."""
        rows = self._db.execute(
            "SELECT * FROM consolidation_log WHERE profile_id = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (profile_id, limit),
        )
        return [
            ConsolidationAction(
                action_id=(d := dict(r))["action_id"],
                profile_id=d["profile_id"],
                action_type=ConsolidationActionType(d["action_type"]),
                new_fact_id=d["new_fact_id"],
                existing_fact_id=d["existing_fact_id"],
                reason=d["reason"],
                timestamp=d["timestamp"],
            )
            for r in rows
        ]

    # -- Candidate search ---------------------------------------------------

    def _find_candidates(
        self,
        new_fact: AtomicFact,
        profile_id: str,
        *,
        exclude_fact_ids: set[str] | frozenset[str] | None = None,
    ) -> list[tuple[AtomicFact, float]]:
        """Find and score candidate matches from existing facts.

        Uses two signals:
        1. Entity overlap — ``get_facts_by_entity`` for each canonical entity.
        2. Semantic similarity — cosine of embedding vectors.

        Returns sorted list of (fact, combined_score), descending.
        """
        seen_ids: set[str] = set(exclude_fact_ids or ())
        candidate_facts: list[AtomicFact] = []

        # --- entity-based candidates ---
        for entity in new_fact.canonical_entities:
            for fact in self._db.get_facts_by_entity(entity, profile_id):
                if fact.fact_id not in seen_ids:
                    seen_ids.add(fact.fact_id)
                    candidate_facts.append(fact)

        # --- semantic candidates (top-K by embedding) ---
        if new_fact.embedding is not None and self._embedder is not None:
            semantic_scored = self._similar(new_fact, profile_id,
                                            self._cfg.max_consolidation_candidates, seen_ids)
            for fact, _ in semantic_scored:
                if fact.fact_id not in seen_ids:
                    seen_ids.add(fact.fact_id)
                    candidate_facts.append(fact)

        # --- score all candidates ---
        scored: list[tuple[AtomicFact, float]] = []
        for cand in candidate_facts:
            entity_overlap = _jaccard(
                set(new_fact.canonical_entities),
                set(cand.canonical_entities),
            )
            semantic_sim = _compute_similarity(
                new_fact.embedding, cand.embedding,
            )
            combined = 0.4 * entity_overlap + 0.6 * semantic_sim
            scored.append((cand, combined))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

    # -- Contradiction detection --------------------------------------------

    def _is_contradicting(
        self, fact_a: AtomicFact, fact_b: AtomicFact,
    ) -> bool:
        """Detect if *fact_a* contradicts *fact_b*.

        Mode B/C: delegates to LLM for nuanced judgment.
        Mode A: keyword-based negation detection.
        """
        if self._llm is not None and self._llm.is_available():
            return self._llm_contradiction_check(fact_a, fact_b)
        return self._keyword_contradiction_check(fact_a, fact_b)

    def _keyword_contradiction_check(
        self, fact_a: AtomicFact, fact_b: AtomicFact,
    ) -> bool:
        """Heuristic: check negation markers in either fact's content."""
        text_a = fact_a.content.lower()
        text_b = fact_b.content.lower()
        for marker in _NEGATION_MARKERS:
            if marker in text_a and marker not in text_b:
                return True
            if marker in text_b and marker not in text_a:
                return True

        # Opposing emotional valence with same entities
        shared_entities = set(fact_a.canonical_entities) & set(fact_b.canonical_entities)
        if shared_entities:
            valence_diff = abs(fact_a.emotional_valence - fact_b.emotional_valence)
            if valence_diff > 1.2:
                return True

        return False

    def _llm_contradiction_check(
        self, fact_a: AtomicFact, fact_b: AtomicFact,
    ) -> bool:
        """Ask the LLM whether two facts contradict each other."""
        assert self._llm is not None  # guarded by caller
        prompt = (
            "Do these two statements contradict each other?\n\n"
            f"Statement A: {fact_a.content}\n"
            f"Statement B: {fact_b.content}\n\n"
            "Answer ONLY 'yes' or 'no'."
        )
        response = self._llm.generate(
            prompt, system="You are a precise fact-checker.",
        )
        return response.strip().lower().startswith("yes")

    # -- Action executors ---------------------------------------------------

    def _execute_add(
        self, new_fact: AtomicFact, profile_id: str, *, reason: str,
    ) -> ConsolidationAction:
        """Store the new fact and link to related facts via semantic edges."""
        self._db.store_fact(new_fact)
        self._create_semantic_edges(new_fact, profile_id)
        action = self._log_action(
            ConsolidationActionType.ADD, new_fact.fact_id, "", profile_id, reason,
        )
        logger.debug("ADD fact %s: %s", new_fact.fact_id, reason)
        return action

    def _execute_update(
        self,
        new_fact: AtomicFact,
        existing: AtomicFact,
        profile_id: str,
        *,
        reason: str,
    ) -> ConsolidationAction:
        """Persist a proposed refinement without rewriting the predecessor."""
        self._db.store_fact(new_fact)
        self._create_semantic_edges(new_fact, profile_id)
        action = self._log_action(
            ConsolidationActionType.UPDATE,
            new_fact.fact_id, existing.fact_id,
            profile_id, reason,
        )
        logger.debug(
            "UPDATE proposal %s -> %s: %s",
            existing.fact_id, new_fact.fact_id, reason,
        )
        return action

    def _execute_supersede(
        self,
        new_fact: AtomicFact,
        existing: AtomicFact,
        profile_id: str,
        *,
        reason: str,
    ) -> ConsolidationAction:
        """Persist a proposed successor without changing the predecessor."""
        self._db.store_fact(new_fact)
        self._create_semantic_edges(new_fact, profile_id)
        action = self._log_action(
            ConsolidationActionType.SUPERSEDE,
            new_fact.fact_id, existing.fact_id,
            profile_id, reason,
        )
        logger.debug(
            "SUPERSEDE proposal %s -> %s: %s",
            existing.fact_id, new_fact.fact_id, reason,
        )
        return action

    def _execute_noop(
        self,
        new_fact: AtomicFact,
        existing: AtomicFact,
        profile_id: str,
        *,
        reason: str,
    ) -> ConsolidationAction:
        """Near-duplicate — just bump access count on existing."""
        self._db.update_fact(
            existing.fact_id,
            {"access_count": existing.access_count + 1},
        )
        action = self._log_action(
            ConsolidationActionType.NOOP,
            new_fact.fact_id, existing.fact_id,
            profile_id, reason,
        )
        logger.debug("NOOP for %s (dup of %s): %s", new_fact.fact_id, existing.fact_id, reason)
        return action

    # -- Helpers ------------------------------------------------------------

    def _create_semantic_edges(
        self, new_fact: AtomicFact, profile_id: str,
    ) -> None:
        """Link new fact to top-K most similar existing facts."""
        if new_fact.embedding is None:
            return
        scored = [(f.fact_id, sim) for f, sim in
                  self._similar(new_fact, profile_id, self._cfg.semantic_edge_top_k, set())]
        for target_id, weight in scored:
            self._db.store_edge(GraphEdge(
                profile_id=profile_id,
                source_id=new_fact.fact_id,
                target_id=target_id,
                edge_type=EdgeType.SEMANTIC,
                weight=weight,
            ))

    def _merge_facts(self, existing_text: str, new_text: str) -> str:
        """Use LLM to merge two fact statements into one richer fact."""
        assert self._llm is not None
        prompt = (
            "Merge these two statements about the same topic into one "
            "concise, accurate fact. Keep all unique information.\n\n"
            f"Existing: {existing_text}\n"
            f"New: {new_text}\n\n"
            "Merged statement:"
        )
        result = self._llm.generate(prompt, system="You are a precise editor.")
        return result.strip() if result.strip() else ""

    def _log_action(
        self,
        action_type: ConsolidationActionType,
        new_fact_id: str,
        existing_fact_id: str,
        profile_id: str,
        reason: str,
    ) -> ConsolidationAction:
        """Persist action to consolidation_log and return the model."""
        action = ConsolidationAction(
            profile_id=profile_id,
            action_type=action_type,
            new_fact_id=new_fact_id,
            existing_fact_id=existing_fact_id,
            reason=reason,
        )
        self._db.execute(
            "INSERT INTO consolidation_log "
            "(action_id, profile_id, action_type, new_fact_id, "
            " existing_fact_id, reason, timestamp) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                action.action_id, action.profile_id,
                action.action_type.value, action.new_fact_id,
                action.existing_fact_id, action.reason, action.timestamp,
            ),
        )
        return action


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, no side effects)
# ---------------------------------------------------------------------------

def _compute_similarity(
    emb_a: list[float] | None, emb_b: list[float] | None,
) -> float:
    """Cosine similarity between two embedding vectors.

    Returns 0.0 if either embedding is None or empty.
    """
    if not emb_a or not emb_b:
        return 0.0
    if len(emb_a) != len(emb_b):
        return 0.0

    dot = sum(a * b for a, b in zip(emb_a, emb_b))
    norm_a = math.sqrt(sum(a * a for a in emb_a))
    norm_b = math.sqrt(sum(b * b for b in emb_b))

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity of two string sets. Returns 0.0 if both empty."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0
