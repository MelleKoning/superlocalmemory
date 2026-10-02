# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Typed Memory Router.

Routes extracted facts to appropriate typed stores (ENGRAM pattern).
Typed separation gave +31 points on LoCoMo benchmark.

Mode A: all-MiniLM similarity against type templates.
Mode B/C: LLM classifies fact type.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import TYPE_CHECKING

from superlocalmemory.encoding.prospective_markers import looks_prospective
from superlocalmemory.storage.models import AtomicFact, FactType, Mode

if TYPE_CHECKING:
    from superlocalmemory.core.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type templates for Mode A (all-MiniLM classification)
# ---------------------------------------------------------------------------

_EPISODIC_TEMPLATES = [
    "someone did something at a place",
    "a person went somewhere or performed an action",
    "an event happened on a specific date",
    "someone traveled to a location",
    "a meeting or gathering occurred",
]

_SEMANTIC_TEMPLATES = [
    "a person is a specific profession or role",
    "something is a fact about the world",
    "a place is located somewhere",
    "an organization does something",
    "a general knowledge statement",
]

_OPINION_TEMPLATES = [
    "someone thinks or believes something",
    "a person likes or dislikes something",
    "someone prefers one thing over another",
    "a subjective judgment or evaluation",
    "an emotional reaction to something",
]

_TEMPORAL_TEMPLATES = [
    "something will happen on a future date",
    "an event is scheduled or planned",
    "a deadline or due date for something",
    "something started or will end at a time",
    "a recurring event or appointment",
]

# ---------------------------------------------------------------------------
# Keyword patterns for Mode A fallback
# ---------------------------------------------------------------------------

_OPINION_MARKERS = re.compile(
    r"\b(think|believe|feel|prefer|like|love|hate|dislike|enjoy|"
    r"opinion|seems?|probably|maybe|might|could be|i guess|"
    r"personally|in my view|i'd say)\b",
    re.IGNORECASE,
)


_EPISODIC_MARKERS = re.compile(
    r"\b(went|visited|traveled|attended|met|saw|did|"
    r"happened|occurred|took place|experienced)\b",
    re.IGNORECASE,
)


class TypeRouter:
    """Route facts to typed stores based on content classification.

    Uses embedding similarity (Mode A) or LLM (Mode B/C) to classify
    each fact as episodic, semantic, opinion, or prospective.
    """

    def __init__(
        self,
        mode: Mode = Mode.A,
        embedder: EmbeddingService | None = None,
        llm: object | None = None,
    ) -> None:
        self._mode = mode
        self._embedder = embedder
        self._llm = llm
        self._template_embeddings: dict[FactType, list[list[float]]] | None = None

    def classify(self, fact: AtomicFact) -> FactType:
        """Classify a fact into a typed store category."""
        if self._mode in (Mode.B, Mode.C) and self._llm is not None:
            return self._classify_llm(fact)
        if self._embedder is not None:
            return self._classify_embedding(fact)
        return self._classify_keywords(fact)

    def route_facts(self, facts: list[AtomicFact]) -> list[AtomicFact]:
        """Classify and update fact_type for a batch of facts."""
        result = []
        for fact in facts:
            classified_type = self.classify(fact)
            # A new fact with only the type changed. Rebuilding it field by
            # field silently reset every field the list forgot — sharing,
            # pinning, lifecycle — whenever the dataclass grew.
            updated = dataclasses.replace(fact, fact_type=classified_type)
            result.append(updated)
        return result

    # -- Classification strategies -----------------------------------------

    def _classify_keywords(self, fact: AtomicFact) -> FactType:
        """Keyword-based classification (fastest, lowest quality)."""
        text = fact.content

        if _OPINION_MARKERS.search(text):
            return FactType.OPINION
        if looks_prospective(text):
            return FactType.PROSPECTIVE
        if _EPISODIC_MARKERS.search(text):
            return FactType.EPISODIC
        return FactType.SEMANTIC

    def _classify_embedding(self, fact: AtomicFact) -> FactType:
        """Embedding similarity against type templates (Mode A)."""
        if self._embedder is None:
            return self._classify_keywords(fact)

        if self._template_embeddings is None:
            self._build_template_embeddings()

        assert self._template_embeddings is not None
        fact_emb = self._embedder.embed(fact.content)
        if fact_emb is None:
            return self._classify_keywords(fact)

        best_type = FactType.SEMANTIC
        best_score = -1.0

        for ftype, template_embs in self._template_embeddings.items():
            avg_sim = sum(
                _cosine(fact_emb, t) for t in template_embs
            ) / max(len(template_embs), 1)
            if avg_sim > best_score:
                best_score = avg_sim
                best_type = ftype

        return best_type

    def _classify_llm(self, fact: AtomicFact) -> FactType:
        """LLM-based classification (Mode B/C, highest quality)."""
        if self._llm is None:
            return self._classify_embedding(fact)

        prompt = (
            f"Classify this fact into exactly one category.\n"
            f"Fact: \"{fact.content}\"\n"
            f"Categories:\n"
            f"- episodic: An event that happened (who did what when)\n"
            f"- semantic: A general fact about the world (X is Y)\n"
            f"- opinion: A subjective belief or preference\n"
            f"- prospective: A scheduled/planned future event\n"
            f"Reply with ONLY the category name (one word)."
        )
        try:
            response = self._llm.generate(prompt).strip().lower()
            type_map = {
                "episodic": FactType.EPISODIC,
                "semantic": FactType.SEMANTIC,
                "opinion": FactType.OPINION,
                "prospective": FactType.PROSPECTIVE,
                # The prompt asked for "temporal" until 4.1.0. Both words are
                # accepted because a model does not always echo the current
                # prompt, and an unrecognised word falls through to SEMANTIC —
                # a silently wrong type rather than a visible failure.
                "temporal": FactType.PROSPECTIVE,
            }
            return type_map.get(response, FactType.SEMANTIC)
        except Exception:
            logger.warning("LLM classification failed, falling back to keywords")
            return self._classify_keywords(fact)

    def _build_template_embeddings(self) -> None:
        """Pre-compute embeddings for type templates."""
        if self._embedder is None:
            return
        self._template_embeddings = {
            FactType.EPISODIC: self._embedder.embed_batch(_EPISODIC_TEMPLATES),
            FactType.SEMANTIC: self._embedder.embed_batch(_SEMANTIC_TEMPLATES),
            FactType.OPINION: self._embedder.embed_batch(_OPINION_TEMPLATES),
            FactType.PROSPECTIVE: self._embedder.embed_batch(_TEMPORAL_TEMPLATES),
        }


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
