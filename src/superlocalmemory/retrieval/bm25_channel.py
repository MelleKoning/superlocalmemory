# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — BM25 Keyword Search Channel.

Persistent BM25Plus index over fact content. Catches exact name/date
matches that embedding similarity misses.

V1 bug fix: V1 kept BM25 tokens in-memory only — a restart lost
the entire index. This version persists tokens to the DB via
store_bm25_tokens / get_all_bm25_tokens and cold-loads on init.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from rank_bm25 import BM25Plus

from superlocalmemory.storage.database import _scope_where
from superlocalmemory.storage.fts_terms import version_match_phrases, version_terms

if TYPE_CHECKING:
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)

# Minimal stopwords — small set to avoid stripping important terms
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "as", "into", "through",
    "and", "but", "or", "nor", "not", "so", "yet", "if", "then", "than",
    "that", "this", "it", "its", "i", "me", "my", "we", "our", "you",
    "your", "he", "him", "his", "she", "her", "they", "them", "their",
})

# Token pattern: words with letters/digits, keeps hyphens and apostrophes
_TOKEN_RE = re.compile(r"[a-zA-Z0-9][\w'-]*[a-zA-Z0-9]|[a-zA-Z0-9]")


def tokenize(text: str) -> list[str]:
    """Tokenize text: lowercase, split, remove stopwords.

    Exported so encoding pipeline can persist tokens at ingest time.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _with_version_terms(tokens: list[str], text: str) -> list[str]:
    """``tokens`` plus each whole dotted version in ``text``, for the in-memory index.

    Only the in-memory corpus and query get these. The persisted
    ``bm25_tokens`` rows stay exactly ``tokenize(content)``, so no stored row
    changes meaning and nothing on disk needs rewriting.
    """
    return [*tokens, *version_terms(text)]


#: Saturation constant for the BM25 -> [0,1) transform. Chosen from measurement,
#: not taste: raw scores on a live store run ~2.5 to ~15 across query
#: lengths, so k = 5 puts an ordinary match near 0.4 and a strong one near 0.7,
#: leaving headroom at both ends rather than pinning everything to one corner.
_BM25_SATURATION = 5.0


def _to_unit_scale(scored: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Map BM25 scores into [0, 1) without disturbing order or magnitude.

    WHY ANY OF THIS. ``engine.apply_channel_weights`` re-scores a candidate as
    ``sum(channel_scores[ch] * weights[ch])`` — a SUM of raw channel scores.
    Every other channel is bounded: semantic cosine and Fisher-Rao are [0, 1],
    the temporal proximity score is Gaussian on [0, 1]. BM25 is not. Measured on
    a live store, one query at a time::

        "slm release"                                    max  2.845
        "memory"                                         max  3.865
        "the release ships once both audits are clean"    max 10.150

    So the sum was decided by a scale rather than by evidence, and any weight
    the bandit converged on was a correction for that scale — wrong the moment
    the query length changed.

    WHY NOT DIVIDE BY THE BATCH MAXIMUM. That was the first implementation here
    and it is wrong, which an existing test caught before it shipped:
    ``test_real_fts5_exact_hit_keeps_bounded_slot`` exists to prove *"a real
    sub-1.0 FTS5 hit remains visible under semantic pressure"*. Dividing by the
    batch maximum makes the best result exactly 1.0 **whatever it scored**, so a
    query with one weak lexical match reports full confidence and outranks a
    semantic channel it should lose to. Batch-relative scaling manufactures
    confidence out of an empty batch, and it also makes a fact's score depend on
    which other facts happened to come back — the same query returning a
    different number on a different day, which HARD-RULES RULE 6 puts above
    speed.

    A saturating transform has neither problem: ``s / (s + k)`` is strictly
    increasing, so order is untouched; it is bounded below 1.0, so nothing is
    ever certain; and it depends only on the score itself, so it is repeatable.
    Applied to the measurements above: 2.676 -> 0.35, 2.845 -> 0.36,
    3.865 -> 0.44, 10.150 -> 0.67.

    FUSION IS UNAFFECTED, verified by reading ``fusion.weighted_rrf`` rather
    than assuming: it computes ``fused += w / (k + rank)`` and keeps the score
    only for reporting. Rescaling a value nothing divides by cannot move a
    fused rank.

    Non-positive scores map to 0.0. FTS5's ``bm25()`` is <= 0 and is negated
    here, so a value at or below zero means "no lexical evidence", and that is
    what it should contribute to a sum.
    """
    if not scored:
        return scored
    return [
        (fid, (s / (s + _BM25_SATURATION)) if s > 0.0 else 0.0)
        for fid, s in scored
    ]


class BM25Channel:
    """Persistent BM25Plus index for keyword retrieval.

    On cold start, loads all tokens from the DB. After that, new facts
    are added incrementally. The BM25Plus model is rebuilt lazily
    before each search when the corpus has changed.

    Attributes:
        document_count: Number of indexed documents.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self._corpus: list[list[str]] = []
        self._fact_ids: list[str] = []
        self._fact_id_set: set[str] = set()
        self._raw_texts: list[str] = []  # V3.3.12: raw content for phrase matching
        self._bm25: BM25Plus | None = None
        self._dirty: bool = False
        self._loaded_profiles: set[str] = set()
        self._loaded_scope_key: tuple[str, bool, bool] | None = None

    @property
    def document_count(self) -> int:
        return len(self._corpus)

    def ensure_loaded(
        self,
        profile_id: str,
        include_global: bool | None = None,
        include_shared: bool | None = None,
    ) -> None:
        """Cold-load BM25 tokens from DB for a profile (once).

        Idempotent: subsequent calls for the same profile/scope are no-ops.

        Args:
            profile_id: Profile to load.
            include_global: Include global-scope facts. Falls back to the
                instance attribute when not supplied.
            include_shared: Include shared-scope facts. Same fallback.
        """
        if include_global is None:
            include_global = bool(getattr(self, "include_global", False))
        if include_shared is None:
            include_shared = bool(getattr(self, "include_shared", False))
        scope_key = (profile_id, include_global, include_shared)
        if scope_key == self._loaded_scope_key:
            return

        # A legacy in-memory index is a visibility-specific view. Reusing a
        # corpus warmed under another profile/scope leaks private documents or
        # omits newly enabled global/shared documents.
        self._corpus = []
        self._fact_ids = []
        self._fact_id_set = set()
        self._raw_texts = []
        self._bm25 = None
        token_map = self._db.get_all_bm25_tokens(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        if not token_map:
            # Fallback: tokenize facts directly if no pre-stored tokens
            facts = self._db.get_all_facts(
                profile_id,
                include_global=include_global,
                include_shared=include_shared,
            )
            for fact in facts:
                if fact.fact_id in self._fact_id_set:
                    continue
                tokens = tokenize(fact.content)
                if tokens:
                    self._corpus.append(_with_version_terms(tokens, fact.content))
                    self._fact_ids.append(fact.fact_id)
                    self._fact_id_set.add(fact.fact_id)
                    self._raw_texts.append(fact.content)
                    # Persist for next cold start
                    self._db.store_bm25_tokens(fact.fact_id, profile_id, tokens)
        else:
            # Load raw texts for phrase matching (V3.3.12)
            fact_content_map = {}
            try:
                facts = self._db.get_all_facts(
                    profile_id,
                    include_global=include_global,
                    include_shared=include_shared,
                )
                fact_content_map = {f.fact_id: f.content for f in facts}
            except Exception:
                pass
            for fid, tokens in token_map.items():
                if fid in self._fact_id_set:
                    continue
                raw_text = fact_content_map.get(fid, "")
                self._corpus.append(_with_version_terms(tokens, raw_text))
                self._fact_ids.append(fid)
                self._fact_id_set.add(fid)
                self._raw_texts.append(raw_text)

        self._dirty = True
        self._loaded_profiles.add(profile_id)
        self._loaded_scope_key = scope_key
        logger.debug(
            "BM25 cold-loaded %d documents for profile=%s",
            len(token_map) if token_map else 0, profile_id,
        )

    def add(self, fact_id: str, content: str, profile_id: str) -> None:
        """Add a single fact to the index and persist tokens.

        Args:
            fact_id: Unique fact identifier.
            content: Raw text content to index.
            profile_id: Owner profile.
        """
        tokens = tokenize(content)
        if not tokens:
            return

        self._corpus.append(_with_version_terms(tokens, content))
        self._fact_ids.append(fact_id)
        self._fact_id_set.add(fact_id)
        if not hasattr(self, '_raw_texts'):
            self._raw_texts = []
        self._raw_texts.append(content)
        self._dirty = True

        # Persist for cold start
        self._db.store_bm25_tokens(fact_id, profile_id, tokens)

    def _fts5_search(
        self,
        query: str,
        profile_id: str,
        top_k: int = 30,
        include_global: bool | None = None,
        include_shared: bool | None = None,
    ) -> list[tuple[str, float]]:
        """v3.5.0: SQLite FTS5 keyword search (C-level indexed, scales to millions).

        Uses the ``atomic_facts_fts`` external-content FTS5 table (kept in sync
        by INSERT/DELETE/UPDATE triggers). Joins ``atomic_facts`` for profile
        scoping (FTS table has no profile_id). ``bm25()`` returns a negative
        score (lower = better match); we negate it to the channel's
        "higher = better" convention. Returns [] on no matches.

        Raises (OperationalError) if the FTS5 table is absent — the caller
        then falls back to the legacy in-memory rank_bm25 path.
        """
        if include_global is None:
            include_global = bool(getattr(self, "include_global", False))
        if include_shared is None:
            include_shared = bool(getattr(self, "include_shared", False))
        tokens = tokenize(query)
        versions = version_terms(query)
        phrases = version_match_phrases(query)
        if not tokens and not phrases:
            return []
        # Quote each token so query punctuation can't break FTS5 MATCH syntax;
        # OR-join for high recall (any token may match). A dotted version is
        # also offered whole, as a phrase (see storage.fts_terms): its single
        # numbers are too common to tell one release from another.
        match_expr = " OR ".join([
            *('"' + t.replace('"', "") + '"' for t in tokens), *phrases,
        ])
        # Rows that only matched a version phrase inside a longer number lose
        # that share below and can fall out of the top_k, so fetch room for
        # the rows that move up into their place.
        fetch_k = int(top_k) * 2 if versions else int(top_k)
        where, params = _scope_where(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
            prefix="af",
        )
        # Soft-deleted and withheld rows are not live. Filtered HERE rather
        # than only at hydration so neither spends one of this channel's top_k
        # slots; the shared clause keeps the definition in one place.
        archive_clause = self._db.visible_fact_clause("af")
        sql = (
            "SELECT af.fact_id AS fact_id, bm25(atomic_facts_fts) AS rank, "
            "af.rowid AS fts_rowid, af.content AS content "
            "FROM atomic_facts_fts "
            "JOIN atomic_facts af ON af.rowid = atomic_facts_fts.rowid "
            f"WHERE atomic_facts_fts MATCH ? AND {where}{archive_clause} "
            "ORDER BY rank LIMIT ?"
        )
        rows = [dict(r) for r in self._db.execute(sql, (match_expr, *params, fetch_k))]
        out: list[tuple[str, float]] = []
        for d in rows:
            fid = d.get("fact_id")
            if not fid:
                continue
            out.append((fid, -float(d.get("rank", 0.0))))
        if versions:
            out = self._without_partial_version_credit(out, rows, versions, phrases)

        # T3b: UNION fact-expansion (alias / paraphrase) matches so a query for
        # a synonym matches a fact that only used the canonical term. Additive —
        # direct content hits stay primary; an alias-only hit is added at a
        # discount so it ranks below content matches. Fail-open if the expansion
        # FTS is absent (legacy DB).
        content_ids = {fid for fid, _ in out}
        try:
            exp_sql = (
                "SELECT af.fact_id AS fact_id, bm25(fact_expansion_fts) AS rank "
                "FROM fact_expansion_fts "
                "JOIN atomic_facts af ON af.fact_id = fact_expansion_fts.fact_id "
                f"WHERE fact_expansion_fts MATCH ? AND {where}{archive_clause} "
                "ORDER BY rank LIMIT ?"
            )
            for r in self._db.execute(exp_sql, (match_expr, *params, int(top_k))):
                d = dict(r)
                fid = d.get("fact_id")
                if fid and fid not in content_ids:
                    out.append((fid, -float(d.get("rank", 0.0)) * 0.85))
        except Exception as exc:  # pragma: no cover — legacy/missing expansion FTS
            logger.debug("Expansion FTS search skipped: %s", exc)

        out.sort(key=lambda x: (-x[1], x[0]))
        return out[:top_k]

    def _without_partial_version_credit(
        self,
        scored: list[tuple[str, float]],
        rows: list[dict],
        versions: tuple[str, ...],
        phrases: tuple[str, ...],
    ) -> list[tuple[str, float]]:
        """Take the version phrase's BM25 share back from rows that are not that version.

        The phrase ``"4.1.20"`` also matches inside ``4.1.20.1``, ``3.4.1.20``
        or a date written ``4/1/20``. Those rows should keep what their single
        numbers earned, a partial match, and lose only the phrase's share.

        FTS5's ``bm25()`` is a sum over the query's phrases, and each phrase's
        IDF is counted over the whole table whatever else the statement
        filters on. So scoring the version phrases alone, for just these rows,
        yields exactly the share the combined query gave them, and subtracting
        it leaves exactly the score the rest of the query earned. A row that
        does not match a phrase is not returned and keeps its score.
        """
        wanted = set(versions)
        partial_rowids = [
            int(d["fts_rowid"]) for d in rows
            if d.get("fact_id") and d.get("fts_rowid") is not None
            and not wanted.intersection(version_terms(d.get("content") or ""))
        ]
        if not partial_rowids:
            return scored
        placeholders = ",".join("?" * len(partial_rowids))
        share_sql = (
            "SELECT af.fact_id AS fact_id, bm25(atomic_facts_fts) AS rank "
            "FROM atomic_facts_fts "
            "JOIN atomic_facts af ON af.rowid = atomic_facts_fts.rowid "
            "WHERE atomic_facts_fts MATCH ? "
            f"AND atomic_facts_fts.rowid IN ({placeholders})"
        )
        share = {
            dict(r)["fact_id"]: -float(dict(r).get("rank", 0.0))
            for r in self._db.execute(
                share_sql, (" OR ".join(phrases), *partial_rowids),
            )
        }
        return [(fid, max(0.0, s - share.get(fid, 0.0))) for fid, s in scored]

    def search(
        self,
        query: str,
        profile_id: str,
        top_k: int = 30,
        include_global: bool | None = None,
        include_shared: bool | None = None,
    ) -> list[tuple[str, float]]:
        """Search BM25 index for matching facts.

        Auto-loads from DB on first call for this profile.

        Args:
            query: Search query text.
            profile_id: Scope to this profile.
            top_k: Maximum results.
            include_global: Include global-scope facts. Falls back to the
                instance attribute when not supplied.
            include_shared: Include shared-scope facts. Same fallback.

        Returns:
            List of (fact_id, bm25_score) sorted by score descending.
        """
        if include_global is None:
            include_global = bool(getattr(self, "include_global", False))
        if include_shared is None:
            include_shared = bool(getattr(self, "include_shared", False))
        # v3.5.0: FTS5 fast path — C-level indexed, ~ms, scales to millions.
        # The legacy in-memory rank_bm25 path rebuilt the whole index over the
        # entire corpus on every corpus change (11s+ at 17.5k facts, does not
        # scale). FTS5 (atomic_facts_fts, kept in sync by triggers) replaces it.
        # Falls back to rank_bm25 ONLY if the FTS5 table is genuinely
        # unavailable (raises) — e.g. a pre-FTS legacy DB.
        try:
            return _to_unit_scale(self._fts5_search(
                query, profile_id, top_k,
                include_global=include_global, include_shared=include_shared,
            ))
        except Exception as exc:  # pragma: no cover — legacy/missing FTS table
            logger.debug(
                "BM25 FTS5 path unavailable, using rank_bm25 fallback: %s", exc,
            )

        self.ensure_loaded(profile_id, include_global=include_global, include_shared=include_shared)

        if not self._corpus:
            return []

        query_tokens = _with_version_terms(tokenize(query), query)
        if not query_tokens:
            return []

        # Rebuild BM25 model if corpus changed
        if self._dirty or self._bm25 is None:
            self._bm25 = BM25Plus(self._corpus, k1=1.2, b=0.75)
            self._dirty = False

        scores = self._bm25.get_scores(query_tokens)

        scored: list[tuple[str, float]] = []
        # V3.3.12: Exact phrase bonus — boost facts containing the full query phrase
        query_lower = query.lower().strip()
        for i, score in enumerate(scores):
            if score > 0.0:
                bonus = score
                # Exact phrase match bonus: if the query appears as a substring in the document
                if len(query_lower) >= 5 and i < len(self._raw_texts):
                    if query_lower in self._raw_texts[i].lower():
                        bonus *= 1.5  # 50% boost for exact phrase match
                scored.append((self._fact_ids[i], bonus))

        scored.sort(key=lambda x: (-x[1], x[0]))
        # Same rescale as the FTS5 path. This fallback applies a 1.5x exact
        # phrase bonus, so its raw ceiling is higher still.
        return _to_unit_scale(scored[:top_k])

    def update_fact(self, fact_id: str, new_content: str, profile_id: str) -> None:
        """Replace a fact's representation in the live index and persist new tokens.

        Removes any existing in-memory entry for the fact_id first so the index
        holds each fact exactly once after the call. Delegates to remove_fact +
        add so the two operations stay in sync.

        Args:
            fact_id: Fact whose content has changed.
            new_content: Updated text to index.
            profile_id: Owner profile (used for DB token persistence).
        """
        self.remove_fact(fact_id)
        self.add(fact_id, new_content, profile_id)

    def remove_fact(self, fact_id: str) -> None:
        """Remove a fact from the live in-memory index.

        Idempotent: a no-op when the fact is not in the index. Does NOT touch
        the persistent ``bm25_tokens`` DB table — the caller is responsible for
        that (so a delete path can clean up storage independently).

        Args:
            fact_id: Fact to evict from the in-memory corpus.
        """
        if fact_id not in self._fact_id_set:
            return
        try:
            idx = self._fact_ids.index(fact_id)
            del self._corpus[idx]
            del self._fact_ids[idx]
            raw_texts = getattr(self, "_raw_texts", [])
            if idx < len(raw_texts):
                del raw_texts[idx]
            self._fact_id_set.discard(fact_id)
            self._dirty = True
        except (ValueError, IndexError):
            self._fact_id_set.discard(fact_id)

    def clear(self) -> None:
        """Clear the in-memory index (does NOT delete DB tokens)."""
        self._corpus = []
        self._fact_ids = []
        self._fact_id_set = set()
        self._bm25 = None
        self._dirty = False
        self._loaded_profiles = set()
        self._loaded_scope_key = None
