# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""SYNAPSE spreading activation -- 5th retrieval channel.

SYNAPSE (arXiv 2601.02744) 5-step algorithm adapted for SLM.
Pure math -- no LLM calls at query time. With M=7, T=3 the
computation is ~21 neighbor lookups (<5ms on SQLite with indexes).

Reads BOTH graph_edges + association_edges via UNION query (Rule 13).
Registered as 5th channel via ChannelRegistry (needs_embedding=True).

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from superlocalmemory.retrieval.scope_policy import (
    authorized_fact_ids,
    filter_authorized_results,
)
from superlocalmemory.storage.database import _scope_where
from superlocalmemory.storage.graph_generation import graph_version
from superlocalmemory.storage.models import _new_id

logger = logging.getLogger(__name__)


def _seed_score(cosine: float, vector_path: bool = True) -> float:
    """A seed's initial activation from its cosine to the query.

    The ONE place the seed scale is defined, so local and cross-scope seeds
    can never drift apart again (#146), and neither can the two seed paths.
    The vec0 path scores ``max(0, 1 - distance)`` = ``max(0, cos)``
    (``VectorStore.search``), and every path now uses that scale.

    Before 4.1.20 the SQL fallback used ``(cos + 1) / 2``: cos 0.45 started the
    walk at 0.725 instead of 0.45, an unrelated memory (cos 0) at 0.5, so the
    answer depended on whether the vector index happened to be available.
    ``vector_path`` is accepted for callers and no longer changes the scale.
    """
    del vector_path
    return max(0.0, cosine)


def _evidence_seeds(seeds: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Only seeds the question actually resembles start the walk.

    A seed scored 0 (cosine <= 0) carries no evidence, yet one sigmoid round
    lifts it to ``sigmoid(-theta)`` ~ 0.45 and it then spreads like a real
    match. Which zero-scored facts fill the ``top_m`` seed slots is an
    arbitrary tie-break, and the vec0 index and the SQL fallback break that
    tie differently, so a question that resembled nothing came back with a
    different spreading-activation answer on a machine without sqlite-vec.
    Dropping them makes that answer "nothing" on every backend.
    """
    return [(fact_id, score) for fact_id, score in seeds if score > 0.0]


# ---------------------------------------------------------------------------
# Configuration (frozen dataclass, Rule 10)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpreadingActivationConfig:
    """Configuration for SYNAPSE spreading activation.

    All hyperparameters from the SYNAPSE paper (arXiv 2601.02744).
    SYNAPSE tuned on 384d (all-MiniLM-L6-v2). SLM uses 768d
    (nomic-embed-text). Phase 3 calibration test verifies convergence.
    """

    alpha: float = 1.0           # Seed scaling factor
    delta: float = 0.5           # Node retention / self-decay per iteration
    spreading_factor: float = 0.8  # S: energy diffusion rate
    # V3.3.20: Recalibrated for SLM graph density (254K edges, 768d).
    # V3.4.40 (2026-05-09): graph grew to 960K edges. top_m=20 caused 5.5s recalls.
    # Reduced to 10 (compromise between SYNAPSE default 7 and the dense-graph 20).
    # SYNAPSE defaults (theta=0.5, top_m=7) were for 384d sparse graphs.
    theta: float = 0.2           # Activation threshold for sigmoid (was 0.5)
    top_m: int = 10              # Lateral inhibition: max active nodes (was 20, then 7 originally)
    max_iterations: int = 3      # T: propagation depth
    tau_gate: float = 0.05       # FOK confidence gate (was 0.12)
    enabled: bool = True         # Ships enabled by default
    # V3.4.40 (2026-05-09): per-node neighbor fan-out clamp.
    # Hub nodes in dense graphs (5K+ edges) caused unbounded work per expansion.
    # v3.4.52: reduced from 100 to 30 — GAM (ICLR 2026) shows that with
    # covering indexes on weight DESC, 30 well-ranked neighbors provides
    # sufficient spreading signal. Combined with streaming merge (SQLite 3.45+),
    # this brings SpreadingActivation from 4.2s to ~60ms.
    max_neighbors_per_node: int = 30
    # v3.4.1: Graph intelligence integration
    use_pagerank_bias: bool = False    # Multiply propagation by target PageRank
    community_boost: float = 0.0       # Boost same-community nodes (0.0 = disabled)


# ---------------------------------------------------------------------------
# SpreadingActivation Channel
# ---------------------------------------------------------------------------

#: Bumped when cross-scope seed selection changes, so activations cached by the
#: old selection are not served after an upgrade.
_CROSS_SCOPE_SEEDING = "|seeds=2"

#: Seed scores are compared to this many decimals in the cache key.
_SEED_SCORE_DECIMALS = 6


def _seed_fingerprint(seeds: list[tuple[str, float]]) -> str:
    """Order-free digest of a seed set: ids and rounded scores."""
    material = "\n".join(
        f"{fact_id}={round(float(score), _SEED_SCORE_DECIMALS)!r}"
        for fact_id, score in sorted(
            seeds, key=lambda item: (str(item[0]), float(item[1])))
    )
    return "|seedset=" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


class SpreadingActivation:
    """SYNAPSE 5-step spreading activation as 5th retrieval channel.

    Algorithm:
        Step 1: Initialization with ALPHA seed scaling
        Step 2: Propagation with fan effect (out-degree normalization)
        Step 3: Lateral inhibition (top-M=7 pruning)
        Step 4: Nonlinear sigmoid gating
        Step 5: Iterate T=3 times, then FOK gate

    Registered as 5th channel via ChannelRegistry (Rule 07).
    Reads BOTH graph_edges + association_edges via UNION query (Rule 13).
    """

    def __init__(
        self,
        db: Any,
        vector_store: Any | None,
        config: SpreadingActivationConfig | None = None,
    ) -> None:
        self._db = db
        self._vector_store = vector_store
        self._config = config or SpreadingActivationConfig()
        # v3.4.1: Graph intelligence caches (loaded lazily per profile)
        self._pr_cache: dict[str, float] = {}
        self._pr_profile: str = ""
        self._comm_cache: dict[str, int | None] = {}
        self._comm_profile: str = ""
        self._graph_version_warned = False

    def search(
        self,
        query: Any,
        profile_id: str = "",
        top_k: int = 7,
        include_global: bool | None = None,
        include_shared: bool | None = None,
    ) -> list[tuple[str, float]]:
        """Channel-compatible interface: (query, top_k) -> [(fact_id, score)].

        Matches ANNSearchable protocol (Rule 07).

        Args:
            include_global: Include global-scope facts. Falls back to the
                instance attribute when not supplied.
            include_shared: Include shared-scope facts. Same fallback.
        """
        if not self._config.enabled:
            return []

        if include_global is None:
            include_global = bool(getattr(self, "include_global", False))
        if include_shared is None:
            include_shared = bool(getattr(self, "include_shared", False))
        try:
            # Step 0: Get seed nodes from VectorStore KNN
            seed_results = self._seed_search(
                query,
                profile_id,
                include_global=include_global,
                include_shared=include_shared,
            )
            if include_global or include_shared:
                seed_results = self._merge_cross_scope_seeds(
                    query,
                    seed_results,
                    profile_id,
                    include_global=include_global,
                    include_shared=include_shared,
                )
            else:
                # v3.8.2 perf: seeds come from this profile's own vector index /
                # get_all_facts(profile_id), so for personal scope they are
                # already authorized, and there are top_m of them at most.
                # filter_authorized_results below remains the security net.
                seed_results = list(
                    {fact_id: score for fact_id, score in seed_results}.items()
                )
            seed_results = _evidence_seeds(seed_results)
            if not seed_results:
                return []

            # Check cache first
            # Keyed by the seeds too: a memory saved since the entry was
            # written that now seeds this question must be walked, not hidden
            # behind an hour-old answer. And by the graph's generation, read
            # BEFORE the walk: a link written during or after the walk moves
            # the generation, so the entry stored below is never read again.
            graph = self._graph_version(
                profile_id, cross_scope=include_global or include_shared,
            )
            query_hash = None if graph is None else self._compute_query_hash(
                query,
                profile_id,
                include_global=include_global,
                include_shared=include_shared,
                seeds=seed_results,
                graph=graph,
            )
            cached = (
                None if query_hash is None
                else self._get_cached_results(query_hash, profile_id)
            )
            if cached:
                # A hit is answered exactly as the propagation that stored it
                # was: same order, same fail-closed filter, for EVERY scope.
                # The cache holds raw activations, and a personal propagation
                # can reach facts outside this profile through an edge; 4.1.19
                # returned personal hits unfiltered, so a repeated question
                # could surface another profile's fact the first answer had
                # removed (and the two answers differed).
                return filter_authorized_results(
                    self._db,
                    cached,
                    profile_id,
                    include_global=include_global,
                    include_shared=include_shared,
                )[:top_k]

            # Run 5-step spreading activation
            activations = self._propagate(
                seed_results,
                profile_id,
                include_global=include_global,
                include_shared=include_shared,
            )

            # FOK gating
            if not self._fok_check(activations):
                return []

            # Cache results — DEFERRED so recall stays READ-ONLY on its hot
            # path.  This is a perf cache for FUTURE recalls, not needed to
            # return the current results.
            if query_hash is not None:
                from superlocalmemory.storage.deferred_writes import (
                    submit_background,
                )
                submit_background(
                    lambda: self._cache_results(
                        query_hash, profile_id, activations)
                )

            # Return top-K sorted by activation
            results = sorted(
                activations.items(), key=lambda x: (-x[1], x[0]),
            )
            return filter_authorized_results(
                self._db,
                results,
                profile_id,
                include_global=include_global,
                include_shared=include_shared,
            )[:top_k]

        except Exception as exc:
            logger.warning(
                "SpreadingActivation.search failed for profile %s: %s",
                profile_id, exc,
            )
            return []

    def _graph_version(self, profile_id: str, *, cross_scope: bool) -> str | None:
        """The graph generation for the cache key, or None to skip the cache.

        None when the store cannot say it (no ``graph_generation`` table): an
        entry could then be older than a link, so the cache is neither read nor
        written and every recall walks the graph. Logged once per channel.
        """
        try:
            return graph_version(self._db, profile_id, cross_scope=cross_scope)
        except Exception as exc:
            if not self._graph_version_warned:
                self._graph_version_warned = True
                logger.warning(
                    "Spreading-activation cache off: the graph generation "
                    "is unreadable (%s); every recall walks the graph.", exc,
                )
            return None

    def _uses_vector_seeds(self) -> bool:
        """Whether ``_seed_search`` takes the vec0 path (else the SQL scan)."""
        return self._vector_store is not None and bool(
            getattr(self._vector_store, "available", False)
        )

    def _merge_cross_scope_seeds(
        self,
        query: Any,
        seed_results: list[tuple[str, float]],
        profile_id: str,
        *,
        include_global: bool,
        include_shared: bool,
    ) -> list[tuple[str, float]]:
        """Seeds for a cross-scope read: the top_m of local + visible external.

        The vector index is partitioned by owner, so it cannot find an opted-in
        global or shared fact; those are scored here and merged in.  #146:

        * Externals are scored on the SAME scale as the seed path.  vec0 seeds
          are ``max(0, cos)``; externals used ``(cos + 1) / 2``, so an unrelated
          external (cos 0 -> 0.5) outranked a relevant local fact (cos 0.45).
        * Authorization runs on the whole merged set FIRST, so a fact the
          caller may not see can never take a seed slot from one it may.
        * Then the set is cut to ``top_m`` by ``(-score, fact_id)``.  That is
          the algorithm's own seed width -- the set a single KNN over every
          visible fact would return -- not a new cap.  Without it all visible
          externals became seeds, one neighbour lookup each, and the channel
          ran past its 8 s guard and answered nothing.
        """
        external_embeddings: list = []
        try:
            # #147: id + embedding only; full hydration was most of the cost.
            external_embeddings = self._db.get_external_visible_embeddings(
                profile_id,
                include_global=include_global,
                include_shared=include_shared,
            )
        except Exception as exc:
            # Not silent: losing this drops every cross-scope seed.
            logger.warning(
                "SpreadingActivation: cross-scope seeds unavailable "
                "for profile %s: %s", profile_id, exc,
            )
        vector_path = self._uses_vector_seeds()
        q_vec = np.array(query, dtype=np.float32)
        q_norm = float(np.linalg.norm(q_vec))
        combined = {fact_id: score for fact_id, score in seed_results}
        for ext_id, fact_vec in external_embeddings:
            if fact_vec.shape != q_vec.shape:
                continue
            denominator = q_norm * float(np.linalg.norm(fact_vec))
            if denominator <= 1e-8:
                continue
            score = _seed_score(
                float(np.dot(q_vec, fact_vec) / denominator), vector_path,
            )
            combined[ext_id] = max(combined.get(ext_id, 0.0), score)
        allowed_seeds = authorized_fact_ids(
            self._db,
            combined,
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        authorized = [
            (fact_id, score)
            for fact_id, score in combined.items()
            if fact_id in allowed_seeds
        ]
        authorized.sort(key=lambda item: (-item[1], item[0]))
        return authorized[: self._config.top_m]

    def _seed_search(
        self,
        query: Any,
        profile_id: str,
        *,
        include_global: bool,
        include_shared: bool,
    ) -> list[tuple[str, float]]:
        """Return bounded semantic graph seeds from vec0 or canonical SQLite.

        sqlite-vec is an acceleration projection, not a prerequisite for the
        graph retrieval layer.  When it is unavailable on a platform, use the
        canonical stored embeddings and the same cosine signal as the
        cross-profile supplement below.  This keeps spreading activation
        present and truthful rather than silently removing a retrieval layer.
        """
        if self._uses_vector_seeds():
            return self._vector_store.search(
                query, top_k=self._config.top_m, profile_id=profile_id,
            )

        q_vec = np.array(query, dtype=np.float32)
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm <= 1e-8:
            return []
        facts = self._db.get_all_facts(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        scored: list[tuple[str, float]] = []
        for fact in facts:
            embedding = getattr(fact, "embedding", None)
            if embedding is None:
                continue
            fact_vec = np.array(embedding, dtype=np.float32)
            if fact_vec.shape != q_vec.shape:
                continue
            denominator = q_norm * float(np.linalg.norm(fact_vec))
            if denominator <= 1e-8:
                continue
            score = _seed_score(
                float(np.dot(q_vec, fact_vec) / denominator), vector_path=False,
            )
            scored.append((fact.fact_id, score))
        return sorted(scored, key=lambda item: (-item[1], item[0]))[:self._config.top_m]

    def _propagate(
        self,
        seeds: list[tuple[str, float]],
        profile_id: str,
        *,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> dict[str, float]:
        """Execute the 5-step SYNAPSE algorithm.

        Step 1: a_i^(0) = alpha * sim(h_i, h_q) for seeds, 0 otherwise
        Step 2: u_i^(t+1) = delta * a_i^(t) + S * SUM(w_ji/deg(j) * a_j^(t))
        Step 3: Lateral inhibition -- keep top-M=7 only
        Step 4: sigmoid(u - theta)
        Step 5: Iterate T=3 times
        """
        cfg = self._config

        # Step 1: Initialization
        activations: dict[str, float] = {}
        for fact_id, similarity in seeds:
            activations[fact_id] = cfg.alpha * similarity

        # Cache neighbor lookups and out-degrees across iterations — same node
        # often survives multiple rounds via self-retention (delta=0.5);
        # caching here cuts ~80% of SQL queries vs per-iteration re-query.
        degree_cache: dict[str, int] = {}
        neighbor_cache: dict[str, list] = {}

        # Steps 2-4, repeated T times
        for _iteration in range(cfg.max_iterations):
            new_activations: dict[str, float] = {}

            for node_id, activation in activations.items():
                if activation < 0.001:
                    continue

                # Get neighbors from BOTH tables (Rule 13) — cached per node.
                # v3.8.2 perf-fix: _get_unified_neighbors already filters edges by
                # the scope predicate (graph_edges via _scope_where; association_edges
                # by profile_id), so for the default personal scope the returned
                # neighbors are inherently authorized. The per-node re-authorization
                # (2 DB round-trips/node, ~30–60 per recall — the primary 3.8 latency
                # regression) is only required when a cross-scope read can surface
                # global/shared neighbors. filter_authorized_results() on the returned
                # set remains the security net for every scope.
                if node_id not in neighbor_cache:
                    raw_neighbors = self._get_unified_neighbors(
                        node_id,
                        profile_id,
                        include_global=include_global,
                        include_shared=include_shared,
                    )
                    if include_global or include_shared:
                        allowed_neighbors = authorized_fact_ids(
                            self._db,
                            (neighbor_id for neighbor_id, _weight in raw_neighbors),
                            profile_id,
                            include_global=include_global,
                            include_shared=include_shared,
                        )
                        neighbor_cache[node_id] = [
                            item for item in raw_neighbors
                            if item[0] in allowed_neighbors
                        ]
                    else:
                        neighbor_cache[node_id] = raw_neighbors
                neighbors = neighbor_cache[node_id]

                # Out-degree for fan effect normalization
                if node_id not in degree_cache:
                    degree_cache[node_id] = max(len(neighbors), 1)
                out_degree = degree_cache[node_id]

                # Step 2: Propagation with fan effect
                for neighbor_id, edge_weight in neighbors:
                    spread = (
                        cfg.spreading_factor
                        * (edge_weight / out_degree)
                        * activation
                    )
                    new_activations[neighbor_id] = (
                        new_activations.get(neighbor_id, 0.0) + spread
                    )

            # Add self-retention (delta * current activation)
            for node_id, activation in activations.items():
                new_activations[node_id] = (
                    new_activations.get(node_id, 0.0) + cfg.delta * activation
                )

            # Step 3: Lateral inhibition -- keep only top-M
            sorted_nodes = sorted(
                new_activations.items(), key=lambda x: (-x[1], x[0]),
            )
            top_m_nodes = sorted_nodes[: cfg.top_m]

            # Step 4: Nonlinear activation (sigmoid with threshold shift)
            activations = {}
            for node_id, raw_activation in top_m_nodes:
                gated = 1.0 / (1.0 + math.exp(-(raw_activation - cfg.theta)))
                activations[node_id] = gated

        return activations

    def _get_unified_neighbors(
        self,
        node_id: str,
        profile_id: str,
        *,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> list[tuple[str, float]]:
        """Get neighbors from BOTH graph_edges and association_edges.

        Uses bidirectional UNION query (Section 4 of LLD).

        V3.4.40 (2026-05-09): clamps fan-out to top
        ``max_neighbors_per_node`` by weight. Without this clamp, hub nodes
        with thousands of neighbors caused 5.5s recalls. Bounded fan-out
        matches SYNAPSE's original sparse-graph assumption while preserving
        the highest-signal edges.
        """
        try:
            # v3.4.59: LIMIT pushed inside each UNION branch so SQLite can use
            # idx_edges_source_weight / idx_edges_target_weight and stop after
            # max_neighbors_per_node rows per branch instead of materializing
            # all 2.1M edges then sorting. Each branch wrapped in SELECT * FROM (...)
            # because SQLite requires parentheses for ORDER BY+LIMIT in compound SELECTs.
            lim = self._config.max_neighbors_per_node
            graph_where, graph_params = _scope_where(
                profile_id,
                include_global=include_global,
                include_shared=include_shared,
                prefix="ge",
            )
            # association_edges has no scope/shared_with columns in the current
            # schema, so it remains owner-profile-only. Endpoint authorization
            # below still prevents a private candidate from entering results.
            assoc_where = "ae.profile_id = ?"
            assoc_params = [profile_id]
            rows = self._db.execute(
                f"""
                SELECT neighbor_id, weight FROM (
                    SELECT * FROM (
                        SELECT target_id AS neighbor_id, weight FROM graph_edges AS ge
                        WHERE source_id = ? AND {graph_where}
                        ORDER BY weight DESC, target_id ASC LIMIT ?
                    )
                    UNION ALL
                    SELECT * FROM (
                        SELECT target_fact_id AS neighbor_id, weight
                        FROM association_edges AS ae
                        WHERE source_fact_id = ? AND {assoc_where}
                        ORDER BY weight DESC, target_fact_id ASC LIMIT ?
                    )
                    UNION ALL
                    SELECT * FROM (
                        SELECT source_id AS neighbor_id, weight FROM graph_edges AS ge
                        WHERE target_id = ? AND {graph_where}
                        ORDER BY weight DESC, source_id ASC LIMIT ?
                    )
                    UNION ALL
                    SELECT * FROM (
                        SELECT source_fact_id AS neighbor_id, weight
                        FROM association_edges AS ae
                        WHERE target_fact_id = ? AND {assoc_where}
                        ORDER BY weight DESC, source_fact_id ASC LIMIT ?
                    )
                )
                ORDER BY weight DESC, neighbor_id ASC
                LIMIT ?
                """,
                (
                    node_id, *graph_params, lim,
                    node_id, *assoc_params, lim,
                    node_id, *graph_params, lim,
                    node_id, *assoc_params, lim,
                    lim,
                ),
            )
            return [
                (dict(r)["neighbor_id"], dict(r)["weight"]) for r in rows
            ]
        except Exception as exc:
            logger.debug(
                "SpreadingActivation: UNION query failed for node %s "
                "profile %s: %s",
                node_id, profile_id, exc,
            )
            return []

    def _fok_check(self, activations: dict[str, float]) -> bool:
        """Feeling-of-Knowing gate.

        If max activation < tau_gate (0.12), reject results as noise.
        """
        if not activations:
            return False
        return max(activations.values()) >= self._config.tau_gate

    def _compute_query_hash(
        self,
        query: Any,
        profile_id: str,
        *,
        include_global: bool = False,
        include_shared: bool = False,
        seeds: list[tuple[str, float]] | None = None,
        graph: str | None = None,
    ) -> str:
        """Deterministic hash for cache key.

        ``seeds`` is the seed set this search starts from. It is part of the
        key because the activations are a function of it: before 4.1.20 a
        memory saved after a question was asked — even the best match, seeding
        first — stayed invisible to that question for the cache's full hour.
        Scores are rounded so float noise cannot defeat the cache.

        ``graph`` is the graph generation (``storage.graph_generation``). The
        activations are a function of the links too: before 4.1.21 a link that
        changed the walk without changing the seeds stayed hidden the same way.
        """
        scope_bytes = f"|g={int(include_global)}|s={int(include_shared)}".encode()
        if include_global or include_shared:
            scope_bytes += _CROSS_SCOPE_SEEDING.encode()
        if seeds is not None:
            scope_bytes += _seed_fingerprint(seeds).encode()
            # Which search produced the seeds. One scale on both paths makes
            # their walks agree, but the two indexes can still disagree (a
            # vector index lagging the store), so one never answers for the other.
            scope_bytes += (
                b"|path=vec" if self._uses_vector_seeds() else b"|path=sql"
            )
        if graph is not None:
            scope_bytes += graph.encode()
        if isinstance(query, np.ndarray):
            data = query.tobytes() + profile_id.encode() + scope_bytes
        elif isinstance(query, list):
            data = (
                np.array(query, dtype=np.float32).tobytes()
                + profile_id.encode()
                + scope_bytes
            )
        else:
            data = str(query).encode() + profile_id.encode() + scope_bytes
        return hashlib.sha256(data).hexdigest()[:16]

    def _get_cached_results(
        self, query_hash: str, profile_id: str,
    ) -> list[tuple[str, float]] | None:
        """Check activation_cache for recent results.

        One row per node, whatever the table holds: two identical recalls in
        flight at once used to store the entry twice, and a hit then returned
        every node twice.
        """
        try:
            rows = self._db.execute(
                "SELECT node_id, MAX(activation_value) AS activation_value "
                "FROM activation_cache "
                "WHERE profile_id = ? AND query_hash = ? "
                "AND expires_at > datetime('now') "
                "GROUP BY node_id",
                (profile_id, query_hash),
            )
            if not rows:
                return None
            # The same total order as a fresh propagation (value desc, then
            # id), so a tie cannot reorder the answer on a hit.
            return sorted(
                ((dict(r)["node_id"], dict(r)["activation_value"]) for r in rows),
                key=lambda item: (-item[1], item[0]),
            )
        except Exception:
            return None

    def _cache_results(
        self,
        query_hash: str,
        profile_id: str,
        activations: dict[str, float],
    ) -> None:
        """Store results in activation_cache with 1-hour TTL.

        Replaces the entry rather than adding to it: the old rows for this key
        are deleted in the same transaction, so two concurrent writers of one
        entry leave exactly one copy (SQLite serialises the two transactions).
        """
        try:
            # One transaction => ONE write-lock acquisition for the whole
            # activation cache, instead of N separately-locked writes.
            with self._db.transaction():
                self._db.execute(
                    "DELETE FROM activation_cache "
                    "WHERE profile_id = ? AND query_hash = ?",
                    (profile_id, query_hash),
                )
                for node_id, value in activations.items():
                    self._db.execute(
                        "INSERT OR REPLACE INTO activation_cache "
                        "(cache_id, profile_id, query_hash, node_id, "
                        " activation_value, iteration, created_at, expires_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, datetime('now'), "
                        "datetime('now', '+1 hour'))",
                        (_new_id(), profile_id, query_hash, node_id, value,
                         self._config.max_iterations),
                    )
        except Exception as exc:
            # Not debug: a cache that is never written is invisible otherwise,
            # and every identical recall then pays the full propagation.
            logger.warning("Spreading-activation cache write failed: %s", exc)

    def cleanup_expired_cache(self) -> int:
        """Delete expired cache entries. Called by maintenance."""
        try:
            result = self._db.execute(
                "DELETE FROM activation_cache "
                "WHERE expires_at < datetime('now')",
                (),
            )
            return len(result) if result else 0
        except Exception:
            return 0

    # ── v3.4.1: Graph Intelligence Helpers ────────────────────────

    def _load_graph_metrics_cache(self, profile_id: str) -> None:
        """Load PageRank + community data in a single SQL query.

        Called lazily on first _get_pagerank() or _get_community() call.
        Populates both _pr_cache and _comm_cache.
        """
        if self._pr_profile == profile_id and self._pr_cache:
            return  # Already loaded for this profile
        self._pr_cache = {}
        self._pr_profile = profile_id
        self._comm_cache = {}
        self._comm_profile = profile_id
        try:
            rows = self._db.execute(
                "SELECT fact_id, pagerank_score, community_id "
                "FROM fact_importance WHERE profile_id = ?",
                (profile_id,),
            )
            for r in rows:
                d = dict(r)
                self._pr_cache[d["fact_id"]] = float(d.get("pagerank_score", 0) or 0)
                self._comm_cache[d["fact_id"]] = d.get("community_id")
        except Exception:
            pass

    def _get_pagerank(self, fact_id: str, profile_id: str) -> float:
        """Look up PageRank score from fact_importance. Cached per profile."""
        self._load_graph_metrics_cache(profile_id)
        return self._pr_cache.get(fact_id, 0.0)

    def _get_community(self, fact_id: str, profile_id: str) -> int | None:
        """Look up community_id from fact_importance. Shares unified cache."""
        self._load_graph_metrics_cache(profile_id)
        return self._comm_cache.get(fact_id)
