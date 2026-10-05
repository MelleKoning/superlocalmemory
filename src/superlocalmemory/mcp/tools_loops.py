# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SLM v3.8.0 — MCP bounded-loop tools.

Exposes the ``superlocalmemory.loops`` engine over MCP so an agent connected
via MCP — not only the ``slm loop`` CLI or the ``/slm-loop`` command — can run
a gated, bounded loop and inspect its durable ledger. Bounded loops therefore
ship on three surfaces: CLI, plugin command, and MCP.

The loop's one invariant holds identically here: an INDEPENDENT gate decides
when the loop is done, never the agent's own claim. Over MCP the gate is an SLM
*recall*: the loop converges the first lap a memory matching ``gate_query``
becomes retrievable with confidence (the "verification lives in memory" model
the engine is designed around). That makes ``slm_loop_run`` a safe, shell-free
multi-agent coordination primitive — one agent waits, under strict bounds, for
a memory another agent will write into shared SLM.

Three tools:
  * ``slm_loop_run``     — run one bounded, gate-verified loop to a terminal
                           outcome. Blocks (polling the gate) until the gate
                           passes or a bound trips. Every lap is persisted to
                           SLM memory (tag ``loop:<name>``) and shows on the
                           dashboard.
  * ``slm_loop_history`` — list recorded runs for a loop name (read-only).
  * ``slm_loop_show``    — show every lap of one run (read-only).

Fail-open: every tool body returns a dict; internal errors surface as
``ok: False`` with a message, never a raised exception.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any, Callable

from mcp.types import ToolAnnotations

from superlocalmemory.core.admission import admits
from superlocalmemory.core.operation_request import OperationKind
from superlocalmemory.loops import (
    Bounds,
    LapResult,
    LedgerEntry,
    Verdict,
    engine_backed_ledger,
    pool_backed_ledger,
    run_bounded_loop,
)
from superlocalmemory.retrieval.answer_check_status import REQUEST_NO_REORDER

logger = logging.getLogger("slm.mcp.tools_loops")

# ─── Exported tool name list (used by server.py + tests) ─────────────────────

_LOOP_TOOL_NAMES = (
    "slm_loop_run",
    "slm_loop_history",
    "slm_loop_show",
)

# ─── Hard caps (a loop tool must never hang the daemon or spin unbounded) ────

_MAX_ITERATIONS = 200
_MAX_WALLCLOCK_S = 120.0
_MIN_POLL_S = 0.25
_MAX_NAME_CHARS = 128
_MAX_QUERY_CHARS = 2000
#: Results each gate lap asks for — and so the most memories its answer check
#: ever judges.
_GATE_LIMIT = 3


def _gate_recall_kwargs(recall: Callable) -> dict:
    """The gate's per-recall request, for a recall that can carry it.

    The gate needs the answer check's verdict, never the opt-in reorder. A pool
    whose transport does not forward the request yet is called as before.
    """
    try:
        params = inspect.signature(recall).parameters
    except (TypeError, ValueError):
        return {}
    takes_it = "answer_check" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    return {"answer_check": REQUEST_NO_REORDER} if takes_it else {}


def _loop_profile(engine: Any, profile_id: str) -> tuple[str | None, dict | None]:
    """``(profile, refusal)`` for a loop read; ``None`` = the engine's own."""
    from superlocalmemory.mcp.request_profile import requested_profile, tool_profile

    if not requested_profile(profile_id):
        return None, None
    return tool_profile(engine, profile_id)


def register_loop_tools(
    server,
    get_engine: Callable,
    get_pool: Callable | None = None,
) -> None:
    """Register the 3 bounded-loop tools on *server*.

    *server* is duck-typed: must support the ``@server.tool()`` decorator.
    Compatible with FastMCP, _FilteredServer, and the test mock server.
    """

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
    @admits(OperationKind.REMEMBER)
    async def slm_loop_run(
        name: str,
        gate_query: str,
        gate_min_score: float = 0.0,
        max_iterations: int = 20,
        max_wallclock_s: float = 15.0,
        poll_interval_s: float = 1.0,
        max_tokens: int = 0,
        no_progress_window: int = 0,
    ) -> dict:
        """Run one bounded loop that finishes only when an INDEPENDENT gate passes.

        The gate is an SLM recall of ``gate_query``: the loop converges on the
        first lap that recall returns a confident match scoring at least
        ``gate_min_score``. The agent cannot end the loop by asserting it is
        done — only the gate can. The call BLOCKS (polling every
        ``poll_interval_s``) until the gate passes or a bound trips, then
        returns the outcome. Every lap is written to SLM memory (tag
        ``loop:<name>``) and is visible on the dashboard.

        Use it to wait, under strict bounds, for a verification or coordination
        condition to hold — e.g. for another agent to record a "build passed"
        memory in shared SLM.

        Cost per lap: one recall of at most 3 memories. When the answer check
        is on, each lap asks it once about those memories (never the optional
        reordering): on this Mac that is one local judgement; with the online
        check it is one request to the provider, which may be billed. A run
        makes at most ``max_iterations`` laps.

        Args:
            name: Loop name, also the memory tag. 1–128 chars.
            gate_query: Recall query the independent gate checks each lap.
            gate_min_score: Minimum top-result score to pass (0.0 = any
                confident hit above the evidence floor).
            max_iterations: Hard cap on laps (1–200).
            max_wallclock_s: Hard cap on wall-clock seconds (0 disables; capped at 120).
            poll_interval_s: Seconds to wait between laps (minimum 0.25).
            max_tokens: Optional token budget (0 disables).
            no_progress_window: Halt after this many consecutive no-change laps
                (0 disables). A pure watcher never "changes", so leave at 0
                unless the runner reports progress.
        """
        try:
            name = (name or "").strip()
            if not name or len(name) > _MAX_NAME_CHARS:
                return {"ok": False, "error": f"name must be 1–{_MAX_NAME_CHARS} chars"}
            gate_query = (gate_query or "").strip()
            if not gate_query or len(gate_query) > _MAX_QUERY_CHARS:
                return {
                    "ok": False,
                    "error": f"gate_query must be 1–{_MAX_QUERY_CHARS} chars",
                }

            iters = max(1, min(int(max_iterations), _MAX_ITERATIONS))
            wall = (
                min(float(max_wallclock_s), _MAX_WALLCLOCK_S)
                if max_wallclock_s and float(max_wallclock_s) > 0
                else None
            )
            poll = max(float(poll_interval_s), _MIN_POLL_S)
            tok = int(max_tokens) if max_tokens and int(max_tokens) > 0 else None
            # 0 = disabled (a pure watcher never "changes", so no-progress must
            # be off by default or it would halt before the gate can pass).
            npw = (
                int(no_progress_window)
                if no_progress_window and int(no_progress_window) > 0
                else 0
            )
            min_score = float(gate_min_score)

            engine = get_engine()
            pool = get_pool() if get_pool is not None else None

            def gate(lap: int) -> Verdict:
                if pool is not None:
                    resp = pool.recall(gate_query, limit=_GATE_LIMIT, fast=True,
                                       **_gate_recall_kwargs(pool.recall))
                    if not isinstance(resp, dict) or resp.get("ok") is False:
                        raise RuntimeError(
                            (resp or {}).get("error", "owned SLM reader rejected recall")
                            if isinstance(resp, dict)
                            else "owned SLM reader returned an invalid response"
                        )
                    all_results = resp.get("results", []) or []
                    floored = bool(resp.get("no_confident_match", False))
                    reason = resp.get("abstention_reason")
                else:
                    resp = engine.recall(gate_query, limit=_GATE_LIMIT, fast=True,
                                         answer_check=REQUEST_NO_REORDER)
                    all_results = getattr(resp, "results", None) or []
                    floored = bool(getattr(resp, "no_confident_match", False))
                    reason = getattr(resp, "abstention_reason", None)
                # The score says a memory is related; the answer check says
                # whether it answers. A gate must never pass on a recall the
                # check already called insufficient.
                judged_out = reason == "judged_insufficient"

                def _content(result: Any) -> str:
                    if isinstance(result, dict):
                        fact = result.get("fact")
                        if isinstance(fact, dict):
                            return str(fact.get("content", "") or "")
                        return str(result.get("content", "") or "")
                    return str(
                        getattr(getattr(result, "fact", None), "content", "") or ""
                    )

                def _score(result: Any) -> float:
                    if isinstance(result, dict):
                        value = result.get("score", result.get("relevance_score", 0.0))
                    else:
                        value = getattr(result, "score", None)
                        if value is None:
                            value = getattr(result, "relevance_score", 0.0) or 0.0
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return 0.0

                # Exclude the loop's own audit records from the gate evaluation.
                # Each lap writes a LedgerEntry (JSON with "run_id" + "lap") into
                # SLM via store_fast; the FTS5 trigger indexes every such insert.
                # Without this filter BM25 finds those self-authored records on the
                # next lap and the gate falsely passes. Only memories written by an
                # external agent — whose content does not parse as a LedgerEntry —
                # can satisfy an independent gate.
                results = [
                    r for r in all_results
                    if LedgerEntry.from_json(_content(r)) is None
                ]
                best = max((_score(result) for result in results), default=0.0)
                passed = (bool(results) and not floored and not judged_out
                          and best >= min_score)
                return Verdict(
                    passed,
                    f"recall '{gate_query[:48]}': hits={len(results)} "
                    f"top={best:.3f} floor={floored} judged_insufficient={judged_out}",
                )

            def runner(lap: int) -> LapResult:
                # Watcher lap: no work of our own. Give the gate condition time
                # to become true (e.g. another agent writing a memory) between
                # polls. Bounded by poll interval so the wall-clock / iteration
                # caps stay meaningful. The first lap checks immediately.
                if lap > 1:
                    time.sleep(poll)
                return LapResult(changed=False, tokens=0)

            ledger = (
                pool_backed_ledger(pool, engine)
                if pool is not None
                else engine_backed_ledger(engine)
            )

            # The loop blocks (sleeps between laps); run it off the event loop.
            # The engine's per-call WAL connection model makes this thread-safe.
            outcome = await asyncio.to_thread(
                run_bounded_loop,
                name,
                bounds=Bounds(
                    max_iterations=iters,
                    max_tokens=tok,
                    max_wallclock_s=wall,
                    no_progress_window=npw,
                ),
                runner=runner,
                gate=gate,
                ledger=ledger,
            )
            laps = await asyncio.to_thread(ledger.laps, outcome.run_id)
            return {
                "ok": True,
                "status": outcome.status.value,
                "reason": outcome.reason,
                "passed": bool(outcome.ok),
                "laps": outcome.laps,
                "run_id": outcome.run_id,
                "ledger": [
                    {
                        "lap": e.lap,
                        "decision": e.decision,
                        "passed": e.passed,
                        "detail": e.detail,
                    }
                    for e in laps
                ],
                "note": (
                    "The gate — an independent SLM recall — decided this "
                    "outcome; the agent's own done-claim never terminates a loop."
                ),
            }
        except Exception as exc:
            logger.exception("slm_loop_run failed (fail-open)")
            return {"ok": False, "error": str(exc)}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def slm_loop_history(name: str, limit: int = 20, profile_id: str = "") -> dict:
        """List recorded bounded-loop runs for a loop name, newest laps summarised.

        Reads the durable SLM-backed ledger (the same rows ``slm loop history``
        and the dashboard show). Read-only.

        Args:
            name: Loop name to list runs for.
            limit: Maximum runs to return (1–200).
            profile_id: The profile whose runs to list (empty = the active one).
        """
        try:
            name = (name or "").strip()
            if not name:
                return {"ok": False, "error": "name is required"}
            lim = max(1, min(int(limit), 200))
            engine = get_engine()
            pid, refused = _loop_profile(engine, profile_id)
            if refused:
                return {"ok": False, **refused}
            ledger = engine_backed_ledger(engine, pid)

            def _collect() -> list[dict]:
                run_ids = ledger.runs(name)[:lim]
                rows: list[dict] = []
                for rid in run_ids:
                    laps = ledger.laps(rid)
                    last = laps[-1] if laps else None
                    rows.append(
                        {
                            "run_id": rid,
                            "laps": len(laps),
                            "final": last.decision if last else "unknown",
                            "ts": last.ts if last else "",
                        }
                    )
                return rows

            rows = await asyncio.to_thread(_collect)
            return {"ok": True, "name": name, "count": len(rows), "runs": rows}
        except Exception as exc:
            logger.exception("slm_loop_history failed (fail-open)")
            return {"ok": False, "error": str(exc)}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def slm_loop_show(run_id: str, limit: int = 200, profile_id: str = "") -> dict:
        """Show every lap of one bounded-loop run, in order, from SLM memory.

        Read-only. Each lap carries the gate verdict plus the agent's recorded
        (advisory, never loop-terminating) done-claim.

        Args:
            run_id: Run identifier returned by ``slm_loop_run``.
            limit: Maximum laps to return (1–1000).
            profile_id: The profile the run was recorded in (empty = the active one).
        """
        try:
            run_id = (run_id or "").strip()
            if not run_id:
                return {"ok": False, "error": "run_id is required"}
            lim = max(1, min(int(limit), 1000))
            engine = get_engine()
            pid, refused = _loop_profile(engine, profile_id)
            if refused:
                return {"ok": False, **refused}
            ledger = engine_backed_ledger(engine, pid)

            def _collect() -> list[dict]:
                return [
                    {
                        "lap": e.lap,
                        "ts": e.ts,
                        "decision": e.decision,
                        "passed": e.passed,
                        "detail": e.detail,
                        "agent_claimed_done": e.agent_claimed_done,
                        "tokens": e.budget.get("tokens", 0),
                    }
                    for e in ledger.laps(run_id)[:lim]
                ]

            laps = await asyncio.to_thread(_collect)
            return {"ok": True, "run_id": run_id, "count": len(laps), "laps": laps}
        except Exception as exc:
            logger.exception("slm_loop_show failed (fail-open)")
            return {"ok": False, "error": str(exc)}
