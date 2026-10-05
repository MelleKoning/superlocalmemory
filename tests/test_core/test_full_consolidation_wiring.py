# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com

"""Steps 8-11 of consolidation must run, and must not fail silently.

WHY THIS EXISTS
---------------
Behavioural assertion mining, soft prompts, skill performance and skill
evolution are steps 8-11 of ``ConsolidationEngine.consolidate(lightweight=False)``.
Before 4.0.8 none of them had ever executed on a real install:

* **No trigger took that path.** The engine self-triggers with
  ``lightweight=True``; the session-end hook shelled out to
  ``slm consolidate --cognitive``, which runs ``CognitiveConsolidator`` — a
  different class with no steps 8-11.
* **When it finally ran, two steps crashed**, and both crashes were caught and
  logged at ``debug`` so nothing surfaced:
  - step 9 passed a ``DatabaseManager`` to ``CrossProjectAggregator`` and
    ``WorkflowMiner``, which take a path → ``Path(db)`` raised TypeError.
  - step 11 built ``EvolutionBudget`` against memory.db, but
    ``evolution_llm_cost_log`` is created by M010 in learning.db →
    "no such table".

The measurable symptom was a Brain pane full of zeros that looked like an empty
store but was an unrun pipeline. These tests pin the wiring, because every one
of these failures was invisible by construction.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest


def _code_of(fn) -> str:
    """Source of *fn* with docstrings and comments removed.

    These tests assert on what the function *does*, and the functions in
    question document the very bugs being pinned — the fix comment for the
    old ``--cognitive`` call contains the word ``--cognitive``. Round-tripping
    through ``ast.unparse`` drops comments and docstrings, so a prose mention
    can never be mistaken for live code.

    Quotes are stripped too: ``ast.unparse`` normalises ``"x"`` to ``'x'``, so
    substring checks must not depend on which the author typed.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    node = tree.body[0]
    body = node.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree).replace('"', "").replace("'", "")


class TestStepsAreWiredCorrectly:
    """Signature mismatches here fail at runtime inside a debug-swallowed try."""

    def test_cross_project_and_workflow_miner_take_a_path(self):
        """Both crashed for releases because they were handed a DatabaseManager."""
        from superlocalmemory.learning.cross_project import CrossProjectAggregator
        from superlocalmemory.learning.workflows import WorkflowMiner

        for cls in (CrossProjectAggregator, WorkflowMiner):
            params = list(inspect.signature(cls.__init__).parameters)
            assert params[1] == "db_path", (
                f"{cls.__name__} no longer takes db_path; consolidation step 9 "
                f"passes a path and will break"
            )

    def test_consolidation_step9_passes_a_path_not_a_manager(self):
        from superlocalmemory.core.consolidation_engine import ConsolidationEngine

        src = _code_of(ConsolidationEngine.consolidate)
        assert "CrossProjectAggregator(learning_db)" in src
        assert "WorkflowMiner(learning_db)" in src
        # The old bug, spelled out so a revert is caught by name.
        assert "CrossProjectAggregator(self._db)" not in src
        assert "WorkflowMiner(self._db)" not in src

    def test_no_dead_importerror_stubs_in_step9(self):
        """Both modules exist and re-export the real classes, so the stub
        branch could never run — it only hid the real signatures."""
        import superlocalmemory.parameterization.cross_project as cp
        import superlocalmemory.parameterization.workflow_miner as wm

        assert cp.CrossProjectAggregator is not None
        assert wm.WorkflowMiner is not None

        from superlocalmemory.core.consolidation_engine import ConsolidationEngine

        src = _code_of(ConsolidationEngine.consolidate)
        assert "except ImportError" not in src, "dead stub branch is back"

    def test_skill_evolver_budget_uses_learning_db(self, tmp_path):
        """evolution_llm_cost_log lives in learning.db (M010), but every
        production caller passes memory.db."""
        from superlocalmemory.evolution.skill_evolver import SkillEvolver

        (tmp_path / "memory.db").touch()
        evolver = SkillEvolver(tmp_path / "memory.db")
        budget_db = getattr(evolver._budget, "_learning_db", None) or getattr(
            evolver._budget, "learning_db", None
        )
        assert budget_db is not None
        assert str(budget_db).endswith("learning.db"), (
            f"budget points at {budget_db}, not learning.db"
        )

    @pytest.mark.parametrize("db_arg", [":memory:", "custom.db"])
    def test_non_production_db_names_are_left_alone(self, tmp_path, db_arg):
        """Only the exact production filename is redirected; tests must not get
        a learning.db invented beside their fixtures."""
        from superlocalmemory.evolution.skill_evolver import SkillEvolver

        arg = db_arg if db_arg == ":memory:" else str(tmp_path / db_arg)
        evolver = SkillEvolver(arg)
        budget_db = str(
            getattr(evolver._budget, "_learning_db", None)
            or getattr(evolver._budget, "learning_db", "")
        )
        assert not budget_db.endswith("learning.db")


class TestTriggersShareOneImplementation:
    """Two triggers, one lock. Two copies would be two definitions of done."""

    def test_runner_exists_and_serialises(self):
        from superlocalmemory.server import consolidation_runner as cr

        assert callable(cr.run_full_consolidation)
        assert callable(cr.is_running)
        assert not cr.is_running()

    def test_http_endpoint_delegates_to_the_runner(self):
        from superlocalmemory.server.routes import v3_api

        src = _code_of(v3_api.trigger_consolidation)
        assert "run_full_consolidation" in src
        # The endpoint must not carry its own inline copy any more.
        assert "def _run_consolidation" not in src

    def test_daemon_schedules_the_timer(self):
        from superlocalmemory.server import unified_daemon as ud

        assert callable(ud._consolidation_timer_loop)
        src = _code_of(ud._consolidation_timer_loop)
        # Idle gate is the owner's hard constraint: scheduled work must never
        # compete with live remember/recall traffic.
        assert "_CONSOLIDATION_IDLE_SEC" in src
        assert "_last_activity" in src
        # A skipped pass must not reset the clock.
        assert "if not result.get(skipped)" in src


class TestHookNoLongerLiesAboutSuccess:
    def test_hook_targets_the_full_path_not_ccq(self):
        from superlocalmemory.hooks import hook_handlers

        src = _code_of(hook_handlers._maybe_consolidate)
        assert "/api/v3/consolidation/trigger" in src
        assert "lightweight: False" in src
        # CognitiveConsolidator does not contain steps 8-11.
        assert "--cognitive" not in src

    def test_marker_is_written_only_after_the_daemon_accepts(self):
        from superlocalmemory.hooks import hook_handlers

        src = _code_of(hook_handlers._maybe_consolidate)
        post_at = src.index("_daemon_post")
        write_at = src.index("open(last_consolidation, w")  # + encoding=...
        assert post_at < write_at, (
            "timestamp written before the request — a failed run would buy "
            "24h of silence, which is the original bug"
        )

    def test_failures_are_no_longer_discarded(self):
        from superlocalmemory.hooks import hook_handlers

        src = _code_of(hook_handlers._maybe_consolidate)
        assert "DEVNULL" not in src
        assert "stderr" in src
