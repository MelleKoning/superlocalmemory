# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""How a recalled memory is handed to the answer check, and what an unmeasured
(backend, recipe) pair is allowed to claim.

Both backends read these helpers, so a mistake here is a mistake in Laya and in
Jev at once.
"""

from __future__ import annotations

from superlocalmemory.retrieval import judge_recipe
from superlocalmemory.retrieval.judge_recipe import (
    RECIPE_V1,
    UNMEASURED,
    JudgeDocument,
    calibration_or_unmeasured,
    coerce_documents,
    document_from_fact,
)
from superlocalmemory.storage.models import AtomicFact, FactType


class TestAFactBecomesADocument:
    def test_the_document_is_exactly_what_the_model_reads(self) -> None:
        """F12: the recipe sends a memory's text and nothing else, so the
        document carries nothing else — no type, date or names a reader would
        assume the model sees. Changing that would change the measured recipe."""
        import dataclasses

        fact = AtomicFact(
            content="Alice moved to Paris", fact_type=FactType.EPISODIC,
            observation_date="2026-03-01", created_at="2026-03-02T10:00:00",
            entities=["Alice", "Paris"],
            canonical_entities=["9f3a1c2b7d4e5f60", "0c1d2e3f4a5b6c7d"],
        )
        assert [f.name for f in dataclasses.fields(JudgeDocument)] == ["content"]
        assert document_from_fact(fact) == JudgeDocument(content="Alice moved to Paris")
        assert RECIPE_V1.render(document_from_fact(fact)) == "Alice moved to Paris"

    def test_the_text_is_cut_at_the_recipes_limit(self) -> None:
        fact = AtomicFact(content="x" * 5000)
        assert RECIPE_V1.render(document_from_fact(fact)) == "x" * RECIPE_V1.max_document_chars

    def test_a_fact_double_never_breaks_it(self) -> None:
        """A MagicMock-shaped or partial fact yields a document, never a raise."""
        class _Partial:
            content = 7
            fact_type = None

        doc = document_from_fact(_Partial())
        assert doc == JudgeDocument(content="")


class TestDocumentsAreCoerced:
    def test_plain_strings_still_work(self) -> None:
        """Callers written before the recipe passed a list of strings."""
        assert coerce_documents(["a", JudgeDocument("b")]) == (
            JudgeDocument("a"), JudgeDocument("b"))

    def test_anything_else_is_dropped(self) -> None:
        assert coerce_documents([None, 3, "a", b"bytes"]) == (JudgeDocument("a"),)


class TestAnUnmeasuredPairCannotAbstain:
    def test_a_measured_pair_keeps_its_threshold(self) -> None:
        assert calibration_or_unmeasured("laya", RECIPE_V1).threshold == 0.6

    def test_an_unmeasured_pair_gets_a_threshold_nothing_falls_below(self) -> None:
        recipe = judge_recipe.JudgeRecipe("sufficiency-never-measured", question="q?")
        calibration = calibration_or_unmeasured("laya", recipe)
        assert calibration is UNMEASURED
        # Every valid probability is >= 0, so "confidence < threshold" is never true.
        assert calibration.threshold <= 0.0
        assert "not_measured" in calibration.status
