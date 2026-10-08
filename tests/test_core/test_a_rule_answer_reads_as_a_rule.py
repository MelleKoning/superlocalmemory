# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A recall the permission rule settled must not read "likely answered (confidence 0.08)".

The on-device check keeps the model's own number when an explicit rule in the
memories answers a yes/no permission question (retrieval/answer_question_forms).
Every surface already carries ``calibration_id``, which names what decided the
verdict, so the rule's id travels there — on verdicts the rule actually settled
and on no other — and both line builders read it. No new field.
"""

from __future__ import annotations

from superlocalmemory.cli.commands import _answer_check_line
from superlocalmemory.core.answer_check_notice import answer_check_line
from superlocalmemory.core.score_contract import finalize_score_contract
from superlocalmemory.retrieval import answer_question_forms as forms
from superlocalmemory.retrieval.sufficiency import SufficiencyVerdict
from superlocalmemory.server.recall_serializer import recall_response_metadata
from superlocalmemory.storage.models import AtomicFact, RecallResponse, RetrievalResult

BASE_ID = "laya:aac6fef/laya-mlx@20aed815fc6a:sufficiency-v1:top3"
RULE_LINE = ("Answer check: answered by a rule stated in your memories "
             "(the model's own confidence was 0.08).")


def _response(verdict: SufficiencyVerdict) -> RecallResponse:
    response = RecallResponse(results=[RetrievalResult(
        fact=AtomicFact(content="Never publish until the owner approves."),
        score=0.5, confidence=1.0)])
    finalize_score_contract(response, verdict=verdict)
    response.answer_check_status = "judged"
    response.answer_check_detail = ""
    return response


def _rule_verdict() -> SufficiencyVerdict:
    return SufficiencyVerdict((0.08, 0.06, 0.02), 0.6, BASE_ID + forms.RULE_SUFFIX,
                              rule_support=(0,))


def test_every_surface_says_a_rule_answered_it() -> None:
    response = _response(_rule_verdict())
    metadata = recall_response_metadata(response)  # what HTTP, MCP and the CLI read
    assert metadata["calibration_id"] == BASE_ID + "+permission-rule-v1"
    assert metadata["answerability"] == "supported"
    assert answer_check_line(response) == RULE_LINE
    assert answer_check_line(metadata) == RULE_LINE
    assert _answer_check_line(metadata) == RULE_LINE


def test_a_model_answer_still_reads_as_before() -> None:
    response = _response(SufficiencyVerdict((0.81, 0.1, 0.1), 0.6, BASE_ID))
    metadata = recall_response_metadata(response)
    expected = "Answer check: likely answered (confidence 0.81)."
    assert answer_check_line(response) == expected
    assert _answer_check_line(metadata) == expected


def test_a_reused_rule_verdict_keeps_both_facts() -> None:
    """``answer_check_reason`` says "reused"; the id still says the rule decided."""
    response = _response(_rule_verdict())
    response.answer_check_detail = "reused"
    metadata = recall_response_metadata(response)
    assert metadata["answerability_reason"] == "judged_from_memo"
    assert _answer_check_line(metadata) == RULE_LINE


def test_the_suffix_is_only_on_verdicts_the_rule_settled() -> None:
    assert forms.calibration_id_for(BASE_ID, (0,)) == BASE_ID + "+permission-rule-v1"
    assert forms.calibration_id_for(BASE_ID, ()) == BASE_ID
    assert forms.settled_by_rule(BASE_ID + forms.RULE_SUFFIX) is True
    assert forms.settled_by_rule(BASE_ID) is False
    assert forms.settled_by_rule(None) is False
