# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Mode B/C: the extraction call already running also suggests a kind, at no extra call.

The kind rides on the fact as a ``model:llm`` suggestion. It never changes the
fact's legacy type, and a flag removes the prompt line entirely so extraction
quality can be compared with and without it.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from superlocalmemory.core.config import EncodingConfig
from superlocalmemory.encoding import fact_extractor as fx
from superlocalmemory.encoding.fact_extractor import FactExtractor
from superlocalmemory.encoding.memory_kind_recipe import LLM_RECIPE
from superlocalmemory.storage.memory_kinds import KindSource, MemoryKind
from superlocalmemory.storage.models import FactType, Mode


def _llm(items: list[dict]) -> MagicMock:
    llm = MagicMock()
    llm.is_available.return_value = True
    llm.generate.return_value = json.dumps(items)
    return llm


def _extract(items: list[dict], **kw) -> tuple[list, MagicMock]:
    llm = _llm(items)
    ext = FactExtractor(config=EncodingConfig(enable_entity_reflexion=False), llm=llm,
                        mode=Mode.C, **kw)
    return ext.extract_facts(["turn one", "turn two"], session_id="s1",
                             session_date="2026-10-03"), llm


def _item(text: str, **kw) -> dict:
    return {"text": text, "fact_type": "semantic", "entities": [], "importance": 6,
            "confidence": 0.9, **kw}


def test_valid_kind_becomes_model_llm_hint() -> None:
    facts, llm = _extract([
        _item("Alice chose Postgres over DynamoDB for billing.", kind="decision"),
        _item("Never deploy the billing service on Fridays.", kind=" Instruction "),
    ])
    by_text = {f.content: f for f in facts}
    decision = by_text["Alice chose Postgres over DynamoDB for billing."]
    rule = by_text["Never deploy the billing service on Fridays."]
    assert (decision.memory_kind, decision.memory_kind_source, decision.memory_kind_recipe,
            decision.memory_kind_confidence) == (
        MemoryKind.DECISION.value, KindSource.MODEL_LLM.value, LLM_RECIPE, None)
    assert decision.memory_kind_at
    assert rule.memory_kind == MemoryKind.RULE.value, "aliases are accepted"
    # The legacy type is the extractor's own 4-way answer, untouched.
    assert decision.fact_type is FactType.SEMANTIC and rule.fact_type is FactType.SEMANTIC
    system = llm.generate.call_args.kwargs["system"]
    assert '"kind"' in system
    for kind in MemoryKind:
        assert kind.value in system


def test_invalid_kind_ignored() -> None:
    facts, _ = _extract([
        _item("Alice works at Google as an engineer.", kind="banana"),
        _item("Bob lives in Pune near the river.", kind=7),
        _item("Carol moved to Berlin last spring.", kind=None),
        _item("Dave owns a red bicycle and a scooter."),
    ])
    assert len(facts) == 4
    for fact in facts:
        assert fact.memory_kind is None and fact.memory_kind_source is None
        assert fact.memory_kind_recipe is None and fact.memory_kind_at is None


def test_flag_off_drops_prompt_line() -> None:
    facts, llm = _extract([_item("Alice chose Postgres over DynamoDB.", kind="decision")],
                          llm_extract_kinds=False)
    system = llm.generate.call_args.kwargs["system"]
    assert system == fx._SYSTEM_PROMPT, "flag off must send exactly the 4.1.18 prompt"
    assert '"kind"' not in system
    assert facts[0].memory_kind is None, "flag off: a stray kind field is not trusted"


def test_flag_on_is_the_default_and_changes_only_the_kind_lines() -> None:
    _, llm = _extract([_item("Alice chose Postgres over DynamoDB.")])
    system = llm.generate.call_args.kwargs["system"]
    added = set(system.splitlines()) - set(fx._SYSTEM_PROMPT.splitlines())
    assert len(added) == 2, added   # the instruction line and the example line
