# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Mode A says "nothing leaves this device" only while that is true.

The online answer check can be turned on in any mode (the owner wants it
optional everywhere). Mode A stays local by default; once the online check
is on, the mode's own description says plainly what now leaves the machine.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from superlocalmemory.core.egress_notice import online_check_notice


def _retrieval(**overrides):
    base = dict(sufficiency_judge="auto", sufficiency_jev_consent=False,
                sufficiency_jev_provider="typesafe", sufficiency_jev_rerank=False,
                sufficiency_jev_rerank_consent=False, sufficiency_jev_rerank_k=20)
    base.update(overrides)
    return SimpleNamespace(**base)


ON = dict(sufficiency_judge="jev", sufficiency_jev_consent=True)


def test_mode_a_is_local_while_the_online_check_is_off():
    from superlocalmemory.mcp.tools_v3 import _mode_description

    text = _mode_description("a", _retrieval())
    assert "nothing leaves this device" in text
    assert "online answer check" not in text


@pytest.mark.parametrize("mode", ["a", "b"])
def test_mode_a_and_b_say_what_leaves_once_the_online_check_is_on(mode):
    from superlocalmemory.mcp.tools_v3 import _mode_description

    text = _mode_description(mode, _retrieval(**ON))
    assert "nothing leaves this device" not in text
    assert "online answer check is on" in text
    assert "TypeSafe" in text


def test_the_reordering_count_is_named_when_it_is_on():
    notice = online_check_notice(_retrieval(
        **ON, sufficiency_jev_provider="openrouter", sufficiency_jev_rerank=True,
        sufficiency_jev_rerank_consent=True, sufficiency_jev_rerank_k=20))
    assert "OpenRouter" in notice
    assert "20" in notice


@pytest.mark.parametrize("overrides", [
    {"sufficiency_judge": "jev", "sufficiency_jev_consent": "true"},
    {"sufficiency_judge": "laya", "sufficiency_jev_consent": True},
    {"sufficiency_judge": "off", "sufficiency_jev_consent": True},
    {},
])
def test_no_notice_unless_the_online_check_is_really_on(overrides):
    assert online_check_notice(_retrieval(**overrides)) == ""


def test_get_mode_reports_the_online_check(monkeypatch):
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.mcp import tools_v3
    from superlocalmemory.storage.models import Mode

    config = SLMConfig.for_mode(Mode.A)
    config.retrieval.sufficiency_judge = "jev"
    config.retrieval.sufficiency_jev_consent = True
    config.save(mode_change=True)

    class _Srv:
        tools: dict = {}

        def tool(self, *a, **k):
            def deco(fn):
                _Srv.tools[fn.__name__] = fn
                return fn
            return deco

    engine = SimpleNamespace(_config=config, _llm=None, profile_id="default")
    tools_v3.register_v3_tools(_Srv(), lambda: engine)
    out = asyncio.run(_Srv.tools["get_mode"]())
    assert out["success"] is True
    assert out["online_answer_check"] is True
    assert "online answer check is on" in out["description"]
