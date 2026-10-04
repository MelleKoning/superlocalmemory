# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The Answer Check tab is reachable: first in Overview, tagged new, its pane
and scripts present, and every script stamped with its own content hash."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_UI = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory" / "ui"
_JS = _UI / "js"


def _html() -> str:
    return (_UI / "index.html").read_text(encoding="utf-8")


def test_nav_puts_answer_check_first_in_overview() -> None:
    shell = (_JS / "od-shell.js").read_text(encoding="utf-8")
    overview = shell[shell.index("{ g: 'Overview', items: ["):shell.index("{ g: 'Memory'")]
    first = re.search(r"\{ k: '([a-z-]+)'[^}]*\}", overview)
    assert first and first.group(1) == "answercheck-pane"
    assert "t: 'Answer Check'" in first.group(0) and "tag: 'new'" in first.group(0)
    assert "case 'answercheck-pane':\n        od('odRenderAnswerCheck');" in shell
    assert "var active = opts.active || 'dashboard-pane';" in shell, "landing pane unchanged"


def test_index_has_the_empty_pane_and_hashed_scripts() -> None:
    html = _html()
    assert re.search(r'<div class="tab-pane fade" id="answercheck-pane"[^>]*></div>', html)
    for name in ("od-answercheck-tryit.js", "od-answercheck.js", "od-shell.js", "od-settings.js"):
        expected = hashlib.sha256((_JS / name).read_bytes()).hexdigest()[:8]
        m = re.search(rf'static/js/{re.escape(name)}\?v=([0-9a-f]+)"', html)
        assert m and m.group(1) == expected, name
    assert html.index("od-answercheck-tryit.js") < html.index("od-answercheck.js?v=")
    assert html.index("answer-check.js?v=") < html.index("od-answercheck-tryit.js")


def test_settings_group_is_a_deep_link_target() -> None:
    settings = (_JS / "od-settings.js").read_text(encoding="utf-8")
    assert "g.id = 'settings-answer-check';" in settings
