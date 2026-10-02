# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The degraded keyword recall never runs the answer check, and says so.

It is the answer a recall gives when the normal path ran out of time. Every
other recall response names what became of the answer check; this one must
too, or a caller cannot tell "not checked" from a response that predates the
field.
"""

from __future__ import annotations


def test_the_keyword_fallback_reports_the_check_as_skipped(engine_with_mock_deps) -> None:
    from superlocalmemory.server.unified_daemon import _recall_keyword_fallback

    body = _recall_keyword_fallback(
        engine_with_mock_deps, "Harbor crane", 5,
        profile_id="b", profile="b", profile_generation=7,
    )
    assert body["answer_check_status"] == "skipped"
    assert body["abstained"] is False
    assert body["answer_confidence"] is None
