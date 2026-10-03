# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""docs/SECURITY-encryption-at-rest.md promised a per-account request check
"planned for 4.1.19". It did not ship in 4.1.19, so the promise must say
that plainly instead of naming a release that will not deliver it (L3-25).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SECURITY_DOC = REPO_ROOT / "docs" / "SECURITY-encryption-at-rest.md"


def test_does_not_promise_a_specific_unshipped_release() -> None:
    text = SECURITY_DOC.read_text(encoding="utf-8")
    assert "planned for 4.1.19" not in text, (
        "still promises the per-account request check for 4.1.19, which did "
        "not ship it"
    )


def test_states_the_feature_is_not_in_this_release() -> None:
    text = SECURITY_DOC.read_text(encoding="utf-8")
    assert "not in this release" in text.lower()
