# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The secret scan must not destroy a link or a file path.

The same scan cleans memory text before it is sent to another machine (a
hosted model, a cloud embedder, the online answer check), so anything it wrongly
removes is missing from what that service reads. A mixed-case URL or path as a whole clears the
randomness bar a key clears; judged one segment at a time it does not.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.security_primitives import redact_secrets

KEPT = (
    "Docs: https://github.com/qualixar/superlocalmemory/blob/main/CHANGELOG.md",
    "Repo at /Users/yourusername/Documents/client-work/agentic-projects/slm",
    "Thumbnail https://img.youtube.com/vi/PMWW_ypsL60/hqdefault.jpg in the README",
    "Config: ~/.superlocalmemory/runtimes/laya/huggingface/hub/models--laya/snapshots",
    "See github.com/qualixar/superlocalmemory/wiki/MCP-Integration for setup",
)

#: 32 distinct characters: as random-looking as a key gets.
RANDOM_SEGMENT = "AbCdEfGhIjKlMnOpQrStUvWxYz012345"


@pytest.mark.parametrize("aggression", ["normal", "high"])
@pytest.mark.parametrize("text", KEPT)
def test_a_link_or_a_path_survives_the_scan(text: str, aggression: str) -> None:
    assert redact_secrets(text, aggression=aggression) == text


@pytest.mark.parametrize("aggression", ["normal", "high"])
def test_a_random_segment_inside_a_link_is_still_redacted(aggression: str) -> None:
    text = f"download https://files.example.com/share/{RANDOM_SEGMENT}/report.pdf"
    out = redact_secrets(text, aggression=aggression)
    assert RANDOM_SEGMENT not in out
    assert "https://files.example.com/share/" in out
    assert "/report.pdf" in out


@pytest.mark.parametrize("aggression", ["normal", "high"])
def test_a_base64_key_with_slashes_is_still_redacted_whole(aggression: str) -> None:
    key = "".join(("wJalrXUtnFEMI", "/K7MDENG/", "bPxRfiCYEXAMPLEKEY"))
    out = redact_secrets(f"the secret access key {key}", aggression=aggression)
    assert key not in out
    assert "K7MDENG" not in out


def test_a_webhook_url_is_still_scrubbed_before_it_leaves_the_machine():
    """Judging a URL by segment must not let a webhook's secret path through:
    its vendor shape is still caught."""
    secret = "".join(("FAKEfake", "FAKEfake", "FAKEfake"))
    url = "".join(("https://hooks.slack.com/services/", "T00000000/B00000000/", secret))
    assert secret not in redact_secrets(f"alerts go to {url}")


def test_a_private_key_body_never_leaves_the_machine():
    body = "MIIE" + "FAKE" * 15
    pem = "".join(("-----BEGIN RSA ", "PRIVATE KEY-----\n", body, "\n-----END RSA ",
                   "PRIVATE KEY-----"))
    assert body not in redact_secrets(pem)
