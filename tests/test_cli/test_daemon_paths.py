# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Unit tests for the shared safe-path-segment helpers (issue #148)."""

from __future__ import annotations

import pytest

from superlocalmemory.cli.daemon_paths import (
    InvalidDaemonId,
    describe,
    quote_path_segment,
    validate_daemon_id,
)


class TestValidateDaemonId:
    @pytest.mark.parametrize("value", [
        "a" * 32,                     # uuid.uuid4().hex shape
        "abcd1234efgh5678",            # uuid.uuid4().hex[:16] shape
        "550e8400-e29b-41d4-a716-446655440000",  # str(uuid.uuid4()) shape
        "simple-profile_name1",
    ])
    def test_accepts_daemon_shaped_ids_unchanged(self, value):
        assert validate_daemon_id(value, label="id") == value

    @pytest.mark.parametrize("value", ["…", "a/b c?", "", "has space", "slash/in/it"])
    def test_rejects_anything_else(self, value):
        with pytest.raises(InvalidDaemonId):
            validate_daemon_id(value, label="id")

    def test_rejected_value_is_kept_for_the_caller(self):
        with pytest.raises(InvalidDaemonId) as exc_info:
            validate_daemon_id("…", label="operation ID")
        assert exc_info.value.value == "…"
        assert exc_info.value.label == "operation ID"


class TestDescribe:
    def test_is_always_ascii(self):
        for value in ["…", "a/b c?", "\x00\x01", "plain"]:
            describe(value).encode("ascii")  # must never raise

    def test_never_contains_the_raw_character(self):
        assert "…" not in describe("…")

    def test_plain_ascii_stays_readable(self):
        assert describe("a/b c?") == "'a/b c?'"


class TestQuotePathSegment:
    def test_round_trips_url_safe_characters(self):
        assert quote_path_segment("abc-123_XYZ") == "abc-123_XYZ"

    def test_escapes_the_unsafe_characters_issue_148_hit(self):
        assert "/" not in quote_path_segment("a/b c?")
        assert " " not in quote_path_segment("a/b c?")
        # percent-encoding never raises, even for the literal ellipsis
        quote_path_segment("…")
