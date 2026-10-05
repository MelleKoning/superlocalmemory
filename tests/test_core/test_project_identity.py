# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""One rule for "which project is this" (GitHub #150)."""

from __future__ import annotations

import unicodedata

import pytest

from superlocalmemory.core.project_identity import (
    project_key,
    project_name,
    session_context_query,
    session_end_project,
    storable_project,
)


@pytest.mark.parametrize("value", [
    "superlocalmemory", "SuperLocalMemory", "  superlocalmemory  ",
    "/Users/x/Documents/superlocalmemory", "/Users/x/Documents/superlocalmemory/",
    "/Users/x/Documents/superlocalmemory//", "C:\\work\\SuperLocalMemory",
    "C:\\work\\SuperLocalMemory\\", "~/code/superlocalmemory",
])
def test_names_and_paths_of_one_project_agree(value) -> None:
    assert project_key(value) == "superlocalmemory"


@pytest.mark.parametrize("value", [None, "", "   ", "/", "\\", "//", "C:", "C:\\", "~",
                                   ".", "..", 42, ["slm"]])
def test_values_that_name_no_project(value) -> None:
    assert project_key(value) is None
    assert storable_project(value) == ""


def test_decomposed_unicode_matches_composed() -> None:
    composed = "café-app"
    decomposed = unicodedata.normalize("NFD", composed)
    assert composed != decomposed
    assert project_key("/p/" + decomposed) == project_key(composed.upper())


def test_names_with_spaces_keep_their_spaces() -> None:
    assert project_key("/Users/x/testing - automation") == "testing - automation"


def test_storable_project_trims_but_keeps_the_path() -> None:
    assert storable_project("  /Users/x/slm/ ") == "/Users/x/slm/"
    assert storable_project("x" * 5000) == "x" * 1024


def test_project_name_keeps_case_for_display() -> None:
    assert project_name("/Users/x/SuperLocalMemory/") == "SuperLocalMemory"
    assert project_name("/") is None


def test_session_context_query_names_the_project_not_the_path() -> None:
    assert session_context_query("/Users/x/work/acme-billing") == "project context acme-billing"
    assert session_context_query("") == "recent important decisions"
    assert session_context_query("/") == "recent important decisions"


@pytest.mark.parametrize("content,expected", [
    ("[superlocalmemory] session ended 2026-10-04 18:20 | branch: main", "superlocalmemory"),
    ("[testing - automation] session ended 2026-01-02 09:00", "testing - automation"),
    ("[slm] session ended yesterday", None),
    ("note: [slm] session ended 2026-10-04 18:20", None),
    ("[] session ended 2026-10-04 18:20", None),
    ("[/] session ended 2026-10-04 18:20", None),
    ("Plain memory.", None),
    (None, None),
])
def test_session_end_prefix(content, expected) -> None:
    assert session_end_project(content) == expected
