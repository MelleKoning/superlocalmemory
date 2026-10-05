# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Run one of N fixed parts of the suite: ``-p tests._shard_plugin --shard 1/3``.

A test's part is a SHA-256 of its node id, so it never depends on collection
order, the machine, or which other tests exist: the same test lands in the
same part on every run, and the N parts together are the whole suite, each
test exactly once. CI uses it to split a suite too long for one job, so a
timeout in one part cannot hide the failures of the others.
"""

from __future__ import annotations

import hashlib

import pytest


def shard_of(nodeid: str, count: int) -> int:
    """The part (0-based) that ``nodeid`` belongs to, out of ``count``."""
    digest = hashlib.sha256(nodeid.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count


def parse_shard(value: str) -> tuple[int, int]:
    """``"2/3"`` -> ``(1, 3)``: the 0-based part and the number of parts."""
    try:
        part, count = (int(x) for x in value.split("/", 1))
    except ValueError:
        raise pytest.UsageError(f"--shard takes PART/COUNT, like 1/3; got {value!r}") from None
    if count < 1 or not 1 <= part <= count:
        raise pytest.UsageError(f"--shard {value}: PART must be between 1 and COUNT")
    return part - 1, count


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--shard", default=None, metavar="PART/COUNT",
                     help="run only part PART of COUNT stable parts of the suite")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    value = config.getoption("--shard")
    if not value:
        return
    part, count = parse_shard(value)
    keep = [item for item in items if shard_of(item.nodeid, count) == part]
    dropped = [item for item in items if shard_of(item.nodeid, count) != part]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
    items[:] = keep
